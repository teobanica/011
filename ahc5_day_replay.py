#!/usr/bin/env python3
'AHC U_5 daytime replay; standalone Python + NumPy.\n\nRun python3 ahc5_day_replay.py\nDefault 7 cumulative active hours, two workers, one BLAS thread each.\nChoose --hours 6 or --hours 8. Ctrl-C saves; repeat the command to resume.\nDistinct ahc5-day-replay-out directory; share.zip is the file to send back.\nEvery unresolved run ID is retained. Data size is capped; review flags stop safely.\n--self-test runs checks; --batches 8 is a small cumulative test.\n'
MODE = 'stationary'
DEFAULT_SEED = 521812
VERSION = 'ahc5-1'
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import heapq
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import sqlite3
import sys
import time
import zipfile

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

N, BATCH = 5, 16
DIM, HDIM = N*N, (N-1)**2
FLAT, LMAX = 1/np.sqrt(N), N*np.sqrt(N)
EYE = np.eye(N)
BASIS = [np.diag(e).astype(complex) for e in EYE]
for j in range(N):
    for k in range(j+1, N):
        e = np.outer(EYE[j], EYE[k])
        BASIS.extend([(e+e.T)/np.sqrt(2), 1j*(e-e.T)/np.sqrt(2)])
BASIS = np.array(BASIS)
OFF = BASIS[N:]
PRODUCTS = (BASIS[:, None] @ BASIS).reshape(DIM*DIM, N, N)
AX = (-2, -1)
FLAT_TOL, GRAD_TOL, ROOT_TOL = 1e-6, 1e-10, 1e-12
ENTRY_TOL, UNITY_TOL, CURV_TOL = 1e-7, 1e-10, 1e-7
ASCENT_STEPS, ROOT_STEPS, STEER_STEPS = 1600, 160, 32
FLAG_STOP, ERROR_STOP = 128, 32
DB_STOP_BYTES = 112*1024**2
METRIC_NAMES = ('L', 'gradient', 'min_modulus', 'flat_error', 'unitarity_error',
                'top', 'min_X', 'pair_top', 'phase_error')
ASCENT_POLICIES = ('haar', 'phase_polar', 'haar_polar', 'near_saddle', 'near_real', 'near_block')
ROOT_POLICIES = ('haar', 'phase_polar', 'near_real', 'near_flat', 'near_saddle',
                 'near_block', 'bordered', 'circulant', 'cap_090', 'cap_095', 'cap_0985', 'cap_0999')
CAPS = {'cap_090': .90*LMAX, 'cap_095': .95*LMAX,
        'cap_0985': .985*LMAX, 'cap_0999': .999*LMAX}


def adj(a):
    return a.conj().swapaxes(-2, -1)


def haar(rng, size, n=N, real=False):
    z = rng.normal(size=(size, n, n))
    if not real:
        z = z+1j*rng.normal(size=z.shape)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=-2, axis2=-1)
    return (q*(d/abs(d))[:, None, :]).astype(complex)


def fourier(n=N):
    return np.exp(2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)


def components(u):
    s = u/np.maximum(abs(u), 1e-300)
    x = adj(s) @ u
    return s, x, (x-adj(x))/(2j)


def coordinates(a, basis=BASIS):
    return np.einsum('...ij,kji->...k', a, basis).real


def horizontal(u):
    left = adj(u)[:, None] @ BASIS[:N-1] @ u[:, None]
    columns = coordinates(left, OFF).swapaxes(1, 2)
    return np.linalg.qr(columns, mode='complete')[0][:, :, N-1:]


def expstep(u, b):
    eig, v = np.linalg.eigh(b)
    return u @ ((v*np.exp(1j*eig)[..., None, :]) @ adj(v))


def polar(u):
    left, _, right = np.linalg.svd(u)
    return left @ right


def bordered(x, phases):
    """Unitary border and a circulant 4x4 core with eigenvalues -x, phases."""
    size = len(x); out = np.empty((size, N, N), complex)
    b = np.sqrt(1-x*x)/2
    out[:, 0, 0] = x; out[:, 0, 1:] = b[:, None]; out[:, 1:, 0] = b[:, None]
    f = fourier(4)
    vals = np.column_stack((-x, phases))
    out[:, 1:, 1:] = (f[None]*vals[:, None, :]) @ f.conj().T
    return out


def calibration():
    lo, hi = .53, .55
    for _ in range(55):
        x = (lo+hi)/2
        val = 3-4*x/np.sqrt(1-x*x)-2*(1-x)/np.sqrt((1-x)**2+4)
        if val > 0:
            lo = x
        else:
            hi = x
    x = (lo+hi)/2; b = np.sqrt(1-x*x)/2; a = (1+x)/4; c = (1-x)/4
    u = np.full((N, N), b, complex); u[0, 0] = x
    row = np.array([-a, c-.5j, -a, c+.5j])
    for j in range(4):
        u[j+1, 1:] = np.roll(row, j)
    return u


def hessian_matrices(u):
    s, x, _ = components(u)
    w = ((u[:, None] @ BASIS)*s[:, None].conj()).real
    w = (w/np.sqrt(abs(u))[:, None]).reshape(-1, DIM, DIM)
    y = (x+adj(x))/2
    h = w @ w.swapaxes(1, 2)-coordinates(y, PRODUCTS).reshape(-1, DIM, DIM)
    h = (h+h.swapaxes(1, 2))/2
    c = horizontal(u)
    red = c.swapaxes(1, 2) @ h[:, N:, N:] @ c
    return h, red, c


def root_jacobian(u):
    """Exact derivative of R=(S*U-U*S)/(2i) in off-diagonal coordinates."""
    s, x, _ = components(u)
    vb = u[:, None] @ OFF
    radial = (vb*s[:, None].conj()).real/abs(u[:, None])
    dx = 1j*(x[:, None] @ OFF-adj(s[:, None]*radial) @ u[:, None])
    return coordinates((dx-adj(dx))/(2j)).swapaxes(1, 2)


def metrics(u):
    return {'L': float(abs(u).sum()), 'gradient': float(np.linalg.norm(components(u)[2])),
            'min_modulus': float(abs(u).min()), 'flat_error': float(abs(abs(u)-FLAT).max()),
            'unitarity_error': float(np.linalg.norm(adj(u) @ u-EYE))}


def pair_top(u):
    answer = -np.inf
    for v in (u, u.T):
        r = abs(v); s = v/r
        for a in range(N):
            for b in range(a+1, N):
                p, q = r[:, a], r[:, b]
                d = q*q/p+p*p/q; h = s[:, a].conj()*s[:, b]
                answer = max(answer, (d.sum()+abs(np.sum(d*h*h)))/4-(p+q).sum()/2)
    return float(answer)


def spectral_data(u):
    h, red, c = hessian_matrices(u[None])
    full = np.linalg.eigvalsh(h[0]); vals, vecs = np.linalg.eigh(red[0])
    direction = np.einsum('a,aij->ij', c[0] @ vecs[:, -1], OFF)
    left = adj(u) @ BASIS[:N-1] @ u
    phase_coords = np.concatenate((np.eye(DIM)[:N], coordinates(left)))
    phase_error = float(np.linalg.norm(h[0] @ phase_coords.T))
    # The left-phase vectors must span four independent directions after right phases.
    gap = float(np.linalg.svd(coordinates(left, OFF), compute_uv=False)[-1])
    x = components(u)[1]; minimum = float(np.linalg.eigvalsh((x+adj(x))/2)[0])
    return {'full': full, 'horizontal': vals, 'witness': direction,
            'top': float(vals[-1]), 'min_X': minimum, 'pair_top': pair_top(u),
            'phase_error': phase_error, 'phase_gap': gap,
            'positive': int(np.sum(vals > CURV_TOL)),
            'zero': int(np.sum(abs(vals) <= CURV_TOL))}


def perturb(rng, u, low, high):
    w = rng.normal(size=(len(u), DIM))
    w *= (10**rng.uniform(low, high, len(u))/np.linalg.norm(w, axis=1))[:, None]
    return expstep(u, np.einsum('bk,kij->bij', w, BASIS))


def starts_for(seed, batch_id):
    rng = np.random.default_rng(np.random.SeedSequence([seed, batch_id]))
    policies = ASCENT_POLICIES if MODE == 'ascent' else ROOT_POLICIES
    family = policies[batch_id % len(policies)]
    if family == 'phase_polar':
        u = polar(np.exp(2j*np.pi*rng.random((BATCH, N, N))))
    elif family == 'near_real':
        u = perturb(rng, haar(rng, BATCH, real=True), -2, -.05)
    elif family == 'near_flat':
        u = perturb(rng, np.tile(fourier(), (BATCH, 1, 1)), -3, -.05)
    elif family == 'near_saddle':
        u = perturb(rng, np.tile(calibration(), (BATCH, 1, 1)), -4, -.05)
    elif family == 'near_block':
        n = 1 if (batch_id//len(policies)) % 2 == 0 else 2
        u = np.zeros((BATCH, N, N), complex)
        u[:, :n, :n] = haar(rng, BATCH, n=n)
        u[:, n:, n:] = haar(rng, BATCH, n=N-n)
        u = perturb(rng, u, -2.3, -.05)
    elif family == 'bordered':
        u = bordered(rng.uniform(.05, .95, BATCH), np.exp(2j*np.pi*rng.random((BATCH, 3))))
        u = perturb(rng, u, -2, -.4)
    elif family == 'circulant':
        f = fourier()
        u = (f[None]*np.exp(2j*np.pi*rng.random((BATCH, 1, N)))) @ f.conj().T
        u = perturb(rng, u, -2, -.2)
    else:
        u = haar(rng, BATCH)
    return family, u, CAPS.get(family)


def ascent(starts, use_polar=False, limit=ASCENT_STEPS):
    u = starts.copy(); n = len(u); steps = np.zeros(n, int)
    scale = np.full(n, .4); active = np.ones(n, bool)
    for _ in range(limit):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        v = u[ids]; _, _, r = components(v)
        g2 = np.sum(abs(r)**2, axis=AX)
        go = ((g2 > ROOT_TOL**2) & (abs(abs(v)-FLAT).max(axis=AX) > FLAT_TOL)
              & (abs(v).min(axis=AX) > 1e-13))
        active[ids[~go]] = False; ids, v, r, g2 = ids[go], v[go], r[go], g2[go]
        if not len(ids):
            break
        # sum((|u|-1/sqrt(N))^2)=2N-2L/sqrt(N); convert to L deficit.
        deficit = (np.sqrt(N)/2)*np.sum((abs(v)-FLAT)**2, axis=AX)
        if use_polar:
            proposal = polar(components(v)[0])
            new = (np.sqrt(N)/2)*np.sum((abs(proposal)-FLAT)**2, axis=AX)
            good = new < deficit
            u[ids[good]] = proposal[good]; steps[ids[good]] += 1
            active[ids[~good]] = False
            continue
        pending = np.arange(len(ids))
        for _ in range(24):
            if not len(pending):
                break
            rows = ids[pending]; t = scale[rows]
            proposal = expstep(v[pending], -t[:, None, None]*r[pending])
            new = (np.sqrt(N)/2)*np.sum((abs(proposal)-FLAT)**2, axis=AX)
            good = deficit[pending]-new >= 1e-4*t*g2[pending]
            if good.any():
                accepted = rows[good]; q = pending[good]
                u[accepted] = proposal[good]; steps[accepted] += 1
                nr = components(proposal[good])[2]
                curvature = np.sum((r[q].conj()*(r[q]-nr)).real, axis=AX)
                scale[accepted] = np.clip(t[good]*g2[q]/np.maximum(curvature, 1e-25), .02, 80.)
            scale[rows[~good]] *= .5
            pending = pending[~good]
        active[ids[pending]] = False
    return u, steps


def refine(starts, cap=None, limit=ROOT_STEPS, tolerance=ROOT_TOL):
    u = starts.copy(); steps = np.zeros(len(u), int); damping = np.full(len(u), .01)
    for _ in range(limit):
        _, _, r = components(u); score = np.sum(abs(r)**2, axis=AX)
        if cap is not None:
            penalty = 3*np.maximum(0, abs(u).sum(axis=AX)-cap); score += penalty**2
        ids = np.flatnonzero((score > tolerance**2) & (abs(u).min(axis=AX) > 1e-13)
                             & (damping < 1e7))
        if not len(ids):
            break
        v = u[ids]; c = horizontal(v)
        jac = root_jacobian(v) @ c; rhs = coordinates(r[ids])
        if cap is not None:
            dl = -(rhs[:, None, N:] @ c)
            jac = np.concatenate((jac, 3*dl*(penalty[ids] > 0)[:, None, None]), axis=1)
            rhs = np.column_stack((rhs, penalty[ids]))
        left, sv, right = np.linalg.svd(jac, full_matrices=False)
        factor = sv/(sv**2+damping[ids, None]**2)
        step = -(right.swapaxes(1, 2) @ ((left.swapaxes(1, 2) @ rhs[..., None])*factor[..., None]))
        coord = (c @ step)[..., 0]
        coord /= np.maximum(1, np.linalg.norm(coord, axis=1, keepdims=True)/.65)
        proposal = expstep(v, np.einsum('ba,aij->bij', coord, OFF))
        new = np.sum(abs(components(proposal)[2])**2, axis=AX)
        if cap is not None:
            new += (3*np.maximum(0, abs(proposal).sum(axis=AX)-cap))**2
        good = new < score[ids]
        u[ids[good]] = proposal[good]
        damping[ids] = np.where(good, np.maximum(1e-13, damping[ids]*.45), damping[ids]*4)
        steps[ids] += 1
    return u, steps


def classify(m, spec=None):
    if not np.isfinite(list(m.values())).all():
        return 'numeric_error'
    if m['unitarity_error'] > UNITY_TOL:
        return 'unitarity_failure'
    if m['flat_error'] <= FLAT_TOL:
        return 'near_flat'
    if m['min_modulus'] <= ENTRY_TOL:
        return 'near_zero'
    if m['gradient'] > GRAD_TOL:
        return 'unresolved'
    if spec is None or spec['phase_gap'] < 1e-10 or spec['phase_error'] > 1e-7:
        return 'hessian_uncertain'
    return 'candidate' if spec['top'] <= CURV_TOL else 'nonflat_uphill'


def run_batch(seed, batch_id):
    family, starts, cap = starts_for(seed, batch_id)
    rows = []
    def block_solve(block, offset):
        try:
            if MODE == 'ascent':
                raw, iters = ascent(block, family == 'haar_polar'); warm = np.zeros(len(block), int)
            else:
                mid, warm = (refine(block, cap, STEER_STEPS) if cap is not None
                             else (block.copy(), np.zeros(len(block), int)))
                raw, iters = refine(mid)
            for k, endpoint in enumerate(raw):
                idx = batch_id*BATCH+offset+k
                original = metrics(endpoint)
                v = endpoint.copy(); polishing = 0
                # A new stationary endpoint after polishing is kept distinct from the raw endpoint.
                if (original['flat_error'] > FLAT_TOL and original['min_modulus'] > ENTRY_TOL
                        and original['gradient'] < 1e-5 and original['unitarity_error'] <= UNITY_TOL):
                    polished, extra = refine(v[None], limit=80, tolerance=ROOT_TOL)
                    if metrics(polished[0])['gradient'] <= original['gradient']:
                        v = polished[0]; polishing = int(extra[0])
                m = metrics(v); spec = None
                if (m['flat_error'] > FLAT_TOL and m['min_modulus'] > ENTRY_TOL
                        and m['gradient'] <= GRAD_TOL and m['unitarity_error'] <= UNITY_TOL):
                    spec = spectral_data(v)
                status = classify(m, spec)
                info = dict(m, id=idx, batch_id=batch_id, offset=offset+k, family=family,
                            status=status, steering_steps=int(warm[k]), solver_steps=int(iters[k]),
                            polish_steps=polishing, raw=original)
                if spec is not None:
                    info.update({key: spec[key] for key in ('top', 'min_X', 'pair_top', 'phase_error',
                                                          'phase_gap', 'positive', 'zero')})
                rows.append({'info': info, 'start': block[k], 'raw': endpoint, 'endpoint': v,
                             'full': None if spec is None else spec['full'],
                             'horizontal': None if spec is None else spec['horizontal'],
                             'witness': None if spec is None else spec['witness']})
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            # Recompute a failed block individually; discard partial results from this block.
            ids = set(range(batch_id*BATCH+offset, batch_id*BATCH+offset+len(block)))
            rows[:] = [r for r in rows if r['info']['id'] not in ids]
            if len(block) > 1:
                for j in range(len(block)):
                    block_solve(block[j:j+1], offset+j)
            else:
                rows.append({'info': {'id': batch_id*BATCH+offset, 'batch_id': batch_id,
                                     'offset': offset, 'family': family, 'status': 'numeric_error',
                                     'error': str(exc)}, 'start': block[0], 'raw': None, 'endpoint': None,
                             'full': None, 'horizontal': None, 'witness': None})
    block_solve(starts, 0)
    if len(rows) != BATCH or len({r['info']['id'] for r in rows}) != BATCH:
        raise RuntimeError('Incomplete batch; campaign not advanced')
    return sorted(rows, key=lambda r: r['info']['id'])


def self_check():
    u = calibration(); m = metrics(u); spec = spectral_data(u)
    full, reduced = spec['full'], spec['horizontal']
    if (max(m['gradient'], m['unitarity_error']) > 2e-13
            or tuple((np.sum(full < -CURV_TOL), np.sum(abs(full) <= CURV_TOL),
                      np.sum(full > CURV_TOL))) != (15, 9, 1)
            or np.sum(reduced > CURV_TOL) != 1 or np.sum(reduced < -CURV_TOL) != 15
            or not -.197859 < spec['pair_top'] < -.197858):
        raise RuntimeError('Exact order-five saddle calibration failed')
    fs = spectral_data(fourier())
    if fs['horizontal'][-1] > -1e-6 or np.sum(abs(fs['full']) < 1e-8) != 9:
        raise RuntimeError('Fourier calibration failed')
    rng = np.random.default_rng(55); v = haar(rng, 1)
    b = rng.normal(size=(N, N))+1j*rng.normal(size=(N, N)); b = (b+adj(b))/2
    b /= np.linalg.norm(b); step = 1e-4
    plus, minus = [abs(expstep(v, sign*step*b[None])).sum() for sign in (1, -1)]
    phase, x, r = components(v[0]); first = -np.trace(r @ b).real
    second = np.sum(((v[0] @ b*phase.conj()).real**2)/abs(v[0]))-np.trace(x @ b @ b).real
    h, _, _ = hessian_matrices(v); bc = coordinates(b)
    if (abs((plus-minus)/(2*step)-first) > 3e-6
            or abs((plus+minus-2*abs(v).sum())/step**2-second) > 5e-5
            or abs(bc @ h[0] @ bc-second) > 1e-11):
        raise RuntimeError('Gradient / Hessian finite-difference test failed')
    # Independently differentiate every column of the root Jacobian.
    jac = root_jacobian(v)[0]; errors = []
    for k, b in enumerate(OFF):
        delta = 1e-6
        rp = components(expstep(v, delta*b[None]))[2][0]
        rm = components(expstep(v, -delta*b[None]))[2][0]
        errors.append(np.max(abs(coordinates((rp-rm)/(2*delta))-jac[:, k])))
    if max(errors) > 1e-7:
        raise RuntimeError('Stationarity Jacobian finite-difference test failed')
    # Calibration is still recognized after phases and permutations.
    vv = u[rng.permutation(N)][:, rng.permutation(N)]
    vv = np.exp(2j*np.pi*rng.random(N))[:, None]*vv*np.exp(2j*np.pi*rng.random(N))[None]
    if np.max(abs(spectral_data(vv)['horizontal']-reduced)) > 1e-10:
        raise RuntimeError('Phase / permutation invariance check failed')
    return {'version': VERSION, 'L': m['L'], 'gradient': m['gradient'],
            'inertia_full': [15, 9, 1], 'top_horizontal': spec['top'],
            'best_pair': spec['pair_top'], 'fourier_top_horizontal': fs['top'],
            'max_jacobian_error': max(errors), 'passed': True}

DDL = '''
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY,utc TEXT,environment TEXT);
CREATE TABLE IF NOT EXISTS aggregates(family TEXT,status TEXT,data TEXT,PRIMARY KEY(family,status));
CREATE TABLE IF NOT EXISTS histogram(family TEXT,bin INTEGER,positive INTEGER,n INTEGER,
 PRIMARY KEY(family,bin,positive));
CREATE TABLE IF NOT EXISTS kept(id INTEGER PRIMARY KEY,info TEXT,start BLOB,raw BLOB,
 endpoint BLOB,full BLOB,horizontal BLOB,witness BLOB);
CREATE TABLE IF NOT EXISTS slots(pool TEXT,slot INTEGER,score REAL,run_id INTEGER,
 PRIMARY KEY(pool,slot));
CREATE INDEX IF NOT EXISTS slots_run ON slots(run_id);
'''


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_text(value):
    return json.dumps(clean(value), sort_keys=True, allow_nan=False)


def set_meta(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json_text(value)))


def get_meta(db, key, default=None):
    row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return default if row is None else json.loads(row[0])


def pack(a, complex_array=True):
    return None if a is None else np.asarray(a, dtype='<c16' if complex_array else '<f8').tobytes()


def update_aggregate(old, info):
    old['n'] = old.get('n', 0)+1
    for key in METRIC_NAMES+('raw_gradient', 'raw_flat_error'):
        val = info.get(key) if not key.startswith('raw_') else info.get('raw', {}).get(key[4:])
        if val is None or not math.isfinite(val):
            continue
        entry = old.setdefault(key, {'min': val, 'max': val, 'sum': 0., 'n': 0})
        entry['min'] = min(entry['min'], val); entry['max'] = max(entry['max'], val)
        entry['sum'] += val; entry['n'] += 1
    return old


class Retention:
    """Fixed-size pools. Counters cover all endpoints, while matrices are selected."""
    def __init__(self, db, seed):
        self.db, self.seed, self.pools = db, seed, {}
        for pool, slot, score, idx in db.execute('SELECT pool,slot,score,run_id FROM slots'):
            heapq.heappush(self.pools.setdefault(pool, []), (-score, idx, slot))
        self.patterns = sum(k.startswith('pattern:') for k in self.pools)

    def offer(self, pool, capacity, score, row):
        heap = self.pools.setdefault(pool, [])
        idx = row['info']['id']
        if len(heap) == capacity and score >= -heap[0][0]:
            return
        old = None
        if len(heap) < capacity:
            slot = len(heap)
        else:
            _, old, slot = heapq.heappop(heap)
        self.db.execute('INSERT OR REPLACE INTO kept VALUES (?,?,?,?,?,?,?,?)',
                        (idx, json_text(row['info']), pack(row['start']), pack(row['raw']),
                         pack(row['endpoint']), pack(row['full'], False),
                         pack(row['horizontal'], False), pack(row['witness'])))
        self.db.execute('INSERT OR REPLACE INTO slots VALUES (?,?,?,?)', (pool, slot, score, idx))
        heapq.heappush(heap, (-score, idx, slot))
        if old is not None:
            self.db.execute('DELETE FROM kept WHERE id=? AND NOT EXISTS '
                            '(SELECT 1 FROM slots WHERE run_id=?)', (old, old))

    def consider(self, row):
        m = row['info']; idx = m['id']; status = m['status']
        key = f"{self.seed}:{idx}".encode()
        priority = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big')/2**64
        self.offer('sample:'+m['family']+':'+status, 16, priority, row)
        if status in ('candidate', 'hessian_uncertain'):
            # Controller stops at 128 flags, after retaining all in-flight batches too.
            self.offer('flags', 1000000, float(idx), row)
        elif status in ('numeric_error', 'unitarity_failure'):
            self.offer('errors', 1000000, float(idx), row)
        elif status == 'unresolved' and m.get('min_modulus', 0) > ENTRY_TOL:
            self.offer('best_unresolved', 128, m['gradient'], row)
        if status in ('candidate', 'nonflat_uphill'):
            self.offer('least_unstable', 128, m['top'], row)
            self.offer('index:'+str(m['positive']), 16, m['top'], row)
            # A grouping fingerprint, NOT a test of mathematical equivalence.
            data = np.r_[np.sort(abs(row['endpoint']).ravel()).round(5),
                         row['horizontal'].round(4)].astype('<f8').tobytes()
            pattern = 'pattern:'+hashlib.sha256(data).hexdigest()[:24]
            if pattern in self.pools or self.patterns < 256:
                if pattern not in self.pools:
                    self.patterns += 1
                self.offer(pattern, 1, m['gradient'], row)


def commit_batch(db, retention, idx, rows, elapsed):
    if idx != get_meta(db, 'next_batch', 0):
        raise RuntimeError('Out-of-order commit refused')
    groups = {}
    hist = Counter()
    with db:
        for row in rows:
            m = row['info']; key = (m['family'], m['status'])
            if key not in groups:
                old = db.execute('SELECT data FROM aggregates WHERE family=? AND status=?', key).fetchone()
                groups[key] = {} if old is None else json.loads(old[0])
            update_aggregate(groups[key], m)
            if m['status'] in ('candidate', 'nonflat_uphill'):
                hist[(m['family'], int(math.floor(1000*m['L']/LMAX)), m['positive'])] += 1
            retention.consider(row)
        db.executemany('INSERT OR REPLACE INTO aggregates VALUES (?,?,?)',
                       [(fam, status, json_text(data)) for (fam, status), data in groups.items()])
        db.executemany('INSERT INTO histogram VALUES (?,?,?,?) ON CONFLICT(family,bin,positive) '
                       'DO UPDATE SET n=n+excluded.n', [(*k, n) for k, n in hist.items()])
        set_meta(db, 'next_batch', idx+1); set_meta(db, 'seconds', elapsed)
    for row in rows:
        m = row['info']
        if m['status'] in ('candidate', 'hessian_uncertain'):
            print('REVIEW FLAG:', json_text(m), flush=True)


def atomic_text(path, data):
    pending = path.with_suffix(path.suffix+'.tmp')
    pending.write_text(data, encoding='utf-8'); pending.replace(path)


def report(db, folder, state, elapsed):
    groups = [dict(family=fam, status=status, **json.loads(data)) for fam, status, data in
              db.execute('SELECT family,status,data FROM aggregates ORDER BY family,status')]
    totals = Counter()
    for g in groups:
        totals[g['status']] += g['n']
    retained = db.execute('SELECT count(*) FROM kept').fetchone()[0]
    best = [json.loads(row[0]) for row in db.execute('SELECT info FROM kept k WHERE EXISTS '
            "(SELECT 1 FROM slots s WHERE s.run_id=k.id AND s.pool='least_unstable')")]
    best.sort(key=lambda v: (v.get('top', float('inf')), v['id']))
    config = get_meta(db, 'config')
    total = sum(totals.values()); next_batch = get_meta(db, 'next_batch', 0)
    if total != next_batch*BATCH:
        raise RuntimeError('Aggregate counts do not match committed batches')
    size = db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
    summary = dict(mode=MODE, version=VERSION, state=state, active_hours=elapsed/3600,
                   starts=total, next_batch=next_batch, counts=dict(totals), retained_endpoints=retained,
                   database_bytes=size, config=config, aggregates=groups, least_unstable=best[:20],
                   note='Floating-point exploration. near_flat is numerical; candidate is not a proof. '
                        'All runs counted; ordinary endpoints sampled, not individually archived.')
    line = f'{MODE} | {elapsed/3600:.3f} h | {total:,} starts | '+', '.join(f'{k}={v:,}' for k, v in sorted(totals.items()))
    lines = ['AHC U_5 '+MODE, state, line, f'Retained matrices: {retained}; database: {size/1024**2:.2f} MiB',
             'Seed: '+str(config['seed']), 'Source SHA-256: '+config['sha256'],
             'Stationarity <=1e-10; small-entry flag <=1e-7; flat error <=1e-6; curvature threshold 1e-7.',
             'All runs counted; ordinary matrices are sampled. Candidate flags require independent review.',
             'An endpoint changed by polishing has separate raw and polished matrices in the archive.',
             '', 'family                 status                  count']
    for g in groups:
        lines.append(f"{g['family']:<22} {g['status']:<22} {g['n']:>10}")
    if best:
        lines += ['', 'Least unstable retained nonflat stationary endpoints:',
                  'id / family / L / gradient / min entry / top horizontal / positive directions / best pair']
        for m in best[:20]:
            lines.append(f"{m['id']} {m['family']} {m['L']:.12f} {m['gradient']:.3g} "
                         f"{m['min_modulus']:.5g} {m['top']:.9g} {m['positive']} {m['pair_top']:.8g}")
    lines += ['', 'To report: send summary.txt first; share.zip contains the compact database and exact source.']
    atomic_text(folder/'summary.txt', '\n'.join(lines)+'\n')
    atomic_text(folder/'summary.json', json.dumps(clean(summary), indent=2)+'\n')
    print(state+' | '+line+f' | {size/1024**2:.2f} MiB', flush=True)
    return summary


def source_config(args):
    return dict(mode=MODE, version=VERSION, seed=args.seed, batch_size=BATCH,
                sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                flat_tol=FLAT_TOL, grad_tol=GRAD_TOL, entry_tol=ENTRY_TOL, curvature_tol=CURV_TOL)


def ignore_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def campaign(args, db, folder, calibration_report):
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA max_page_count=32768')  # 128 MiB with the default 4096-byte pages.
    db.executescript(DDL)
    config = source_config(args); previous = get_meta(db, 'config')
    if previous is not None and previous != config:
        raise SystemExit('Source or seed differs from this campaign. Use a new --out folder.')
    with db:
        set_meta(db, 'config', config)
        set_meta(db, 'calibration', calibration_report)
        environment = dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                           workers=args.workers, blas_threads=1, policy_list=ASCENT_POLICIES if MODE=='ascent' else ROOT_POLICIES)
        db.execute('INSERT INTO sessions(utc,environment) VALUES (?,?)',
                   (datetime.now(timezone.utc).isoformat(), json_text(environment)))
    (folder/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    atomic_text(folder/'calibration.json', json.dumps(clean(calibration_report), indent=2)+'\n')
    base = get_meta(db, 'seconds', 0.); began = time.monotonic()
    elapsed = lambda: base+time.monotonic()-began
    stop_at = args.hours*3600
    retention = Retention(db, args.seed)
    if args.summarize or base >= stop_at:
        report(db, folder, 'Summary only' if args.summarize else 'Requested time already reached', base)
        return 'finished'
    flags = len(retention.pools.get('flags', []))+sum(k.startswith('novelty:') for k in retention.pools); errors = len(retention.pools.get('errors', []))
    if flags >= FLAG_STOP or errors >= ERROR_STOP:
        report(db, folder, 'Review required before a new campaign', base)
        return 'review'
    stopped = False; reason = 'Finished'
    def stop(signum, frame):
        nonlocal stopped, reason
        stopped, reason = True, 'Paused safely'
    old_int, old_term = signal.signal(signal.SIGINT, stop), signal.signal(signal.SIGTERM, stop)
    print('Output:', folder.resolve(), '| workers:', args.workers,
          '| Ctrl-C saves after active batches | progress every minute', flush=True)
    report(db, folder, 'Starting/resuming', base)
    next_commit = get_meta(db, 'next_batch', 0); cursor = next_commit
    ready, jobs = {}, {}; last_report = began
    try:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=ignore_interrupt) as pool:
            def fill():
                nonlocal cursor
                while (not stopped and elapsed() < stop_at and len(jobs)+len(ready) < args.workers
                       and (args.batches is None or cursor < args.batches)):
                    jobs[pool.submit(run_batch, args.seed, cursor)] = cursor
                    cursor += 1
            fill()
            while jobs or ready:
                done, _ = wait(jobs, timeout=2., return_when=FIRST_COMPLETED) if jobs else ([], [])
                for task in done:
                    idx = jobs.pop(task); ready[idx] = task.result()
                while next_commit in ready:
                    commit_batch(db, retention, next_commit, ready.pop(next_commit), elapsed())
                    next_commit += 1
                flags = len(retention.pools.get('flags', []))+sum(k.startswith('novelty:') for k in retention.pools); errors = len(retention.pools.get('errors', []))
                pages = db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
                if flags >= FLAG_STOP or errors >= ERROR_STOP or pages >= DB_STOP_BYTES:
                    stopped, reason = True, 'Stopped for review/storage limit; all completed flags retained'
                if time.monotonic()-last_report >= 60:
                    with db:
                        set_meta(db, 'seconds', elapsed())
                    report(db, folder, 'Running', elapsed()); last_report = time.monotonic()
                fill()
    except Exception as exc:
        reason = 'Stopped on error: '+type(exc).__name__+': '+str(exc)
        # Only fully committed batches count; missing batches replay on restart.
        print(reason, flush=True)
    finally:
        total = elapsed()
        with db:
            set_meta(db, 'seconds', total)
        report(db, folder, reason, total)
        signal.signal(signal.SIGINT, old_int); signal.signal(signal.SIGTERM, old_term)
    return 'error' if reason.startswith('Stopped on error') else 'finished'


def share_archive(folder):
    with closing(sqlite3.connect('file:'+str((folder/'results.sqlite').resolve())+'?mode=ro', uri=True)) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('Database integrity check failed')
    target = folder/'share.zip'; temporary = folder/'share.zip.tmp'
    names = ['summary.txt', 'summary.json', 'calibration.json', 'results.sqlite', Path(__file__).name]
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for name in names:
            z.write(folder/name, name)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise RuntimeError('Share archive verification failed')
    temporary.replace(target)
    print('Saved', target.resolve(), f'({target.stat().st_size/1024**2:.2f} MiB)', flush=True)


def inspect_or_replay(args, folder):
    cfg = None
    if (folder/'results.sqlite').exists():
        with closing(sqlite3.connect('file:'+str((folder/'results.sqlite').resolve())+'?mode=ro', uri=True)) as db:
            cfg = get_meta(db, 'config')
            if args.inspect is not None:
                row = db.execute('SELECT info,endpoint FROM kept WHERE id=?', (args.inspect,)).fetchone()
                if row is None:
                    raise SystemExit('This endpoint was not retained. Use --replay with its run ID.')
                print(json.dumps(json.loads(row[0]), indent=2))
                if row[1] is not None:
                    print(np.frombuffer(row[1], dtype='<c16').reshape(N, N))
                return
    if args.inspect is not None:
        raise SystemExit('No results database found in this output folder.')
    if cfg is not None and cfg != source_config(args):
        raise SystemExit('Replay requires the same seed and source as the stored campaign.')
    rows = run_batch(args.seed, args.replay//BATCH)
    row = rows[args.replay % BATCH]
    print(json.dumps(clean(row['info']), indent=2)); print(row['endpoint'])
    print('Replay is not counted as a new campaign start; floating-point paths depend on the environment.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hours', type=float, default=7., help='cumulative active campaign hours (default 7)')
    p.add_argument('--workers', type=int, default=3, help='worker processes (default 3)')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/('ahc5-'+MODE+'-out'))
    p.add_argument('--batches', type=int, help='optional cumulative batch limit, 16 starts per batch')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--summarize', action='store_true', help='refresh reports and share.zip without searching')
    p.add_argument('--inspect', type=int, metavar='RUN_ID')
    p.add_argument('--replay', type=int, metavar='RUN_ID')
    args = p.parse_args()
    if (not math.isfinite(args.hours) or args.hours <= 0 or args.workers < 1 or args.seed < 0
            or (args.batches is not None and args.batches < 1)
            or any(v is not None and v < 0 for v in (args.inspect, args.replay))):
        p.error('hours, workers and batches must be positive; seed and IDs nonnegative')
    if args.inspect is not None and args.replay is not None:
        p.error('choose either --inspect or --replay')
    calibration_report = self_check()
    print('N=5 saddle (15,9,1), negative pair tests, Fourier, derivatives and invariance: OK.', flush=True)
    if args.self_test:
        print(json.dumps(clean(calibration_report), indent=2)); return
    folder = args.out.resolve()
    if args.inspect is not None or args.replay is not None:
        inspect_or_replay(args, folder); return
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('another process is using this output folder')
        with closing(sqlite3.connect(folder/'results.sqlite')) as db:
            state = campaign(args, db, folder, calibration_report)
        share_archive(folder)
    if state == 'error':
        raise SystemExit(1)



DAY_MODE='day-replay'
CATALOGUE_REAL=[[[0.17674457709203117, 0.48500251792347765, 0.48458661109608514, 0.4850521012386693, 0.48493164372394704], [0.4850655110583048, 0.3874854124898066, 0.02978227586019092, -0.1180920517507475, -0.47069284285270935], [0.4850058041240561, -0.483832237226102, 0.3849750233390119, 0.03149775666770377, -0.1187952494026989], [0.4850633168619937, -0.11839219304104963, -0.46592224507356816, 0.3944660909466755, 0.030734735980907663], [0.4850712028971723, 0.03079099903254789, -0.11455690088997124, -0.4801595160546161, 0.38678021598937196]], [[-0.6146913433273127, -0.23536459044272384, -0.11808393680528471, -0.0017229913183703395, 0.37216862119116817], [0.37696510630980845, -0.6143900720114192, -0.23554181986723682, -0.11906596206369295, -0.004465157471407756], [-0.003256149288572438, 0.3739706589841895, -0.6130442592785036, -0.23630201850114937, -0.12150893962472156], [-0.11927165228572562, -0.002406366926208798, 0.37532651751286056, -0.6127998295936459, -0.23674580302127785], [-0.23444896738026483, -0.11601509169057059, 0.0023565707091557646, 0.38346519289464276, -0.6149577080814647]], [[0.26289471256380836, -0.13218003631984224, -0.13931324770465714, 0.24022342372511288, -0.4384531229813604], [0.4405495410157283, 0.5666057179412947, 0.21089602791773074, 0.4469326574462179, 0.04087209500832983], [0.5552236034223089, 0.14602320094610063, 0.1350388778125404, -0.384903676960258, -0.4002774052646046], [-0.238034591585854, 0.1535691353155974, 0.44315308793440195, -0.23892814568830595, -0.497824824108413], [0.22489979490448797, -0.49671658076100245, 0.51275685239443, 0.22672618451760926, 0.15427928127256388]], [[-0.2416644702736415, -0.2567359359331209, -0.11567319697798698, -0.5599867481704692, 0.35201575164785043], [-0.17626787224158144, 0.2484284022041136, -0.12276399954318579, -0.0950126878402546, 0.13712050244311563], [-0.58489807422125, -0.18931344822518556, 0.08795668192810785, -0.07807254748756795, 0.22870980609804065], [-0.22685823900429541, -0.31465744980658344, -0.2574386924828252, 0.27264504619045143, 0.11201858184087159], [-0.23356656769616355, 0.04484527092421694, 0.1374652722461767, 0.21795832041038482, -0.5487169850887471]], [[0.12817685111277466, -0.2714706455191737, -0.10195272610066021, -0.004115494828144342, 0.20491480472732], [-0.5972878915183323, 0.10795667304115428, 0.15974549323027445, 0.13596038047890574, 0.008657683450483315], [0.06832257296787668, 0.2108820373662369, 0.06367307827750705, 0.3700587436920188, 0.3614022891599957], [0.18949482032480128, -0.03536093806698631, -0.06157420920077233, -0.4262881242558953, 0.5502097754432475], [0.026524443013342253, 0.2118675978021969, -0.04986275599752938, -0.49919390898239346, 0.18621048016551417]], [[-0.5317114540807579, 0.20473149121370293, -0.033384793873835535, 0.5640277321988204, -0.07316941381399922], [-0.340918321569542, -0.5980604152532151, -0.15197154971193041, -0.14633273465051047, -0.04708318887207785], [-0.42986238170047236, -0.158874792874405, -0.22947378316273967, -0.14225934142908453, -0.5733157627538089], [0.07783428309769885, -0.6161950300311747, 0.07343122135734804, 0.5662215906181854, 0.23884291001674185], [-0.5316511640244003, 0.1472291380940603, 0.27935623159435025, -0.10913584068667649, 0.46317902915283066]], [[0.3050091946118159, 0.22439142730923914, 0.18663367362568598, -0.5330935063570905, 0.07138580160359256], [0.17021489218903568, 0.3131914034868965, -0.4006278268326339, 0.08497851100941818, -0.2092151034420242], [0.04419218799909982, -0.009040121854415545, -0.23011308361442728, 0.19783430196479596, -0.21290335611181344], [0.22091090834972094, 0.04788569341747217, -0.08384555291852876, -0.03739247919262242, 0.595228047804032], [-0.14771561001883776, 0.3483649003410469, -0.09121972142023352, -0.1057265961090679, -0.08942926179285784]], [[0.5239541584201615, 0.15613254175153365, -0.5887026415347697, -0.15464554618868814, -0.1618098181851588], [0.18636468711268514, 0.6230821711516319, 0.1806222275859114, 0.31030405953375334, 0.512935918058278], [0.02241157009024615, 0.09936946809491805, 0.4464853346270309, 0.36615906085813044, -0.10728839994031458], [0.19275561265014154, -0.08665103550222984, 0.36833542077098763, -0.6108727909402896, 0.3821280837510014], [0.3055922650607756, -0.4229949448851043, -0.030745934401804457, 0.22264981685834712, 0.16900601759924622]], [[0.32345816987276194, 0.5156246984697429, 0.388443409402688, 0.6114530819976146, 0.32346061304706386], [-0.3232091364256736, -0.5151434493235734, 0.6114163118856665, 0.38837834093539564, -0.3233100293537178], [-0.5156464767227131, 0.6770740803215671, 0.06977933159369579, -0.06979633494346253, -0.5156447991880013], [0.388532234438248, -0.06979479666012799, -0.4847912536853382, 0.4849031941540422, -0.6114582528901791], [0.6114021536308377, 0.06978557234580098, 0.4848709134435484, -0.48491012242796316, -0.38853145455219357]], [[-0.07678277137520707, 0.14208129121036173, 0.11294461011859343, 0.523285507193509, -0.24328695173894108], [-0.024316736002579928, 0.5670852183772352, 0.07828555362591834, -0.343068907540326, -0.19135438633509513], [0.19786463695769102, 0.24797405521186866, -0.5778866533548723, -0.3881928821029091, -0.08394488498532775], [-0.5621911079166743, 0.4167178122816957, -0.26090245562056835, 0.13432464412376077, -0.17514685948133452], [0.2292855973310538, 0.08195661864701281, -0.16611335866244664, -0.08027549844736927, 0.5950834140160701]], [[-0.035387430392637684, -0.6055390597235887, -0.16420316969494061, -0.6332880529685292, -0.23999274278431723], [-0.16692251560991306, -0.2049038087103868, -0.3711079792458749, -0.11447610025674113, 0.08355095490144585], [-0.2489248795185012, -0.11577199875523975, 0.139786540614494, 0.051594144454405244, 0.5912357101745384], [-0.04685053707942182, 0.17562152186459093, 0.44846787085991086, 0.03668266769891843, -0.537763026584392], [0.3105617596054783, 0.016918974149901768, 0.001735982930410394, -0.17428902884317463, 0.24263103574566275]], [[0.1355644793509476, -0.3987358773871719, 0.2441294442107173, -0.44752806633454395, -0.5286251304741516], [-0.5210813124727491, 0.13655006362539815, -0.39636519513375873, 0.24334334059839524, -0.4457935602569562], [-0.44751099383109233, -0.5185183979965234, 0.1355287139902444, -0.3895499355068382, 0.24653945701747934], [0.24467713349582307, -0.44096279396313176, -0.5246610154138441, 0.13541592385630277, -0.38905866303270414], [-0.40503214849849045, 0.2335410654888277, -0.43425930866224327, -0.5102568908952284, 0.13657898902738003]], [[0.10844153755108454, 0.0705324705147339, 0.6229179972528102, -0.26141419905439445, -0.05661544151913499], [-0.21294191261559853, -0.320238183419153, -0.16089445604674887, 0.36675628637293123, 0.2865513262251769], [-0.26769213410307857, 0.11329532461350508, 0.021128377908262493, -0.42444641321612797, -0.13323604363269836], [0.4290666340435885, -0.4623073783423073, -0.26464768536091837, -0.12191763098205041, -0.5377566215160047], [-0.050042917286293724, 0.21380952146530408, 0.45890972055533485, 0.32768465688630216, -0.5012688840578949]], [[0.5771881313883104, 0.31797650640526837, -0.2649293319302911, 0.2756706975872002, 0.25209875225260653], [0.04588521327089472, 0.4942821912083934, -0.36786301343883376, -0.0947909358364371, -0.41342313855919166], [0.2954050304177138, -0.2592268799547577, 0.009006208410603154, 0.5988958686406267, -0.46522971947770553], [-0.2537848954578518, 0.3131997467249334, 0.20205730561138655, -0.2644547753996663, -0.10348989451034678], [-0.3810575340901132, 0.3423352575408054, 0.2204697093692228, 0.5163528801189965, -0.01216590386454863]], [[-0.11731002391633043, -0.36576816706778187, 0.3375604978891725, -0.07444598086732764, -0.1097984119017233], [0.5832421384003862, -0.05494185403909692, 0.11814737033198003, 0.3522966534070409, 0.041339475988034856], [0.3377340938540174, 0.5438680653414969, 0.5851798690704505, -0.2799347868875365, -0.008531246760809345], [0.10832768127460472, -0.49362614351365014, 0.16863766831046276, -0.035238161028538655, -0.29559641767886646], [0.14744082099353184, 0.15152407719186767, 0.05885305444617717, 0.29727217918310567, -0.22985816036912826]], [[-0.17622794927767496, 0.08647271356649186, -0.1763167578376179, 0.5925592515457683, 0.5927600403412436], [0.5923483483325185, -0.1759349838241319, 0.08452378264548226, -0.17554689821012692, 0.5923564517279053], [0.5928725161582619, 0.5930152389561362, -0.1764550900104577, 0.08557254815197039, -0.17637551531484025], [-0.17607698021137708, 0.592764985550907, 0.5926980300629235, -0.1758561666280802, 0.08535284188860293], [0.08733064488106596, -0.17679258714156246, 0.5931739469215861, 0.5929092206675503, -0.17663708347625903]], [[-0.4723698996225419, 0.5296015987500445, -0.24246553238771276, -0.030193500955630934, -0.46960972395674927], [-0.4246867168809405, -0.46947704924542494, 0.4601995990827697, -0.23006049300712078, -0.015506914944034453], [-0.01570715172459146, -0.46071960130333234, -0.5851096845657399, 0.3687129755604407, -0.26178897215210356], [-0.2525453697735078, -0.04783008473840678, -0.3236355372017836, -0.6276006983491239, 0.3650085287012917], [0.46749182252675875, -0.25356563563593676, 0.13408031064016068, -0.3352466343649758, -0.5845739718436974]], [[-0.17511678090199512, 0.07206904943215915, -0.010546512426334852, 0.5215480014854312, -0.2978011985090667], [-0.13697863686081774, 0.3289841678211036, 0.18947159034210478, 0.0947511943405173, -0.559517773977067], [0.25543249870872636, -0.49361485850657816, -0.2606923795101862, 0.5291237340572426, -0.35840851745831076], [-0.12818765286829478, 0.3810604405725911, 0.13665961064607487, -0.06127108748392712, -0.21786962008672692], [-0.09653227841650237, -0.0990070869243299, -0.3372185534130958, -0.15192159063365954, -0.011477595705239289]], [[-0.08486929022098927, 0.4710296412803844, 0.10997992078654838, -0.18451664123065087, 0.17357103236616078], [0.0587158221929204, -0.04401503464295975, 0.34022578063387465, -0.3478291072130442, 0.08710750551511774], [0.026727582042952686, -0.07586985464001238, -0.2018177558299693, 0.6480704047962237, 0.09602542111291622], [0.061157104990124056, 0.06174228073467177, 0.459130357544663, -0.006603629532577619, -0.5569618886925555], [-0.5092137683563679, 0.523622377803366, 0.1714264031152335, 0.2924087312271999, 0.3113121965252715]], [[-0.11888798712230605, -0.1208875758846904, 0.3171081077989491, -0.004045569981110407, -0.11757570128094895], [0.5319154730512898, 0.007196953761137621, -0.4408175107710017, 0.5192446547689589, -0.03132733062611211], [0.22416114659844977, 0.33328465561730247, 0.41771450418893613, -0.1243784700784059, -0.20550477154352773], [-0.39696784509564087, 0.5786823328909635, 0.10008871673704628, 0.49393257540546615, -0.142629499372656], [0.5268707128242773, 0.023580621000289348, 0.45423915834978185, 0.293834287317101, -0.19715315472798325]], [[0.10100681196821715, 0.4225523227270554, -0.19143368642458108, -0.48710822597195724, -0.09757209787690875], [-0.0929243770564973, -0.12833217505297811, -0.3175701867086136, -0.10356919128810468, -0.4279615330092685], [-0.3349267908284619, -0.15257198985501516, 0.14696682556293442, 0.43863473099967576, -0.0256081274578582], [-0.190313541688127, 0.45636740393942143, 0.2467104033561022, 0.3881225516236283, -0.3242560720143086], [0.517497742394463, 0.2342943562289233, 0.322076191073354, -0.15512637296970277, -0.25587727348512224]], [[0.24336317062958784, -0.4879708260981349, 0.12464773436307212, 0.5464884745702072, -0.009401220875581411], [-0.12714412029816996, -0.056679147900163096, -0.4853684456512814, 0.5075454502495815, 0.25833606310828505], [0.4958005907448256, 0.6482239902499729, 0.15826273961646486, 0.10017135665016541, 0.15056452739440798], [0.5177565023346299, -0.21175107352747824, -0.3783002406113391, -0.3636257950120294, 0.5427680397199207], [-0.3864971668476098, 0.06627906087810148, 0.3309545951237235, 0.01809454147496865, 0.5394253269592358]], [[0.5367784041574346, 0.28740291847218186, -0.18761236656167216, -0.0980171772354978, -0.09673305704655819], [-0.2962198760831656, 0.025739748676703537, 0.25152741557581276, 0.5056628609009011, -0.1553644769031234], [-0.15987865247217037, 0.04197717906927823, 0.48331537418563625, -0.06842184332798419, 0.37807129756242475], [0.08447056522917, -0.16339884632446572, -0.3604989347078613, 0.6051707036526741, 0.46337830733988183], [-0.4095016380458153, 0.39542118244008345, -0.1619235817668452, -0.13835037051314783, 0.5416812877781111]], [[0.29030596402238373, 0.024538826730821774, -0.3778987329651842, -0.5208144939939273, -0.3614063712176856], [-0.31447662432966816, -0.1453721108684069, 0.04659928905064409, -0.10985309847619484, 0.15094224019947017], [-0.5580217925730947, -0.49690817812256005, -0.23382536390121433, 0.2516286470198362, -0.16683264550068302], [-0.5620679581221936, 0.3358570195898146, -0.3745147036510153, -0.24771819410452176, 0.17622319371548611], [-0.11967319453672054, 0.3308889485964, 0.20893779752535105, -0.4006934424169281, 0.19717037663908174]], [[-0.6273670537939787, 0.2833465401751617, -0.3931153669356299, 0.30465723872577466, -0.34738377918518853], [-0.25531669230462584, 0.006717610898025373, 0.5249481308318914, 0.5773116802872355, 0.07869610945130272], [0.32842214770330236, -0.6111091910632166, -0.36563159148605834, 0.3860060410406228, -0.2221374066286149], [0.08724252028426316, -0.18839446883320188, -0.009972549864034806, 0.47286930638550134, 0.38501697598634244], [-0.14362540803736262, -0.3505041350014559, 0.5518037575410276, -0.07473075849460593, -0.5831650667499984]], [[-0.3129083312229518, 0.10703032747331068, 0.4246648672964402, -0.2886658702025129, -0.31978970532581874], [-0.37152777354133004, 0.19890071199019896, 0.23572147400907023, 0.3699806840267014, 0.5133390833008261], [-0.23720413498387566, -0.11418855442121091, -0.07017423719299631, 0.4275064674193707, -0.5268637064661837], [-0.09440167707527479, 0.5244775301436769, -0.5904073044365074, -0.3332538161680103, 0.03788027178317634], [-0.08672596322577775, -0.37408305882954884, 0.09682890829229376, -0.3347109471958531, -0.2677354659200969]], [[0.4456490591947852, -0.48188483989813663, 0.5175275661620591, 0.2611930606808526, -0.4813722227745798], [-0.4455785879284866, 0.4819194671781353, 0.4815868759582539, -0.26120098782399526, -0.5175006044275978], [-0.4435741082399702, -0.5173858264724999, 0.4818247288908555, -0.2607431563410902, 0.4818825494349938], [0.6304928176697691, 0.2610557694830435, 0.26074489056623157, -0.6304166748368248, 0.2606547725621122], [0.07643430466075303, 0.44588347885013124, 0.4456306169902239, 0.6306019342303332, 0.44553706837161705]], [[0.0630134186403902, -0.424671258847411, 0.25965502548098707, -0.27248891346752957, -0.1017629393038688], [-0.4517467476399672, -0.3011170087707236, 0.35207469226566057, 0.2182716461691222, 0.020621313309655557], [0.06577932381832804, 0.4760285557239019, -0.16683466642499184, 0.13047341240883728, 0.4914813338710414], [0.031019255613108802, 0.44799317217151313, -0.0928270994052288, -0.17871240339171082, -0.46789852151323585], [0.07744107569102995, 0.16508935171641312, 0.14948201516770543, 0.45220167379717985, 0.262106617802957]], [[0.19858537644227625, 0.48768476789846965, 0.1789698471554703, 0.4625565647112643, -0.11977328053076591], [-0.24809567557785478, 0.08802206511236177, 0.5955596819183414, 0.05188388965378929, -0.5370520882834141], [-0.45520768923315763, 0.45944754073242255, -0.14895254880041095, -0.08234865999647258, 0.13187613921947489], [-0.47124889652548096, -0.2772620402931201, -0.1340638411192054, 0.17543109575088053, -0.24263598989653484], [0.4053853097321713, -0.4077420704721689, -0.05876718228575306, 0.23333078221518455, -0.16217930724247664]], [[-0.24817607064872596, 0.12985308975109308, -0.3873898253659436, -0.06751460093620842, 0.3468332890246777], [0.4235673971172449, -0.2057622194551484, 0.09103079748287488, -0.3766155169718315, 0.35772167294856694], [-0.4066255028909129, 0.3221538478764554, -0.2853526577058702, -0.5601442779886818, 0.11680395362822082], [0.34370088493396006, 0.10950209511794388, -0.23367174046552577, 0.39972275164370547, 0.17978970828360047], [0.44286818780698883, -0.20996455562958014, -0.34561710527233275, -0.46367256578980576, -0.4600953132789735]], [[-0.24885369055590792, -0.4543492501382794, 0.3585674926509895, 0.061772332708159454, 0.08540534976256617], [-0.11603839620342668, 0.26386093412127637, -0.2688515198626618, -0.25739759528300765, 0.015679396397217295], [0.0014605421582918278, 0.48425237217259803, 0.24928128362774263, -0.037728256464662886, 0.09448901066078808], [0.13914221648059655, 0.029999244227269975, 0.3439611807595465, -0.2938649439748811, -0.4079780422649851], [0.4227756797152263, -0.3985199415828602, 0.45173640280257366, 0.1474967938419191, 0.07256805046008144]], [[0.577931660213354, 0.39226696033399094, 0.39137468066529235, 0.3969497580489403, 0.3972273600774328], [0.3868099693157071, -0.2989932420283642, -0.2960277425589057, -0.3176687217873588, 0.478596871972232], [0.39556021398955526, 0.4889733127078256, -0.3233910780440755, -0.3420995488521791, -0.34328870130685746], [0.3971109811676212, -0.33253800081307483, 0.49507786441839335, -0.34794073615820476, -0.34907237031281296], [0.3948526375010755, -0.3233023714588628, -0.32067011128804535, 0.509549814009128, -0.34091160378464186]], [[0.3637544217978516, 0.29039080060124817, 0.5444116963099626, 0.033892853601016494, -0.37768889174267356], [-0.1574958888856418, 0.40239169532111013, -0.060821808145812596, -0.2990086268710338, 0.16536939676073056], [0.08465983631441629, 0.139467085578292, -0.41967708224433803, -0.4880677863128336, -0.22193801509552488], [-0.044541064540618076, 0.4249333794985319, -0.11420110591839863, 0.4209000244726199, 0.19405868757133166], [-0.04464171309071231, -0.2202531436426072, -0.16561519105578032, 0.33332040476338265, -0.2638512500847365]], [[0.5334643431659877, 0.301839476107829, -0.4443898366689931, -0.16880183593590353, -0.016732897602805663], [0.37982852127252037, -0.3046250215187319, -0.10126995332192429, 0.023762489669664457, 0.23945433882485478], [-0.2124927556730101, 0.40026926316840705, -0.14458881407889423, -0.03406963997512817, 0.33746017286157337], [0.06357546055180947, 0.447537187390943, 0.012358913436792409, -0.10018991268076763, -0.36200512339877994], [-0.33678094929823305, -0.3199025195831043, -0.3464324587112196, -0.43558653024645416, 0.226272772640899]], [[0.5531921670820058, 0.4486917542272879, 0.30261504047548876, 0.28294322893076573, 0.44599851334391677], [0.3381676230096912, -0.4114510227876773, 0.18895613343935475, -0.41001745141667234, -0.18292836802528017], [0.4438817365974033, 0.12871929409309305, -0.41196928738358063, 0.14111255270690345, -0.3359872826904373], [0.4489869516948332, -0.3362686647607809, -0.17228879748925666, -0.4268280206782521, 0.2914927140608316], [0.29954787894437934, 0.04970098775334891, -0.40906417812432255, 0.19673135496004868, -0.38984941222942265]], [[-0.37753992752751697, -0.31692304121666887, -0.4159366006162892, -0.07333156129210283, 0.38716463953676117], [0.3656687405161088, 0.08347722087670553, -0.15463471025528583, -0.4592905014296138, -0.33773090878340356], [-0.5224826659020521, 0.3397905225486418, -0.5162593203633417, 0.03071605201909559, -0.3814940681697195], [-0.3864780345298691, 0.023299359524954138, 0.23671677096287835, -0.40629132022927855, -0.3364419719369744], [0.3765435280950565, 0.21820965919582466, -0.32666039214312875, 0.43605110028508526, -0.12702689875492432]], [[0.276033353828001, 0.4767574622659577, 0.47624343727339413, 0.4757351363724534, 0.476988970905481], [0.4771216577420039, 0.3632883958995704, -0.49469698166909587, -0.25358313709187125, 0.13685384642679954], [0.4758716884648593, 0.12087160235409182, 0.36769989713617807, -0.49502380777702887, -0.2527415187627561], [0.4768041547350305, -0.2602493121691592, 0.12469738062100562, 0.36667204093167016, -0.49445121597767366], [0.47645182722123686, -0.494824020251286, -0.24932866996533462, 0.11645092906300021, 0.3646024738445654]], [[0.039016294388294376, 0.2504193048395978, -0.33946873553318113, 0.2653304571304704, -0.10577879489758744], [0.3575609479801762, -0.3882308305060261, 0.03708429952259657, -0.056154142017684226, -0.2142658036122992], [-0.13980822501738663, 0.0884673708547068, -0.3030026119791392, -0.1832413773270204, 0.4523905197621136], [-0.14186954828174, 0.46232127931634465, -0.3639083898478263, -0.3610996534411818, -0.3133861959445297], [0.36053679476545364, 0.38258013663300067, -0.3802142825489377, 0.09931072492834174, 0.1351003809323885]], [[0.2391292036446338, -0.3188230897087608, 0.44924075564115706, 0.21803997217303042, -0.22621525422227493], [0.10001425279878434, -0.4596570599598183, 0.37677132398318797, -0.34748985488245154, 0.39357764306599374], [0.44805173301060647, -0.34188699660789124, -0.19790234985941232, -0.07404507740580166, -0.1924676776065356], [-0.30631910118754013, -0.3096258681016236, 0.2656789856315487, -0.310928667709841, -0.5140602817802946], [0.4709718352615144, 0.3231630941418057, 0.4583879778047565, -0.01950154983370799, 0.14713155615899748]], [[0.40936380334701117, 0.4207138428785006, 0.49388578461137767, 0.4200839522487144, 0.404723440784635], [0.3507937148721419, 0.18523327022272523, -0.4077375027911672, -0.36964829434625346, 0.35712598291638287], [0.4882735013017912, -0.13709687321022132, 0.272827701923847, -0.1441117121366463, -0.35392260889281313], [0.5130930802054605, -0.1577511733774244, 0.03534005099354275, -0.1508479325948973, -0.46596881466619244], [0.33025866895207034, 0.06869738908863454, -0.32234472546571696, -0.29333007812368644, 0.3216439959646459]]]
CATALOGUE_IMAG=[[[-0.16608697685346524, 0.008165491714845197, 0.021677961008128122, 0.004310073202373692, 0.011636948150126353], [0.0023595823367813813, 0.3802849215750052, 0.030904662756617886, -0.1140689956166121, -0.4686075255341888], [-0.0079679114601049, -0.45502871505084863, 0.38282606739200953, 0.02915430432054546, -0.11333647765408363], [-0.0027742173490899586, -0.11375744843576119, -0.4733510608899572, 0.37303896520732566, 0.029957605707011038], [-0.00021393211066349943, 0.029899774482266474, -0.11761881187194963, -0.4589026088858473, 0.38100214039939156]], [[-0.20258329308816256, 0.07561793018686512, 0.1614192797300425, 0.24720759110701587, 0.5295053876974681], [0.5261014605803858, -0.2034951537845159, 0.07506405860094859, 0.16069628706926292, 0.2471732675043404], [0.24719215054654756, 0.5282342136012318, -0.2075142750888515, 0.07263551371264124, 0.15885709801981085], [0.1605436792933092, 0.24720188348439676, 0.527271697942957, -0.20823498037068947, 0.07117574412555717], [0.0784107358360077, 0.16291254862664256, 0.2472023632057533, 0.5213826656480414, -0.20177327739612388]], [[-0.49108321507670266, 0.20183685816638453, 0.1927124703800581, 0.5025624943422269, -0.26916573770840974], [0.26572060009273, -0.2973837004164371, -0.0985646020689721, 0.2548382528253881, 0.07263319352848666], [-0.04475605244784913, 0.1920596368604755, 0.1957313942790332, -0.4026481261499536, 0.32321716263876943], [-0.03936003718643441, -0.17917360449879996, -0.4712008247709907, -0.03351149833567081, -0.4020566494644409], [0.07724221305930822, 0.414351921689472, -0.38907693826042594, 0.07170435468640862, 0.17432733217287713]], [[-0.06524902073055938, 0.5081292332147104, -0.36218983641815994, -0.10258411339135429, -0.1436895149114442], [0.1461433823272813, -0.4676162032167779, -0.03874604783589108, -0.5209166667346451, -0.5928189369067469], [-0.272280032944338, -0.14739234907249196, 0.6372416046697943, 0.2268672818473825, -0.05008231145547635], [-0.031041752181573547, -0.42587823824866744, -0.5513271658082952, 0.4539231909920192, -0.0634357401557447], [-0.6014057133271434, 0.23569681355451302, -0.18952496472077576, -0.10029086103878712, -0.3357423281224061]], [[0.19592628247619495, 0.5831954468609518, -0.5998682903656243, -0.12866747261478498, -0.3202681515655377], [-0.24390438334209424, 0.636071953723619, 0.16404154466247134, 0.1842363317119148, 0.2501683341596365], [-0.6396446409069542, -0.10171125622266165, -0.11188384561408937, -0.4830039858660893, -0.11811075943234307], [0.14715909524759657, 0.2373050574746357, 0.5259184359706849, -0.31410192557667777, 0.14621170080920923], [-0.23845448237944633, -0.11258866393000734, -0.5271577468812211, 0.17659848744388273, 0.537990999481737]], [[-0.2203627295314737, 0.08495510827221657, -0.1445060920692593, -0.2649599562061979, 0.45156335525537056], [0.2191644058892341, 0.15332675562651527, -0.5896463132246152, 0.23722194304088826, 0.06224013286975165], [0.15646584204284714, -0.15456763602141046, 0.5793728021948547, 0.04193846543791258, -0.05085114694127534], [0.005699577938373591, -0.03858807132628424, 0.2688779931871621, -0.22398936791515842, -0.3274330513636603], [-0.08036174644881065, 0.3430570863518436, 0.2513301286525836, 0.3595776772503271, -0.27308635530515546]], [[-0.5396381896216904, -0.09521977512575885, 0.3027518543609291, 0.3235124215829693, -0.18948872715819617], [-0.2897634280032032, -0.4494847856362607, -0.22896924665663712, -0.029309701203320732, 0.5675893599768513], [0.06078794342476092, 0.4934016121567035, 0.517887245213817, 0.4520934235842047, 0.37524799538533127], [0.25324755537457466, 0.07607474143191692, -0.4537614955050079, 0.546559448035058, 0.10785273373719648], [0.6020133578689822, -0.5171951949205899, 0.34375831026175374, 0.21963652952108587, -0.18167090445585013]], [[0.33729767043442227, 0.10181806037830059, 0.20427158014822255, -0.3605562671837466, -0.09253037745454193], [-0.0035343189421288826, -0.008141658260611497, -0.04604236000572231, -0.2400571960978259, 0.3538282387995308], [0.18504596601678297, 0.157702241022525, 0.4346820959842433, -0.1408673450909321, -0.6138296814626303], [0.34170358580697124, 0.38263254541121916, -0.13507436680190754, 0.10568900719995718, -0.08884927895184878], [-0.5430571297214432, 0.45757289605910045, -0.1838449768983473, -0.3230218658456715, -0.078620951585675]], [[-0.0017103715890497311, -0.005182987975171739, 0.008519291918399296, 0.003514082989950989, 0.0011596704821236095], [0.012804966199855472, 0.022867436572021317, -0.007570589393876698, -0.011096167601704005, 0.009936697022027449], [0.0020985899592728353, -0.0059811607553814025, 0.0016154201724418468, -0.0004861362530591141, -0.0024767128880787783], [-0.0018877090871806947, 0.0006716093864517125, -0.010840551037259838, 0.0029949116516267953, -0.002454635641394415], [-0.008638680409371085, -0.0013185569253750455, 0.006346218796424467, 0.0015001211588310036, 0.0020419272103563866]], [[-0.617542726314024, -0.1974884594555609, 0.21548155367755456, -0.4016855864571362, -0.0004940078014474729], [-0.24206916631474312, 0.20723018593488865, 0.15832887805045956, 0.2507155529451282, 0.572637314270459], [-0.1415569509834651, -0.5504896573109681, -0.17486283406965805, -0.17280292270136052, -0.15540243825639954], [0.34513792979851426, 0.08307122688785844, 0.33538704873720293, 0.07151339836950536, -0.3871409839106667], [0.08134432854646162, 0.1564601361087934, 0.5804621839004227, -0.4172653929557543, 0.10200845664720774]], [[0.3120454217713932, -0.07376755927633494, -0.03656759929715498, 0.03574821816632365, 0.20255394069012372], [-0.5867333972381291, 0.4685197952446392, -0.35181705061100116, -0.25195913922879315, -0.14601064410283415], [0.5834107450944103, 0.25136632565900346, -0.23884703982591957, -0.19874176019769202, -0.22972163187076136], [0.16157010067360716, 0.4802639382859751, -0.2457090518617605, -0.2743037864511113, 0.28797592415188783], [-0.0466477110810617, 0.1673726936211684, 0.6100132719994328, -0.6098811585545731, 0.19938604758403058]], [[0.030134018019292396, -0.28462510983090283, -0.15168444573837617, 0.23923411974583325, 0.3480606850080063], [0.3592562244537024, 0.025295596999878277, -0.2879173234603273, -0.15294239175586247, 0.24245089366946412], [0.23926605402355353, 0.36294550549107635, 0.030294467941056384, -0.2970733929664432, -0.1477350762111652], [-0.15079939308186438, 0.25112974445080394, 0.3540078924571715, 0.030794718213092154, -0.29771649228952907], [-0.2755915668968785, -0.16752888509130465, 0.2625513036491757, 0.3744711928987188, 0.025138951050906102]], [[0.5491403629882352, -0.11482823787376825, 0.1276221318463415, -0.08686441258616913, 0.4302587400187667], [0.5991410365598726, 0.3709784241299281, -0.01970060870604267, 0.3354913447439423, 0.007540314881032623], [0.06499023757851566, 0.610699374109002, -0.4966066931074951, -0.29235668560484557, -0.08879181053136957], [-0.06503626012860467, 0.27821474037597543, 0.11013683100132654, 0.1037878615063272, 0.3515025216868747], [-0.12512417847708052, 0.13793672347322927, -0.17198649269431765, 0.5276479145772587, -0.19965234711271934]], [[0.21124218689115426, 0.2001581759173635, -0.02078365943873438, -0.22024055529953612, -0.47157684149962875], [0.093428103159601, 0.20401866037281235, -0.38809187271802864, -0.04299932255476677, 0.4853202914960983], [0.19296658703198122, -0.05848876922638377, 0.3756211418734731, 0.13817792538345708, 0.263628159423044], [0.5160000945400434, 0.16249970571757452, 0.5804672650435633, -0.2937633427413273, -0.011140241970165776], [0.10501612011213944, -0.5104664259112033, -0.2754874642474893, -0.25306634292355723, -0.10337433936947361]], [[0.57330350838077, -0.13352618161464777, -0.47800780819748456, -0.07968375388607549, -0.37357717002055946], [-0.04761472909817497, -0.09419711978304868, 0.3710212825436705, -0.16583926318851416, -0.5837204829008368], [0.19378151373471073, 0.21597704029513776, 0.001753644103636698, 0.2706512669597942, 0.1087148680235673], [0.012522706190863259, -0.3142797875664198, 0.35096571063783605, 0.5841205581387059, 0.2534528576917887], [0.36038423595027963, -0.358686558499008, 0.09180426288049295, -0.5040513921384193, 0.5381484744423873]], [[0.11645120673603386, -0.42423395727807983, 0.11631669941516731, -0.1026175881629808, -0.10145132117575137], [-0.10382808011994439, 0.11689334935613314, -0.42462655459436, 0.11747535991011199, -0.10378183891793936], [-0.10079194202434064, -0.09994879868941108, 0.1161067391080011, -0.42441644607879186, 0.11622758388021333], [0.11667934991612551, -0.10142242305180765, -0.1018129714387447, 0.11701188963421444, -0.4244606849649501], [-0.42405817898495113, 0.1155921917302428, -0.09900254888805435, -0.10057580246290147, 0.11582967839847257]], [[0.03185765390889853, -0.33889826012994134, -0.2710406216755325, -0.15234648346461663, 0.06012436567029916], [0.16918899996752482, 0.06115171636020142, -0.36148387735312343, -0.41378787045938226, -0.005008137140799315], [-0.004339496906712755, 0.10902147108202007, 0.010045974794560073, -0.2969831204552815, -0.3747669399302298], [-0.4004610697532928, -0.14776123144420975, 0.16586920164524385, -0.038042706914710504, -0.30152447024629636], [-0.35200235858982687, -0.26068567551528393, -0.18057916586809863, 0.14093348481206705, -0.026972875728658877]], [[-0.5486207835932991, 0.2795113705724513, -0.36225031411760017, 0.22437343611017987, 0.20652110102655555], [0.5176994935240347, 0.42254624584102773, 0.21776379547247457, 0.04999180691993362, -0.13634702691203812], [0.1344425258940077, -0.2966395499685363, 0.25174590814544795, -0.20587333664818255, -0.053664267590216905], [-0.5199459954681893, -0.3762563663008431, 0.5594415089730721, -0.08787959619053713, -0.18934989519038506], [0.0464595395133571, 0.040921465355396036, 0.4567706250617641, 0.556106444624158, 0.5676477970380607]], [[0.5257544412126762, -0.2484987040463961, 0.27292507748367445, 0.22921014376472146, 0.4790247145785298], [0.4593657475148706, 0.4610066317213508, -0.5560808267685167, -0.013615958347896496, -0.13165658774903616], [0.46233113038581, 0.45684592908615423, 0.28361959660451835, 0.0705989952814045, -0.1253008105116352], [-0.009480254370201757, 0.004237627092722282, 0.2643520439408372, 0.5297537922769263, -0.34751682607760487], [-0.15595469532455408, -0.09716035375097776, -0.2391582513957194, 0.032875747866531274, -0.4033314973422083]], [[-0.2691641272786755, -0.5158186525820024, -0.1435720380663498, 0.29422323881572243, -0.6412140221297249], [0.02620035125673385, 0.06146763839780223, 0.1419308280941168, -0.11834490851414574, -0.46204224495349144], [0.19061862776589605, 0.41183009443803953, -0.5004938168426057, -0.2666714778086345, -0.2809595540628532], [0.31938708554639905, -0.3100020120304148, 0.12207957813130449, -0.12498876512319419, 0.06766092316464], [-0.0776387844161336, -0.0572190617451068, 0.09017330418272589, 0.44416431697562847, 0.41904065595876816]], [[0.2395557302442558, 0.3547141625417398, -0.17590471187188475, 0.13711937553520775, 0.5430030278882351], [0.5438176741631457, -0.40878439885935836, 0.4511338815666722, 0.11567434328960341, -0.02056108248752779], [0.3793417874978404, 0.028790952285001833, -0.484228147819459, 0.4972253178915278, 0.15313835209183424], [-0.17711600663677826, -0.3100022502681417, 0.08199573667400717, -0.32471068950355014, 0.4463525538649274], [0.19122941500378443, -0.3587199278175182, -0.44792804074955517, -0.006552818081370483, -0.3436577553788578]], [[-0.09163089418158692, -0.1314627814600073, 0.5377193447465625, -0.07764984583698968, 0.25987202741628446], [-0.22683964878523175, 0.5021807087348238, -0.26286236581636757, 0.2169718122879763, -0.02973772158576999], [-0.11486804398696468, 0.10536036323324655, 0.017100821577017585, 0.12371431669974879, 0.4861514336118024], [0.2147341900645548, 0.06315851070402373, 0.1352501480729977, -0.1708214149402127, -0.13994793014210968], [0.3770686561346472, 0.0684722114020807, 0.3071930105382728, 0.45118851639389523, -0.024107012575982245]], [[0.05856429161755963, -0.4812298105323099, -0.18006513682700875, -0.4994051724584315, 0.2413805486401486], [-0.4514583219254503, -0.5599285967240378, -0.06599858181159607, -0.05760177442550687, -0.20852753512191358], [0.4223000192215216, -0.39955173762591717, 0.2666184548629002, 0.143728856556086, 0.4021706669526616], [-0.04411441098580246, 0.14875594653590385, 0.3541729917805528, -0.25507561059268546, 0.2016892022868676], [-0.19028110309562082, 0.07103353491589565, -0.5276930026663506, -0.07873187299913226, 0.10611566268125779]], [[-0.17430803040586987, 0.055955288709814276, 0.44270855102597645, 0.012494430772015444, 0.37521897979410385], [0.1255599257643828, -0.563618011524426, -0.039517775897645734, -0.5092505716332701, 0.4986183792011642], [-0.10050834908699718, 0.1564803836541429, 0.4655422077007861, -0.068317836633971, -0.19971134284534836], [-0.07463149251507542, 0.398251064462962, -0.3621361411177493, 0.07970852134771997, 0.1921709884454298], [-0.3367137413966597, -0.07192597517237981, 0.2664692083078341, -0.4011670654811417, -0.5316146083383023]], [[0.008554974199219753, 0.241922684880623, -0.036085406923111185, 0.25104942781398487, 0.18752695345672288], [-0.30109007779129165, -0.39471093475803837, -0.25491045258688616, -0.08521210422161038, 0.03199168526107938], [-0.17592822453214982, -0.14215542327939845, 0.14884685962798488, 0.08271145382526712, 0.32633850660021096], [0.3850072609291859, 0.3469140706670739, -0.08436286990035964, 0.34197148708692204, -0.43853370121836605], [0.36771400636886825, 0.18162791542222287, 0.18990126925734663, 0.04039627833645782, -0.021642176979634463]], [[-0.5044906441597107, 0.0873705297055765, -0.4148278971784687, 0.2338910672872562, -0.18912313540170808], [-0.00035981102147810307, 0.3138017252125085, 0.2871731231636026, 0.3950503379093681, -0.1715618302275891], [-0.2859496709616948, 0.3535448910898711, 0.3648405027007372, -0.33194682759409455, 0.12395608433589357], [-0.10088321875045926, 0.2781110464684871, 0.061981034691485824, -0.16424040302559206, -0.36959180319128176], [0.5872827607685815, 0.46096019230529434, 0.09855588759176243, 0.1612501094691385, -0.25758636661341805]], [[0.01504386309254806, 0.008933233274840665, -0.022863705464482725, -0.002376835762595246, 0.023949628837535326], [-0.017003639539808362, -0.006813863451526442, -0.019154176702825664, 0.0012281106801163783, 0.023466043174370805], [-0.04551277048323314, -0.02587348701972677, 0.011735936675358263, -0.015507118280304958, 0.009055944480014356], [0.011730746895385648, -0.00879485734333393, -0.015477930729837549, 0.015284621145692073, -0.016927901129985563], [0.0032845366955390597, -0.004162339059717263, -0.015580596229114172, 6.403787118431262e-05, -0.018058861668699553]], [[0.030234030076702927, 0.15801651607698, 0.3504508936224641, -0.5656842304212083, 0.44154184584066963], [0.03521041665275253, 0.36526359316050644, 0.34294694907494677, 0.14746502686504806, -0.5091299186359209], [0.431172403634927, 0.12233565690013692, 0.480198845990286, -0.22622332398978312, -0.0039111724734593135], [-0.4520538720388979, 0.24277691467794202, 0.4826513528726339, 0.19352085816171533, 0.0718325763514087], [-0.6230984964720802, 0.2052655636475403, -0.2141388119981862, -0.440780665074027, -0.026240552082271187]], [[0.43933849804735653, 0.12493232480399603, 0.42330772913432124, -0.1359996031088615, 0.23681483844468493], [0.08698512969899583, -0.2503577889542114, -0.19254310397058802, 0.25773230108720535, -0.33338898056465066], [0.20928674667165534, 0.146158902972608, -0.41738570679498266, -0.49420004134366236, -0.22743473994478525], [-0.1701160197947459, -0.3944365964458006, 0.42240333395353624, -0.4692961893230011, 0.10122081523201926], [0.17905429535269035, 0.21205198349173232, -0.027698680033209522, -0.3767684117618469, -0.6045345762780268]], [[0.1916959381257505, 0.6125341915993755, 0.12911049654948117, 0.09744795730985582, -0.45663255605772357], [-0.25654714774372256, -0.17382755551614648, 0.48676398061681414, -0.2768716642311006, -0.30088597637739384], [-0.03736438833733817, -0.5369142008687618, -0.13004828133163446, 0.12265809986688143, -0.020277133427113965], [-0.35650453098345164, -0.24609632499614395, -0.4366042792095694, 0.242319805002488, -0.43147749289273113], [0.2198570411486789, 0.1633607779219258, -0.35357908147563966, -0.07178072323347542, -0.09196207475165304]], [[-0.3956880169710938, 0.11709602637462727, 0.29987750714549327, 0.10118549616951367, 0.5670207649276033], [0.2430827032480177, -0.03390082342114651, 0.016521812230022578, -0.5707945678952219, 0.6259505709916791], [-0.49520061160379825, 0.09984483140991539, 0.42788388676027295, -0.40659190324082434, -0.2990159219858961], [0.47525279852843066, 0.4935275024350844, 0.35625340004937167, 0.10946273082459991, -0.017155586442640928], [0.19939380812671992, -0.24764203329560439, -0.12013071406155132, -0.5541221141888371, -0.09374542002566552]], [[0.16122963785556138, -0.07827280389976603, -0.08261875897240403, -0.049304052418519165, -0.04701515081240712], [-0.10187270310530469, 0.2657123279439015, 0.26901222209311876, 0.24307731938209567, -0.3618632810031871], [-0.059431617078262125, -0.3477141059254461, 0.2354107275412269, 0.2072869959142051, 0.2053116352159575], [-0.047988213512157485, 0.22230267208300158, -0.3389659395321576, 0.19732522424081017, 0.19531635948940204], [-0.06396401065008558, 0.23553253832342447, 0.23910390989382052, -0.31679486587263717, 0.20923498370249374]], [[-0.04038492925476037, -0.32238491872901187, 0.15523732778492016, 0.4213150550042742, 0.18975300301356024], [0.4358061558496825, 0.129367788243792, -0.4182771912886799, 0.16430001053447293, 0.5331136347165952], [-0.45559266083594085, -0.3990038009076891, 0.050261792022905746, -0.270826551263786, 0.25912192084111224], [-0.36326892993754295, 0.3740512611993838, 0.4185892501065456, -0.038707499240238294, 0.37549477609759097], [0.5483126413742694, -0.2922956972039257, 0.32637375871837476, -0.32191531224955927, 0.380938943035598]], [[-0.17258806685975292, 0.16158941017101464, 0.05299032957686209, 0.4017812081895854, -0.4218038122592836], [-0.23668670514266324, 0.34700258939680073, 0.5452946870441955, -0.3415458254776504, 0.32302323583644676], [0.36475371784964933, -0.03830023805552101, 0.37520190046900853, 0.4207584883644338, 0.45426749541435624], [0.4311384463084218, -0.0008762792672023597, 0.3421483071347455, -0.5516636907110897, -0.2171422034791763], [-0.06161817107190911, -0.4530610081753792, 0.30527324208759854, -0.10273595588209389, -0.3323898669242051]], [[-0.06992325001400196, 0.016549692629059202, -0.28008002689423794, 0.18924916149458365, -0.051802593880014534], [-0.23593080115795584, -0.02699518538623943, -0.5331161626581948, -0.043662039815378466, 0.3695374086052262], [0.06758096294780055, 0.5425331122996856, -0.01737787212889713, -0.42624573910623126, -0.05463205830341114], [-0.0029832608775650633, -0.05287249780530721, 0.3746161428005523, 0.13934139155887626, -0.4753344425379409], [0.1616885405466159, -0.4462375986557125, 0.05183803972504521, 0.5217351901002951, 0.22274563614592383]], [[-0.36233408251226623, -0.22437661002798376, 0.3175202804471778, 0.38132343135471586, -0.029809714424026273], [-0.13065775671974236, 0.4281683971258737, 0.3561926214366836, 0.3269385806660151, -0.27610587816084975], [-0.02888335263388779, 0.1879560532885115, 0.08543241987366554, -0.3870937888032215, 0.07243861631056742], [0.03768031326304395, -0.43560734905229254, -0.3078152815026907, 0.158820608892509, -0.4523752864055846], [0.09486858611468843, -0.5198279954610161, 0.20994776823431596, -0.012492423416002202, -0.41732575968342195]], [[-0.10398383342216477, 0.030831147627832272, 0.0379529438166261, 0.04386704356353743, 0.02701405420986315], [0.0245593404603311, -0.09389214765691081, 0.01888342117867941, -0.42517911502162636, 0.34937833176243316], [0.042360047307924344, 0.35522422458394365, -0.07477285358306064, -0.005754737340362553, -0.4256799407552833], [0.030100423879691394, -0.42113178790421985, 0.35389936034580544, -0.07966058238497872, 0.024488411261838913], [0.035240514012525335, 0.015194612541359646, -0.4276878553352253, 0.35669787651074447, -0.08865229863536365]], [[0.3929968695047369, 0.2423871738812644, 0.2949502371007834, -0.4535303413280171, 0.4827587169867811], [-0.34116755525141534, 0.3540718066025217, 0.4481734256950333, 0.3439594200378165, 0.3317513768934406], [0.3851329457812541, 0.3848926278358336, 0.24602300561162513, 0.4589855443388208, -0.273710151107552], [-0.5093598536008583, 0.17465442155298613, 0.14109778862448943, -0.15992451642251354, -0.2639371476319238], [-0.14950291561241102, -0.23636223011339494, -0.3824813269853438, 0.43860237651489203, 0.36617726476159346]], [[-0.4062355956655012, -0.25517042365831966, 0.29891403805855893, 0.32490207963799317, 0.33998053826432073], [0.3542102195385314, -0.2328549213429972, -0.15749119678750736, 0.21450153471381836, -0.33256979123667446], [0.1464915689301429, -0.2233235079261231, -0.3375462201512763, -0.5344943266204577, 0.36016149255236807], [0.20404984350839392, 0.4118711557839096, 0.31012059627635824, -0.26473282992107544, -0.03532780576983156], [-0.25065873049023524, 0.17616283652425696, 0.1099572509007093, -0.47098813655508903, -0.3373722507423305]], [[0.10102118725404911, 0.0279976101721628, 0.21199249208169113, 0.036241920865871, 0.11824948306184771], [-0.15709982158925104, 0.4786021828018491, 0.10739707902252477, -0.35599375312989506, -0.14211831923850773], [0.15798861258677518, 0.3590835998665952, -0.3214794687501837, 0.3563262532598679, -0.37163180796940126], [-0.010339478970121356, -0.35050128621050275, 0.4201607841775826, -0.3535272062642665, 0.21504529183218], [0.19663620195472045, -0.508578491988216, -0.2718048596500423, 0.42110435648191924, 0.21043223183977053]]]
# Daytime controller and follow-up protocol. The numerical core above is
# copied from the frozen first stationary campaign; it remains available for
# the baseline replay before the additional refinement below.
_baseline_run_batch=run_batch
_base_retention=Retention
_base_report=report
_base_self_check=self_check
_base_campaign=campaign
MODE=DAY_MODE
VERSION='ahc5-day-1'
DEFAULT_SEED=521812 if DAY_MODE=='day-replay' else 521819
ORIGINAL_BATCHES=1433336
ENTRY_BOUND=9-4*np.sqrt(5)
BOUNDARY_CUTOFF=ENTRY_BOUND-1e-6
DAY_POLICIES=('root_catalogue','root_weak','root_soft','root_bordered',
              'root_circulant','ascent_weak','polar_weak','root_haar')
ROOT_POLICIES_ORIGINAL=ROOT_POLICIES
if DAY_MODE=='day-search':ROOT_POLICIES=DAY_POLICIES
CATALOGUE=np.array(CATALOGUE_REAL)+1j*np.array(CATALOGUE_IMAG)
CATALOGUE_CACHE=None
METRIC_NAMES=METRIC_NAMES+('worker_peak_mib',)
DDL+='''
CREATE TABLE IF NOT EXISTS baseline_aggregates(family TEXT,status TEXT,data TEXT,
 PRIMARY KEY(family,status));
CREATE TABLE IF NOT EXISTS followups(id INTEGER PRIMARY KEY,baseline_status TEXT,
 action INTEGER,baseline_gradient REAL,baseline_min REAL,baseline_L REAL,
 status TEXT,gradient REAL,min_modulus REAL,L REAL,top REAL,restart_steps INTEGER,escape_steps INTEGER);
CREATE TABLE IF NOT EXISTS signature_counts(signature INTEGER,positive INTEGER,n INTEGER,
 PRIMARY KEY(signature,positive));
'''


def peak_mib():
    import resource
    amount=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return amount/(1024**2 if sys.platform=='darwin' else 1024)


def catalogue_cache():
    global CATALOGUE_CACHE
    if CATALOGUE_CACHE is None:
        spectra=[];directions=[];witness=[]
        for u in CATALOGUE:
            _,h,c=hessian_matrices(u[None]);values,vectors=np.linalg.eigh(h[0])
            spectra.append(values)
            directions.append(np.einsum('ak,aij->kij',c[0]@vectors,OFF))
            witness.append(directions[-1][-1])
        spectra=np.array(spectra)
        CATALOGUE_CACHE=dict(spectra=spectra,directions=np.array(directions),witness=np.array(witness),
                             moduli=np.sort(abs(CATALOGUE).reshape(len(CATALOGUE),-1),axis=1),
                             weak=np.argsort(spectra[:,-1])[:8])
    return CATALOGUE_CACHE


def signature_match(u,spec):
    data=catalogue_cache()
    mod=np.sort(abs(u).ravel())
    errors=np.maximum(np.max(abs(data['moduli']-mod),axis=1),
                      np.max(abs(data['spectra']-spec['horizontal']),axis=1))
    j=int(np.argmin(errors))
    return (j if errors[j]<2e-5 else None),float(errors[j])


def day_starts(seed,bid):
    rng=np.random.default_rng(np.random.SeedSequence([seed,bid]))
    family=DAY_POLICIES[bid%len(DAY_POLICIES)]
    cache=catalogue_cache()
    indices=(bid//len(DAY_POLICIES)*BATCH+np.arange(BATCH))%len(CATALOGUE)
    if family in ('root_weak','ascent_weak','polar_weak'):
        indices=cache['weak'][rng.integers(0,len(cache['weak']),BATCH)]
    if family=='root_haar':return family,haar(rng,BATCH),None
    if family=='root_bordered':
        u=bordered(rng.uniform(.05,.95,BATCH),np.exp(2j*np.pi*rng.random((BATCH,3))))
        return family,perturb(rng,u,-4,-1),None
    if family=='root_circulant':
        f=fourier();u=(f[None]*np.exp(2j*np.pi*rng.random((BATCH,1,N))))@f.conj().T
        return family,perturb(rng,u,-4,-1),None
    base=CATALOGUE[indices].copy()
    if family=='root_soft':
        dirs=cache['directions'][indices]
        order=np.argsort(abs(cache['spectra'][indices]),axis=1)[:,:3]
        choose=np.take_along_axis(dirs,order[:,:,None,None],axis=1)
        b=np.einsum('bk,bkij->bij',rng.normal(size=(BATCH,3)),choose)
        scale=10**rng.uniform(-4,0,BATCH)
        b*= (scale/np.linalg.norm(b,axis=(1,2)))[:,None,None]
        return family,expstep(base,b),indices
    if family=='ascent_weak':
        b=cache['witness'][indices]*rng.choice([-1,1],size=BATCH)[:,None,None]
        noise=np.einsum('bk,kij->bij',rng.normal(size=(BATCH,DIM)),BASIS)
        noise/=np.linalg.norm(noise,axis=(1,2))[:,None,None]
        b+=.1*noise;b/=np.linalg.norm(b,axis=(1,2))[:,None,None]
        b*= (10**rng.uniform(-4,-.3,BATCH))[:,None,None]
        return family,expstep(base,b),indices
    return family,perturb(rng,base,-4,0),indices


def initial_search_batch(seed,bid):
    family,u,indices=day_starts(seed,bid)
    if family in ('ascent_weak','polar_weak'):
        endpoints,steps=ascent(u,use_polar=family=='polar_weak',limit=1600)
    else:endpoints,steps=refine(u,limit=160)
    result=[]
    for k,v in enumerate(endpoints):
        m=metrics(v);spec=None
        if (m['min_modulus']>ENTRY_TOL and m['flat_error']>FLAT_TOL
                and m['gradient']<=GRAD_TOL and m['unitarity_error']<=UNITY_TOL):
            spec=spectral_data(v)
        m.update(id=bid*BATCH+k,batch_id=bid,offset=k,family=family,solver_steps=int(steps[k]),
                 status=classify(m,spec),seed_representative=None if indices is None else int(indices[k]))
        result.append(dict(info=m,start=u[k],raw=v.copy(),endpoint=v.copy(),
                           full=None if spec is None else spec['full'],
                           horizontal=None if spec is None else spec['horizontal'],
                           witness=None if spec is None else spec['witness']))
    return result


def finish_row(row):
    original=dict(row['info']);v=row['endpoint']
    if v is None:
        original.update(baseline=original.copy(),action=0,restart_steps=0,escape_steps=0)
        row['info']=original;return row
    baseline=metrics(v);before=v.copy();action=extra=escape=0
    status=original['status'];spec=None
    # In the replay this follows the original classification, without changing
    # the original numerical path. An entry bound excludes an endpoint only;
    # it is not an assertion about its basin or any uncomputed limiting point.
    if status=='unresolved' and baseline['min_modulus']>=BOUNDARY_CUTOFF:
        v,its=refine(v[None],limit=320,tolerance=1e-12);v=v[0];extra=int(its[0]);action|=1
        m=metrics(v)
        if m['gradient']>GRAD_TOL and m['min_modulus']>=BOUNDARY_CUTOFF:
            r=components(v)[2];v=expstep(v,-.6*r/np.linalg.norm(r));action|=2
            v,its=refine(v[None],limit=800,tolerance=1e-12);v=v[0];escape=int(its[0])
    m=metrics(v)
    if not np.isfinite(list(m.values())).all():status='numeric_error'
    elif m['unitarity_error']>UNITY_TOL:status='unitarity_failure'
    elif m['flat_error']<=FLAT_TOL:status='near_flat'
    elif m['gradient']<=GRAD_TOL and m['min_modulus']>ENTRY_TOL:
        spec=spectral_data(v);status=classify(m,spec)
    elif m['min_modulus']<BOUNDARY_CUTOFF:
        status='small_entry_excluded';action|=4
    else:
        spec=spectral_data(v)
        status='unresolved_uphill' if spec['top']>CURV_TOL else 'unresolved'
    info=dict(m,id=original['id'],batch_id=original['batch_id'],offset=original['offset'],
              family=original['family'],status=status,baseline=original,raw=baseline,
              action=action,restart_steps=extra,escape_steps=escape)
    if spec is not None:
        info.update({key:spec[key] for key in ('top','positive','zero','min_X','pair_top','phase_gap','phase_error')})
        if status in ('candidate','nonflat_uphill'):
            sig,error=signature_match(v,spec)
            info.update(known_signature=sig,signature_error=error)
    return dict(info=info,start=row['start'],raw=before,endpoint=v,
                full=None if spec is None else spec['full'],
                horizontal=None if spec is None else spec['horizontal'],
                witness=None if spec is None else spec['witness'])


def run_batch(seed,bid):
    if DAY_MODE=='day-replay':rows=_baseline_run_batch(seed,bid)
    else:
        try:rows=initial_search_batch(seed,bid)
        except (np.linalg.LinAlgError,FloatingPointError,ValueError) as exc:
            family,u,_=day_starts(seed,bid)
            rows=[dict(info=dict(id=bid*BATCH+k,batch_id=bid,offset=k,family=family,
                                 status='numeric_error',error=str(exc)),start=u[k],raw=None,
                       endpoint=None,full=None,horizontal=None,witness=None) for k in range(BATCH)]
    finished=[]
    for row in rows:
        try:finished.append(finish_row(row))
        except (np.linalg.LinAlgError,FloatingPointError,ValueError) as exc:
            m=dict(row['info']);m.update(baseline=dict(row['info']),status='numeric_error',error=str(exc),
                                        action=0,restart_steps=0,escape_steps=0)
            row['info']=m;finished.append(row)
    finished[-1]['info']['worker_peak_mib']=peak_mib()
    return finished


class Retention(_base_retention):
    def consider(self,row):
        super().consider(row)
        m=row['info'];status=m['status']
        if status.startswith('unresolved'):
            self.offer('day_unresolved',128,m.get('gradient',float('inf')),row)
        if status in ('candidate','nonflat_uphill') and m.get('known_signature') is None:
            fp=np.r_[np.sort(abs(row['endpoint']).ravel()).round(5),row['horizontal'].round(4)].astype('<f8')
            self.offer('novelty:'+hashlib.sha256(fp.tobytes()).hexdigest()[:24],1,m['gradient'],row)


def commit_batch(db,retention,idx,rows,elapsed):
    if idx!=get_meta(db,'next_batch',0):raise RuntimeError('Out-of-order commit refused')
    groups={};basegroups={};hist=Counter();signatures=Counter()
    with db:
        for row in rows:
            m=row['info'];base=m['baseline'];key=(m['family'],m['status']);bk=(base['family'],base['status'])
            for table,cache,kk,value in [('aggregates',groups,key,m),('baseline_aggregates',basegroups,bk,base)]:
                if kk not in cache:
                    previous=db.execute('SELECT data FROM '+table+' WHERE family=? AND status=?',kk).fetchone()
                    cache[kk]={} if previous is None else json.loads(previous[0])
                update_aggregate(cache[kk],value)
            if m['status'] in ('candidate','nonflat_uphill'):
                hist[(m['family'],int(math.floor(1000*m['L']/LMAX)),m['positive'])]+=1
                signatures[(m['known_signature'] if m['known_signature'] is not None else -1,m['positive'])]+=1
            # Every initially or finally unresolved run can be regenerated from its ID.
            if base['status']=='unresolved' or m['status'].startswith('unresolved'):
                db.execute('INSERT INTO followups VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (m['id'],base['status'],m['action'],base.get('gradient'),base.get('min_modulus'),base.get('L'),
                     m['status'],m.get('gradient'),m.get('min_modulus'),m.get('L'),m.get('top'),
                     m['restart_steps'],m['escape_steps']))
            retention.consider(row)
        for table,cache in [('aggregates',groups),('baseline_aggregates',basegroups)]:
            db.executemany('INSERT OR REPLACE INTO '+table+' VALUES (?,?,?)',
                           [(fam,status,json_text(data)) for (fam,status),data in cache.items()])
        db.executemany('INSERT INTO histogram VALUES (?,?,?,?) ON CONFLICT(family,bin,positive) '
                       'DO UPDATE SET n=n+excluded.n',[(*k,n) for k,n in hist.items()])
        db.executemany('INSERT INTO signature_counts VALUES (?,?,?) ON CONFLICT(signature,positive) '
                       'DO UPDATE SET n=n+excluded.n',[(*k,n) for k,n in signatures.items()])
        set_meta(db,'next_batch',idx+1);set_meta(db,'seconds',elapsed)
    for row in rows:
        if row['info']['status'] in ('candidate','hessian_uncertain'):
            print('REVIEW FLAG:',json_text(row['info']),flush=True)


def source_config(args):
    return dict(mode=DAY_MODE,version=VERSION,seed=args.seed,batch_size=BATCH,
                sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                baseline_source_sha256='189951c6b9ea172a8cec0f29fd353234a247df942bd691104e9b9a7f99ddd659',
                catalogue_size=len(CATALOGUE),flat_tol=FLAT_TOL,grad_tol=GRAD_TOL,
                entry_tol=ENTRY_TOL,curvature_tol=CURV_TOL,boundary_cutoff=BOUNDARY_CUTOFF,
                followup_steps=320,escape_steps=800,escape_length=.6)


def report(db,folder,state,elapsed):
    summary=_base_report(db,folder,state,elapsed)
    counts=Counter()
    for _,status,value in db.execute('SELECT family,status,data FROM baseline_aggregates'):
        counts[status]+=json.loads(value)['n']
    assert sum(counts.values())==summary['starts']
    tracked=db.execute('select count(*) from followups').fetchone()[0]
    basetracked=db.execute("select count(*) from followups where baseline_status='unresolved'").fetchone()[0]
    assert basetracked==counts['unresolved']
    failed=sum(v for k,v in summary['counts'].items() if k.startswith('unresolved'))
    assert db.execute("select count(*) from followups where status like 'unresolved%'").fetchone()[0]==failed
    transitions=[dict(before=a,after=b,count=n) for a,b,n in db.execute(
        'select baseline_status,status,count(*) from followups group by baseline_status,status order by 1,2')]
    worker=max((g.get('worker_peak_mib',{}).get('max',0) for g in summary['aggregates']),default=0)
    sessions=[json.loads(v) for v, in db.execute('select environment from sessions')]
    signature_counts=[dict(signature=s,positive=i,count=n) for s,i,n in db.execute('select * from signature_counts order by 1,2')]
    novelty=db.execute("select count(*) from slots where pool like 'novelty:%'").fetchone()[0]
    summary.update(job=DAY_MODE,baseline_counts=dict(counts),followup_identifiers=tracked,
                   followup_transitions=transitions,signature_counts=signature_counts,novel_signature_representatives=novelty,
                   worker_peak_mib=worker,parent_peak_mib=peak_mib(),
                   observed_process_peak_sum_mib=peak_mib()+max(x['workers'] for x in sessions)*worker,
                   replay_original_total=ORIGINAL_BATCHES*BATCH if DAY_MODE=='day-replay' else None,
                   note='Baseline and final endpoints are counted separately; follow-up iterates are not new starts. '
                        'All unresolved IDs saved for regeneration. Signature matches are not equivalence proofs. '
                        'small_entry_excluded excludes the endpoint, not its basin. Unresolved remains unresolved '
                        'even when its current endpoint has positive curvature. No exhaustive coverage claim.')
    atomic_text(folder/'summary.json',json.dumps(clean(summary),indent=2)+'\n')
    text=(folder/'summary.txt').read_text()
    extra=['','DAYTIME FOLLOW-UP','Baseline counts: '+json_text(dict(counts)),
           f'All unresolved identifiers retained: {tracked}',
           'Observed peak memory (parent + worker peaks): %.1f MiB'%summary['observed_process_peak_sum_mib'],
           f'Unmatched invariant-signature representatives: {novelty}',
           'Invariant signatures do not prove equivalence; small-entry exclusion is about endpoints only.']
    if DAY_MODE=='day-replay':
        extra.append(f'Original starting-sequence coverage: {summary["starts"]:,} / {ORIGINAL_BATCHES*BATCH:,}.')
        extra.append('This is a replay of the same starts, not additional independent evidence; paths depend on libraries.')
    for row in transitions:extra.append(f'{row["before"]} -> {row["after"]}: {row["count"]:,}')
    atomic_text(folder/'summary.txt',text+'\n'.join(extra)+'\n')
    return summary


def self_check():
    result=_base_self_check();data=catalogue_cache()
    assert len(CATALOGUE)==40
    for u in CATALOGUE:
        assert metrics(u)['gradient']<1e-10 and metrics(u)['unitarity_error']<1e-10
        assert spectral_data(u)['top']>.20
    for bid in range(len(DAY_POLICIES)):
        _,u,_=day_starts(521819,bid)
        assert max(metrics(v)['unitarity_error'] for v in u)<1e-10
        assert np.array_equal(u,day_starts(521819,bid)[1])
    result.update(catalogue_size=40,targeted_families_checked=8)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hours',type=float,default=7.,help='cumulative active hours, default 7; choose 6 or 8')
    parser.add_argument('--workers',type=int,default=2,help='default 2 workers, one BLAS thread each')
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/('ahc5-'+DAY_MODE+'-out'))
    parser.add_argument('--batches',type=int,help='cumulative batch limit for a short test')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--inspect',type=int,metavar='RUN_ID')
    parser.add_argument('--replay',type=int,metavar='RUN_ID')
    args=parser.parse_args()
    if not math.isfinite(args.hours) or args.hours<=0 or not 1<=args.workers<=4 or args.seed<0:
        parser.error('positive finite hours, 1--4 workers and a nonnegative seed required')
    if args.batches is not None and args.batches<1:parser.error('batch limit must be positive')
    if args.inspect is not None and args.replay is not None:parser.error('choose inspect or replay')
    if any(v is not None and v<0 for v in (args.inspect,args.replay)):parser.error('IDs must be nonnegative')
    if DAY_MODE=='day-replay':
        if args.seed!=521812:parser.error('the replay uses the original seed 521812')
        args.batches=min(args.batches or ORIGINAL_BATCHES,ORIGINAL_BATCHES)
    calibration_report=self_check()
    print('Calibration, derivatives, 40 seed matrices and all targeted starting families: OK.',flush=True)
    if args.self_test:print(json.dumps(clean(calibration_report),indent=2));return
    folder=args.out.resolve()
    if args.inspect is not None or args.replay is not None:
        inspect_or_replay(args,folder);return
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('another process is using this output folder')
        with closing(sqlite3.connect(folder/'results.sqlite')) as db:
            db.execute('PRAGMA cache_size=-8192')
            state=_base_campaign(args,db,folder,calibration_report)
        share_archive(folder)
    if state=='error':raise SystemExit(1)


if __name__=='__main__':main()

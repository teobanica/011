#!/usr/bin/env python3
"""AHC U_5 overnight ascent campaign. Python 3 + NumPy only.

Run: python3 ahc5_ascent.py
Defaults: 7 cumulative active hours, 3 processes, one BLAS thread per process.
Outputs beside this script in ahc5-ascent-out/. Run again to resume.
--self-test checks mathematics only; --hours .02 is a short timed test;
--batches 12 limits the TOTAL number of batches (16 starts each).
--inspect RUN_ID prints a retained endpoint; --replay RUN_ID reruns its batch.

Every start contributes to aggregates. Only bounded samples and selected
endpoints are retained. Every candidate flag is saved, then too many flags
stop the campaign for inspection. These are numerical flags, not proofs.
No old N=4 source or output is modified. Ctrl-C finishes active batches safely.
"""
MODE = 'ascent'
DEFAULT_SEED = 521811
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
    flags = len(retention.pools.get('flags', [])); errors = len(retention.pools.get('errors', []))
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
                flags = len(retention.pools.get('flags', [])); errors = len(retention.pools.get('errors', []))
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


if __name__ == '__main__':
    main()

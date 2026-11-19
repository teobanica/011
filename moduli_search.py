#!/usr/bin/env python3
"""Reproduce the bounded 300-start modulus-only exploration in the order-four section of the manuscript.

NumPy and SciPy are required. This is NOT a search over unitary matrices.
Every endpoint is retained in data/moduli_search.json; optimizer iterates are
not. The centered reciprocal matrix has Frobenius norm less than sqrt(3).
The frozen overnight search programs are separate and are not invoked here.
"""
from itertools import combinations
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
import scipy
from scipy.optimize import least_squares
from curvature import anchor_cubics

ROOT = Path(__file__).resolve().parent
PAIRS = list(combinations(range(4), 2))
BASIS = np.linalg.qr(np.c_[np.ones(4), np.eye(4)[:, :3]])[0][:, 1:]


def balanced(g):
    """Add row and column potentials to positive R until R**-2 is bistochastic."""
    n = len(g)
    x = np.zeros((n, 7)); x[:, :4] = 2
    for _ in range(50):
        rr = g+x[:, :4, None]+np.pad(x[:, 4:], ((0, 0), (0, 1)))[:, None, :]
        m = rr**-2
        grad = np.c_[1-m.sum(2), 1-m.sum(1)[:, :3]]
        if abs(grad).max() < 1e-12:
            return m
        w = 2*rr**-3
        h = np.zeros((n, 7, 7)); ids = np.arange(4)
        h[:, ids, ids] = w.sum(2)
        ids = np.arange(3); h[:, ids+4, ids+4] = w.sum(1)[:, :3]
        h[:, :4, 4:] = w[:, :, :3]; h[:, 4:, :4] = w[:, :, :3].transpose(0, 2, 1)
        step = np.linalg.solve(h, -grad[..., None])[..., 0]
        dr = step[:, :4, None]+np.pad(step[:, 4:], ((0, 0), (0, 1)))[:, None, :]
        rate = np.minimum(1, np.min(np.where(dr < 0, -.9*rr/np.minimum(dr, -1e-300),
                                            np.inf), axis=(1, 2)))
        old = (1/rr).sum((1, 2))+x.sum(1)
        for _ in range(20):
            new = x+rate[:, None]*step
            rrnew = rr+rate[:, None, None]*dr
            obj = (1/rrnew).sum((1, 2))+new.sum(1)
            bad = obj > old+1e-12
            if not bad.any():
                break
            rate[bad] *= .5
        x = new
    return rrnew**-2


def make(x):
    a, b = x[:6].reshape(3, 2), x[6:].reshape(3, 2)
    g = (BASIS@a)@(BASIS@b).T
    g = g/np.sqrt(1+np.sum(g*g)/3)
    return balanced(g[None])[0]


def pair_values(m):
    r = m**-.5; out = []
    for mt, rt in [(m, r), (m.T, r.T)]:
        for a, b in PAIRS:
            d = rt[:, a]-rt[:, b]
            delta = d[:, None]-d[None, :]; np.fill_diagonal(delta, 1)
            out.append(np.dot(mt[:, a]*mt[:, b], delta.prod(1)))
    return np.array(out)


def residual(x, kind):
    m = make(x); r = m**-.5; cubes = anchor_cubics(m[None])[0]
    p = pair_values(m)
    if kind in ('cubic', 'pair'):
        # Preserve the original sequential centering for these two protocols.
        g = r.copy(); g -= g.mean(0); g -= g.mean(1)[:, None]
        scale = np.linalg.norm(g)**3
        values = (cubes.ravel()-cubes.mean()) if kind == 'cubic' else p
        return np.r_[values/max(scale, 1e-15), 10*max(0, .12-np.linalg.norm(m-.25))]
    g = r-r.mean(0)-r.mean(1)[:, None]+r.mean()
    scale = np.linalg.norm(g)
    mixed = np.array([r[i, a]-r[i, b]-r[j, a]+r[j, b]
                      for i, j in PAIRS for a, b in PAIRS])
    out = (cubes.ravel()-cubes.mean())/max(scale**3, 1e-15)
    if kind == 'negative':
        out = np.r_[out, max(0, cubes.mean()/max(scale**4, 1e-15)+.001)]
    else:
        out = np.r_[out, 5*p/max(scale**3, 1e-15)]
    return np.r_[out, 10*max(0, .12-np.linalg.norm(m-.25)),
                 np.maximum(0, .01-abs(mixed)/max(scale, 1e-15))]


def diagnostics(m):
    r = m**-.5; cubes = anchor_cubics(m[None])[0]
    mixed = np.array([r[i, a]-r[i, b]-r[j, a]+r[j, b]
                      for i, j in PAIRS for a, b in PAIRS])
    return {'cubic': float(np.ptp(cubes)), 'pair': float(np.linalg.norm(pair_values(m))),
            'C': float(cubes.mean()), 'minmix': float(abs(mixed).min()),
            'spread': float(np.linalg.norm(m-.25)), 'm': m.tolist()}


def summary(rows):
    return {kind: {'starts': sum(r['kind'] == kind for r in rows),
                   'min_combined_residual': min(r['err'] for r in rows if r['kind'] == kind),
                   'residuals_below_1e_minus_7': sum(r['kind'] == kind and r['err'] < 1e-7
                                                    for r in rows)}
            for kind in sorted({r['kind'] for r in rows})}


def verify_saved(path):
    report = json.loads(path.read_text())
    for row in report['endpoints']:
        m = np.array(row['m']); assert (m > 0).all()
        assert max(abs(m.sum(0)-1).max(), abs(m.sum(1)-1).max()) < 1e-9
        r = m**-.5; g = r-r.mean(0)-r.mean(1)[:, None]+r.mean()
        assert np.linalg.svd(g, compute_uv=False)[2] < 1e-9
        assert np.linalg.norm(g) < np.sqrt(3)+1e-9
        current = diagnostics(m)
        for key in ('cubic', 'pair', 'C', 'minmix', 'spread'):
            assert abs(current[key]-row[key]) < 1e-8
    assert len(report['endpoints']) == 300
    print(json.dumps(summary(report['endpoints']), indent=2))
    print('All 300 saved modulus endpoints and recorded diagnostics checked.')


def run(path):
    report = {'meaning': 'Bounded modulus-only exploration; no unitary realization or AHC certification.',
              'parametrization': 'R=G+row+column, ||G||_F<sqrt(3), M=R**-2 bistochastic',
              'seeds': {'cubic_then_pair': 185, 'negative_then_joint': 9182},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__}, 'endpoints': []}
    start = time.monotonic()
    for seed, kinds, count, limit, tol in [(185, ('cubic', 'pair'), 30, 180, 1e-11),
                                          (9182, ('negative', 'joint'), 120, 250, 1e-10)]:
        rng = np.random.default_rng(seed)
        for kind in kinds:
            for k in range(count):
                fit = least_squares(lambda x: residual(x, kind), rng.normal(size=12),
                                    max_nfev=limit, ftol=tol, xtol=tol, gtol=tol)
                row = dict(kind=kind, k=k, err=float(np.linalg.norm(residual(fit.x, kind))),
                           **diagnostics(make(fit.x)))
                report['endpoints'].append(row)
                report['summary'] = summary(report['endpoints'])
                path.write_text(json.dumps(report, indent=2)+'\n')
                if k % 10 == 0:
                    print(kind, k, 'residual', row['err'], 'seconds', round(time.monotonic()-start, 1),
                          flush=True)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-saved', action='store_true', help='check endpoints without rerunning searches')
    args = parser.parse_args()
    path = ROOT/'data'/'moduli_search.json'
    verify_saved(path) if args.verify_saved else run(path)

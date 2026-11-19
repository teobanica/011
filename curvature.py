#!/usr/bin/env python3
"""Real cubic curvature formula and the rank-two-only obstruction in the order-four section of the manuscript.

Run python3 curvature.py from code011/. Requires NumPy and mpmath.
Checks the formula on the saved initial nonflat stationary returns, then gives
an interval certificate for an auxiliary bistochastic matrix. That matrix
cannot be the squared moduli of a critical unitary: its anchor values differ.
It is NOT an AHC counterexample. No new unitary search is run.
"""
from itertools import combinations
from pathlib import Path
import csv
import json

import numpy as np
from mpmath import mp, iv

ROOT = Path(__file__).resolve().parent
A = [[-15233, -48634], [7967, -41084], [13665, 113540], [-6399, -23822]]
D = [[79990, -3775], [-104383, 15717], [37032, 149], [-12639, -12091]]
CENTER = ['2.007055173083', '2.007781759308', '2.008127664289',
          '2.002764453315', '-0.001933545200', '-0.000421319991',
          '-0.004970340424']


def adj(a):
    return a.conj().swapaxes(-1, -2)


def anchor_cubics(m):
    """Return all sixteen cubic expressions, whether or not m is realizable."""
    r = 1/np.sqrt(m)
    # Axes: batch, fixed row, fixed column, varying row, varying column.
    t = (r[:, None, None, :, :] - r[:, :, None, None, :]
         - r.transpose(0, 2, 1)[:, None, :, :, None]
         + r[:, :, :, None, None])
    first = np.einsum('nij,nabij->nab', m, t)
    second = np.einsum('nij,nabij->nab', m, t*t)
    third = np.einsum('nij,nabij->nab', m, t*t*t)
    return (first**3 - 3*first*second + 2*third)/2


def saved_returns_check():
    result = {'returns': 0, 'max_stationarity_residual': 0.0,
              'max_cubic_discrepancy': 0.0, 'min_curvature': float('inf'),
              'max_curvature': -float('inf')}
    for family in ('haar', 'phase', 'near_real', 'near_flat'):
        with (ROOT/'data'/f'{family}.csv').open() as stream:
            ids = [int(r['id']) for r in csv.DictReader(stream)
                   if r['status'] == 'nonflat_uphill']
        with np.load(ROOT/'data'/f'{family}.npz') as data:
            u = data['finals'][ids]
        if not len(u):
            continue
        r = abs(u)
        x = adj(u) @ (u/r)
        residual = np.linalg.norm(x-adj(x), axis=(1, 2))/2
        direct = (1/r).sum(axis=(1, 2)) - np.einsum('nij,njk,nki->n', x, x, x).real
        cubes = anchor_cubics(r*r)
        error = float(abs(cubes-direct[:, None, None]).max())
        assert error < 2e-8
        assert residual.max() < 1e-10
        result['returns'] += len(u)
        result['max_stationarity_residual'] = max(result['max_stationarity_residual'],
                                                 float(residual.max()))
        result['max_cubic_discrepancy'] = max(result['max_cubic_discrepancy'], error)
        result['min_curvature'] = min(result['min_curvature'], float(direct.min()))
        result['max_curvature'] = max(result['max_curvature'], float(direct.max()))
    assert result['returns'] == 3596
    assert result['min_curvature'] > 3.58
    return result


def certificate():
    """Certify a zero of seven rational normalization equations in a fixed box."""
    mp.dps = 70
    iv.dps = 60
    for matrix in (A, D):
        assert all(sum(row[k] for row in matrix) == 0 for k in range(2))
        assert matrix[0][0]*matrix[1][1] != matrix[0][1]*matrix[1][0]
    numerators = [[sum(A[i][k]*D[j][k] for k in range(2))
                   for j in range(4)] for i in range(4)]
    g = [[iv.mpf(n)/10**10 for n in row] for row in numerators]
    center = iv.matrix([iv.mpf(v) for v in CENTER])
    radius = mp.mpf('1e-10')
    box = [iv.mpf([str(mp.mpf(v)-radius), str(mp.mpf(v)+radius)]) for v in CENTER]

    def reciprocal(xs):
        return [[g[i][j]+xs[i]+(xs[4+j] if j<3 else 0)
                 for j in range(4)] for i in range(4)]

    def system(xs):
        r = reciprocal(xs)
        f = iv.matrix([1-sum(v**-2 for v in row) for row in r]
                      + [1-sum(r[i][j]**-2 for i in range(4)) for j in range(3)])
        h = iv.matrix(7)
        for i in range(4):
            h[i, i] = sum(2*r[i][j]**-3 for j in range(4))
        for j in range(3):
            h[4+j, 4+j] = sum(2*r[i][j]**-3 for i in range(4))
            for i in range(4):
                h[i, 4+j] = h[4+j, i] = 2*r[i][j]**-3
        return f, h

    # Any rational preconditioner passing the interval tests suffices.
    # The following high-precision inverse merely chooses that preconditioner.
    rc = [[mp.mpf(numerators[i][j])/10**10 + mp.mpf(CENTER[i])
           + (mp.mpf(CENTER[4+j]) if j<3 else 0) for j in range(4)] for i in range(4)]
    hc = mp.matrix(7)
    for i in range(4):
        hc[i, i] = sum(2*rc[i][j]**-3 for j in range(4))
    for j in range(3):
        hc[4+j, 4+j] = sum(2*rc[i][j]**-3 for i in range(4))
        for i in range(4):
            hc[i, 4+j] = hc[4+j, i] = 2*rc[i][j]**-3
    inverse = hc**-1
    y = iv.matrix([[iv.mpf(mp.nstr(inverse[i, j], 40))
                    for j in range(7)] for i in range(7)])
    f0, _ = system(center)
    _, jac = system(box)
    remainder = iv.eye(7)-y*jac
    image = center-y*f0+remainder*iv.matrix([iv.mpf(['-1e-10', '1e-10'])]*7)
    for i in range(7):
        assert image[i].a > box[i].a and image[i].b < box[i].b
        assert abs(image[i]-center[i]).b < iv.mpf('5e-13')
        assert sum(abs(remainder[i, j]) for j in range(7)).b < iv.mpf('4.5e-9')
    # The contraction maps the box into itself. Its unique fixed point solves
    # f=0, because ||I-y*jac||<1 also implies that y is nonsingular.
    r = reciprocal(box)
    assert all(v.a > 0 for row in r for v in row)
    m = [[v**-2 for v in row] for row in r]
    mixed = [abs(r[i][a]-r[i][b]-r[j][a]+r[j][b])
             for i, j in combinations(range(4), 2)
             for a, b in combinations(range(4), 2)]
    assert all(v.a > 0 for v in mixed)
    cubes = []
    for a in range(4):
        for b in range(4):
            t = [[r[i][j]-r[i][b]-r[a][j]+r[a][b]
                  for j in range(4)] for i in range(4)]
            moments = [sum(m[i][j]*t[i][j]**k for i in range(4) for j in range(4))
                       for k in range(1, 4)]
            first, second, third = moments
            cubic = (first**3-3*first*second+2*third)/2
            assert cubic.a > iv.mpf('-.008') and cubic.b < iv.mpf('-.000017')
            cubes.append(cubic)
    assert cubes[0].b < cubes[-1].a  # unequal: excludes critical-unitary realizability
    return {
        'meaning': 'Auxiliary bistochastic data of centered reciprocal rank two; '
                   'NOT the moduli of any critical unitary and NOT an AHC counterexample.',
        'A': A, 'D': D, 'G_denominator': 10**10, 'center': CENTER, 'box_radius': '1e-10',
        'certified_image_radius_bound': '5e-13', 'certified_contraction_bound': '4.5e-9',
        'certified_cubic_range': ['-.008', '-.000017'],
        'cubic_interval_endpoints_rounded_for_display':
            [[float(v.a), float(v.b)] for v in cubes],
        'min_absolute_mixed_difference_lower_bound_rounded': min(float(v.a) for v in mixed),
        'result': 'passed',
    }


def main():
    result = {'saved_return_formula_check': saved_returns_check(),
              'rank_two_only_obstruction': certificate()}
    output = ROOT/'data'/'curvature.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    print('Saved', output)


if __name__ == '__main__':
    main()

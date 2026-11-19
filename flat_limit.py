#!/usr/bin/env python3
"""Exact polynomial checks for the regular flat-limit theorem in the order-four section of the manuscript.

Run: python3 flat_limit.py. Requires SymPy. No numerical search is performed.
All variables are algebraically independent. The geometric support argument
and the asymptotic conclusion are proved in the manuscript.
"""
from itertools import combinations, product
import json
from pathlib import Path
import sympy as s

H = s.Matrix([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
F = s.ones(4, 1).row_join(H)
x = s.symbols('x0:9')
A = s.Matrix(3, 3, x)
h = s.symbols('h0:9')
B = s.Matrix(3, 3, h)


def moments(g, a, b):
    t = s.Matrix(4, 4, lambda i, j: g[i, j]-g[i, b]-g[a, j]+g[a, b])
    return t, [sum(v**k for v in t)/4 for k in range(1, 4)]


def cubic_coefficients(a):
    g = H*a*H.T
    out = []
    for i, j in product(range(4), repeat=2):
        _, m = moments(g, i, j)
        out.append(s.expand(m[2]-s.Rational(3, 2)*m[0]*m[1]+m[0]**3/2))
    return s.Matrix(4, 4, out)


def quartic_coefficients(a, b):
    g, g1 = H*a*H.T, H*b*H.T
    out = []
    for i, j in product(range(4), repeat=2):
        t, m = moments(g, i, j)
        z, _ = moments(g1, i, j)
        n1 = sum(z)/4-sum(u*v for u, v in zip(g, t))/4
        n2 = sum(u*v for u, v in zip(t, z))/2-sum(u*v*v for u, v in zip(g, t))/4
        n3 = 3*sum(u*u*v for u, v in zip(t, z))/4-sum(u*v**3 for u, v in zip(g, t))/4
        out.append(s.expand(n3-s.Rational(3, 2)*(m[0]*n2+n1*m[1])
                            +s.Rational(3, 2)*m[0]**2*n1))
    return s.Matrix(4, 4, out)


def check():
    assert H.T*H == 4*s.eye(3)
    assert H.T*s.ones(4, 1) == s.zeros(3, 1)
    c = cubic_coefficients(A)
    transformed = F.T*c*F/16
    for k in range(3):
        assert s.expand(transformed[k+1, 0]+96*s.prod(A[k, j] for j in range(3))) == 0
        assert s.expand(transformed[0, k+1]+96*s.prod(A[i, k] for i in range(3))) == 0
    g = H*A*H.T
    pairs = list(combinations(range(4), 2))
    roots = []
    for a, b in pairs:
        v = (H.row(a)-H.row(b)).T/2
        z = A*v
        d = [g[i, a]-g[i, b] for i in range(4)]
        leading = sum(s.prod(d[i]-d[j] for j in range(4) if i != j)
                      for i in range(4))/16
        assert s.expand(leading-32*s.prod(z)) == 0
        roots.append(tuple(v))
    # Transposition gives exactly the same checks for row pairs.
    a, b, c0, d0 = s.symbols('a b c d')
    block = s.Matrix([[a, b, 0], [c0, d0, 0], [0, 0, 0]])
    tc = F.T*cubic_coefficients(block)*F/16
    for i, j, expected in [(1, 1, 48*b*c0*d0), (1, 2, 48*a*c0*d0),
                           (2, 1, 48*a*b*d0), (2, 2, 48*a*b*c0)]:
        assert s.expand(tc[i, j]-expected) == 0
    diagonal = s.diag(a, b, 0)
    qd = F.T*quartic_coefficients(diagonal, B)*F/16
    assert s.expand(qd[3, 3]+48*a*b*(a*a+b*b)) == 0
    line = s.Matrix([[a, b, 0], [0, 0, 0], [0, 0, 0]])
    ql = F.T*quartic_coefficients(line, B)*F/16
    assert s.expand(ql[0, 0]-32*(a**4+b**4)) == 0
    assert cubic_coefficients(diagonal) == s.zeros(4)
    assert cubic_coefficients(line) == s.zeros(4)
    return {'arithmetic': 'exact rational polynomial arithmetic in SymPy',
            'leading_pair_roots': [list(map(int, v)) for v in roots],
            'leading_support': 'At most two coefficients; support proof is in the manuscript.',
            'different_rows_and_columns_obstruction': '-48*a*b*(a^2+b^2)',
            'surviving_common_quartic': '32*(a^4+b^4)',
            'scope': 'C^2 modulus-only paths through J4 with nonzero first derivative; '
                     'not a global AHC proof or a uniform neighborhood theorem.',
            'result': 'passed'}


if __name__ == '__main__':
    report = check()
    path = Path(__file__).resolve().parent/'data'/'flat_limit.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

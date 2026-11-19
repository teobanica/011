"""Exact modulus identities for the order-four section of the manuscript. Run with Python 3 and SymPy."""
from itertools import combinations, permutations

from sympy import I, Matrix, Poly, Symbol, conjugate, expand, eye, factor
from sympy import interpolate, ones, prod, simplify, sqrt, symbols


def denominators(nodes):
    return [prod(t-u for j, u in enumerate(nodes) if i != j)
            for i, t in enumerate(nodes)]


def group_checks():
    a, b, c, d, m, x, y, t = symbols('a b c d m x y t', real=True)
    P = eye(4) - ones(4)/4
    cyclic = Matrix([[a, b, c, d], [d, a, b, c], [c, d, a, b], [b, c, d, a]])
    klein = Matrix([[a, b, c, d], [b, a, d, c], [c, d, a, b], [d, c, b, a]])
    for R, charpoly in (
        (cyclic, t*(t-a+b-c+d)*((t-a+c)**2+(b-d)**2)),
        (klein, t*(t-a-b+c+d)*(t-a+b-c+d)*(t-a+b+c-d)),
    ):
        # charpoly creates its own variable; use its coefficients directly.
        assert (P*R*P).charpoly().all_coeffs() == Poly(expand(charpoly), t).all_coeffs()
    values = [m+x, m+y, m-x, m-y]
    nodes = [values[i]-values[(i+1)%4] for i in range(4)]
    D = denominators(nodes)
    cleared = sum(D[i]*values[(i+2)%4]**2*values[(i+3)%4]**2 for i in range(4))
    assert expand(cleared-64*m**3*x*y*(x-y)*(x+y)) == 0
    print('cyclic factorization:', factor(cleared))


def matching_check():
    identity = tuple(range(4))
    derangements = [p for p in permutations(identity) if all(i != p[i] for i in identity)]
    counts = {'cyclic': 0, 'Klein': 0}
    for triple in combinations(derangements, 3):
        group = (identity,) + triple
        if any(len({p[i] for p in group}) != 4 for i in identity):
            continue
        assert all(tuple(p[q[i]] for i in identity) in group for p in group for q in group)
        cyclic = any(any(p[p[i]] != i for i in identity) for p in triple)
        counts['cyclic' if cyclic else 'Klein'] += 1
    assert counts == {'cyclic': 3, 'Klein': 1}
    print('normalized matching decompositions:', counts)


def pair_check():
    t = Symbol('t', real=True)
    nodes = [-2, -1, 1, 2]
    D = denominators(nodes)
    A, B = 1+2*I, 2-I
    z = [(A+B*d)/D_i for d, D_i in zip(nodes, D)]
    assert sum(z).expand() == 0
    assert expand(sum(d*v for d, v in zip(nodes, z))) == 0
    w2 = [simplify(v*conjugate(v)) for v in z]
    assert sum(v*den for v, den in zip(w2, D)) == 0
    Q = Poly(interpolate([(d, w*den**2) for d, w, den in zip(nodes, w2, D)], t), t)
    a2, a1, a0 = Q.all_coeffs()
    assert 4*a0*a2 > a1**2
    B1 = sqrt(a2)
    A1 = a1/(2*B1) + I*sqrt(a0-a1**2/(4*a2))
    reconstructed = [(A1+B1*d)/den for d, den in zip(nodes, D)]
    ratio = simplify(reconstructed[0]/z[0])
    assert simplify(ratio*conjugate(ratio)) == 1
    assert all(simplify(r-ratio*v) == 0 for r, v in zip(reconstructed, z))
    print('pair reconstruction verified exactly:', Q.as_expr())


if __name__ == '__main__':
    group_checks()
    matching_check()
    pair_check()

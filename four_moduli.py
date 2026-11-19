"""Exact four-symbol enumeration and algebra for the order-four section of the manuscript, with Python 3 and SymPy."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations, product
from math import gcd

import sympy as sp


PAIRS = list(combinations(range(4), 2))
PERMS = list(permutations(range(4)))
X, Y, Z = sp.symbols('x y z')
A, B, C = sp.symbols('a b c')
LEVELS = (sp.Integer(1), X, Y, Z)
SQUARES = tuple(1/r**2 for r in LEVELS)
LAST = ((0, 2, 3, 3), (2, 1, 2, 3), (3, 2, 1, 2), (3, 3, 2, 0))


def profile_families():
    profiles = list(combinations_with_replacement(range(4), 4))
    counts = [tuple(sum(s >= k for s in row) for k in (1, 2, 3)) for row in profiles]
    families = set()
    for a, b, c in product(range(1, 33), repeat=3):
        if gcd(gcd(a, b), c) > 1:
            continue
        groups = defaultdict(list)
        for row, (i, j, k) in zip(profiles, counts):
            groups[a*i+b*j+c*k].append(row)
        families.update(frozenset(g) for g in groups.values()
                        if len(g) >= 2 and len(set().union(*g)) == 4)
    maximal = [g for g in families if not any(g < other for other in families)]
    return families, maximal


def compatible(r, s):
    # All diagonal symbol pairs represent the same (zero) difference.
    return len({(a, b) if a != b else (-1, -1) for a, b in zip(r, s)}) == 4


def arrays(family):
    words = {row for profile in family for row in permutations(profile)}
    prefixes = [{row[:n] for row in words} for n in range(5)]
    mates = {r: {s for s in words if compatible(r, s)} for r in words}
    for r in family:  # Permute the columns until the first row is sorted.
        for s in mates[r]:
            if not all(col in prefixes[2] for col in zip(r, s)):
                continue
            for t in mates[r] & mates[s]:
                if t <= s or not all(col in prefixes[3] for col in zip(r, s, t)):
                    continue
                for u in mates[r] & mates[s] & mates[t]:
                    if u <= t:
                        continue
                    rows = (r, s, t, u)
                    cols = tuple(zip(*rows))
                    if not all(col in words for col in cols):
                        continue
                    if len({tuple(sorted(row)) for row in rows+cols}) < 2:
                        continue
                    if len(set().union(*rows)) == 4:
                        yield rows


def canonical(rows):
    # Lexicographic minimum over row/column permutations and transposition.
    return min(tuple(sorted(tuple(r[i] for i in p) for r in matrix))
               for matrix in (rows, tuple(zip(*rows))) for p in PERMS)


def norm_polynomial(r, s):
    difference = sum(SQUARES[k] for k in r)-sum(SQUARES[k] for k in s)
    return sp.together(difference).as_numer_denom()[0]


def rank_polynomial(rows):
    r = [[LEVELS[a] for a in row] for row in rows]
    m = sp.Matrix(3, 3, lambda i, j: r[i+1][j+1]-r[0][j+1]-r[i+1][0]+r[0][0])
    return m.det()


@lru_cache(None)
def pair_polynomial(r, s):
    nodes = [LEVELS[a]-LEVELS[b] for a, b in zip(r, s)]
    weights = [SQUARES[a]*SQUARES[b] for a, b in zip(r, s)]
    common = sp.lcm([w.as_numer_denom()[1] for w in weights])
    return sum(sp.cancel(w*common)*sp.prod(d-e for j, e in enumerate(nodes) if j != i)
               for i, (d, w) in enumerate(zip(nodes, weights)))


@lru_cache(None)
def fixed_sign(factor):
    gap_variables = {X: 1+A, Y: 1+A+B, Z: 1+A+B+C}
    coefficients = sp.Poly(factor.xreplace(gap_variables), A, B, C).coeffs()
    return min(coefficients) > 0 or max(coefficients) < 0


def necessary_factor(equation):
    """Discard only nonvanishing factors and redundant powers on 1 < x < y < z."""
    if equation == 0 or sp.expand(equation) == 0:
        return sp.Integer(0)
    return sp.prod(f for f, _ in sp.factor_list(equation)[1] if not fixed_sign(f))


def last_case(rows, rank):
    assert rows == LAST
    assert rank == X*Z-X+Y**2-3*Y*Z+Y+Z**2
    h, t = sp.symbols('h t')
    r = (1-h, 1-h*(t+t**2), 1-h*t, 1)
    # Multiply a homogeneous rescaling of F(1,x,y,z) through by r[0]^degree.
    def specialize(f):
        poly = sp.Poly(f, X, Y, Z)
        degree = poly.total_degree()
        return sum(coeff*r[0]**(degree-i-j-k)*r[1]**i*r[2]**j
                   for (i, j, k), coeff in poly.terms())

    assert sp.expand(specialize(rank)) == 0
    eqs = [norm_polynomial(rows[0], rows[1]),
           pair_polynomial(rows[0], rows[1]), pair_polynomial(rows[0], rows[2])]
    reduced = []
    for eq in eqs:
        factors = sp.factor_list(specialize(necessary_factor(eq)))[1]
        # These explicit factors are nonzero since 0<h<1 and 0<t<1.
        retained = [f for f, _ in factors if f not in (h, t-1, t+1)]
        assert len(retained) == 1
        reduced.append(retained[0])
    f, g, k = reduced
    assert [sp.degree(p, h) for p in reduced] == [5, 6, 6]
    res1, res2 = (sp.resultant(f, p, h) for p in (g, k))
    common = sp.Poly(sp.gcd(res1, res2), t).monic().as_expr()
    assert sp.expand(common-t**26*(t**2+t-1)**4) == 0


def exclude(rows):
    cols = tuple(zip(*rows))
    rank = necessary_factor(rank_polynomial(rows))
    if rank == 1:
        return 'norm or rank'
    if any(necessary_factor(norm_polynomial(rows[0], r)) == 1 for r in rows+cols):
        return 'norm or rank'
    pairs = [necessary_factor(pair_polynomial(matrix[i], matrix[j]))
             for matrix in (rows, cols) for i, j in PAIRS]
    if 1 in pairs:
        return 'pair'
    if sp.degree(rank, X) == 1 and rank.coeff(X).is_Integer and rank.coeff(X) != 0:
        substitution = sp.solve(rank, X)[0]
        if any(necessary_factor(sp.together(p.subs(X, substitution)).as_numer_denom()[0]) == 1
               for p in pairs if p != 0):
            return 'linear substitution'
    last_case(rows, rank)
    return 'resultant'


if __name__ == '__main__':
    families, maximal = profile_families()
    assert (len(families), len(maximal)) == (157, 93)
    candidates = sorted({canonical(rows) for g in maximal for rows in arrays(g)})
    assert len(candidates) == 27
    tally = Counter()
    for rows in candidates:
        reason = exclude(rows)
        print(' '.join(''.join(map(str, row)) for row in rows), reason, flush=True)
        tally[reason] += 1
    assert tally == {'norm or rank': 10, 'pair': 13, 'linear substitution': 3, 'resultant': 1}
    print('157 families, 93 maximal, 27 arrays.', dict(tally))

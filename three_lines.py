"""Checks for the three-line theorem. Run with Python, NumPy and mpmath."""
from itertools import combinations

import numpy as np
from mpmath import iv
from models import bordered, critical_corner


def sharp(a):
    return np.array([[a[1, 1], -a[1, 0]], [-a[0, 1], a[0, 0]]]).conj()


def positive_root(a):
    eig, vectors = np.linalg.eigh(a)
    assert eig.min() > 0
    return (vectors * np.sqrt(eig)) @ vectors.conj().T


def complete(a, v, w):
    left = positive_root(np.eye(2) - a @ a.conj().T)
    right = positive_root(np.eye(2) - a.conj().T @ a)
    return np.block([[a, left @ v], [w @ right, -w @ a.conj().T @ v]])


def curve(u, t):
    a, b, c, d = u[:2, :2], u[:2, 2:], u[2:, :2], u[2:, 2:]
    v = np.linalg.solve(positive_root(np.eye(2) - a @ a.conj().T), b)
    right = positive_root(np.eye(2) - a.conj().T @ a)
    w = np.linalg.solve(right.T, c.T).T
    moved = complete(np.cos(t)*a + 1j*np.sin(t)*sharp(a), v, w)
    eta = np.linalg.det(w) * np.linalg.det(v)
    predicted_d = np.cos(t)*d - 1j*np.sin(t)*eta*sharp(d)
    assert np.linalg.norm(moved[2:, 2:] - predicted_d) < 1e-12
    assert np.linalg.norm(moved.conj().T @ moved - np.eye(4)) < 1e-12
    return moved


def rotation(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def sanity_checks():
    # Two strict contractions, one singular, tested with nonreal off-diagonal blocks.
    v = rotation(.41) @ np.diag([1, np.exp(.63j)]) @ rotation(.53)
    w = rotation(.39) @ np.diag([1, np.exp(-.71j)]) @ rotation(.49)
    for low in [0., .33]:
        a = rotation(.37) @ np.diag([.74, low]) @ rotation(-.54)
        u = complete(a, v, w)
        assert abs(u).min() > 1e-3
        for t in [-.2, 0., .2]:
            curve(u, t)

    # The same real rank-one diagonal blocks can have either behavior in D.
    p, q, r, s = [rotation(t) for t in (.37, .54, .39, .53)]
    a = p @ np.diag([.74, 0.]) @ q.T
    v = p @ np.diag([1, np.exp(.63j)]) @ s.T
    step = .0001
    for extra in (0., .31):
        w = r @ np.diag([1, np.exp(1j*(extra-.63))]) @ q.T
        sample = complete(a, v, w)
        rates = (abs(curve(sample, step))**2 - abs(curve(sample, -step))**2)/(2*step)
        assert np.linalg.norm(sample[2:, 2:].imag) < 1e-12
        assert np.linalg.norm(rates[:2, :]) + np.linalg.norm(rates[2:, :2]) < 1e-10
        if extra == 0.:
            assert np.linalg.norm(rates[2:, 2:]) < 1e-10
        else:
            assert np.linalg.norm(rates[2:, 2:]) > .01

    u = bordered(critical_corner(), 0.)
    swap = [1, 0, 3, 2]
    gains = abs(u[np.ix_(swap, swap)])**2 - abs(u)**2
    expected = (gains/abs(u)).sum()
    step = .0002
    plus, minus = curve(u, step), curve(u, -step)
    assert np.max(abs(abs(plus)**2 - abs(minus)**2)) < 1e-12
    observed = (abs(plus).sum()+abs(minus).sum()-2*abs(u).sum())/step**2
    assert abs(expected - observed) < 2e-5
    print('Four-column curvature, formula and symmetric difference:', expected, observed)


def pair_curvature(u, pair):
    factors = []
    scale = shift = iv.mpf(0)
    for row in u:
        v, w = row[pair[0]], row[pair[1]]
        p, q = abs(v), abs(w)
        h = iv.mpc(v.real, -v.imag)*w/(p*q)
        d = q*q/p + p*p/q
        factors.append(d*h**2)
        scale += d
        shift += p+q
    return (scale + abs(sum(factors)))/4 - shift/2


def interval_check():
    iv.dps = 45
    lo, hi = '0.60960978900', '0.60960978902'

    def deriv(x):
        return 2 - 2*iv.sqrt(3)*x/iv.sqrt(1-x**2) + 2*x/iv.sqrt(x**2+3)

    assert deriv(iv.mpf(lo)) > 0 and deriv(iv.mpf(hi)) < 0
    x = iv.mpf([lo, hi])
    border = iv.sqrt((1-x**2)/3)
    u = [[iv.mpc(border) for _ in range(4)] for _ in range(4)]
    u[0][0] = iv.mpc(x)
    inner = [iv.mpc(-x/3), iv.mpc(-x/3, 1/iv.sqrt(3)), iv.mpc(-x/3, -1/iv.sqrt(3))]
    for i in range(3):
        for j in range(3):
            u[i+1][j+1] = inner[(j-i) % 3]

    columns = pair_curvature(u, (1, 2))
    singleton_rows = pair_curvature(list(zip(*u)), (0, 3))
    assert iv.mpf('.6092') < columns < iv.mpf('.6094')
    assert iv.mpf('-.6677') < singleton_rows < iv.mpf('-.6674')
    # For columns 2,3 (one-based), rows 2,3 have identical phase quotients
    # by the explicit formula. Rows 1,2,4 represent three distinct lines.
    products = [iv.mpc(row[1].real, -row[1].imag)*row[2] for row in u]
    for j, k in combinations([0, 1, 3], 2):
        test = iv.mpc(products[j].real, -products[j].imag)*products[k]
        assert abs(test.imag) > iv.mpf('0.001')

    total = iv.mpf(0)
    swap = [1, 0, 3, 2]
    for i in range(4):
        for j in range(4):
            if (i,j) < (swap[i],swap[j]):
                a, b = abs(u[i][j]), abs(u[swap[i]][swap[j]])
                total += (a-b)**2*(a+b)/(a*b)
    assert iv.mpf('1.4478') < total < iv.mpf('1.4480')
    print('Certified block-curve curvature:', total)
    print('Certified column and singleton-row curvatures:', columns, singleton_rows)


if __name__ == '__main__':
    sanity_checks()
    interval_check()

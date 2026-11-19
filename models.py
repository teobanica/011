"""Check two analytic saddles against the saved Haar representatives."""
from itertools import permutations
from math import cos, sin, pi, sqrt
from pathlib import Path
import csv
import numpy as np

DATA = Path(__file__).parent/'data'
PERMS = list(permutations(range(4)))


def dephase(U):
    V = U*U[0, :].conj()/abs(U[0, :])
    return V*V[:, :1].conj()/abs(V[:, :1])


def real_saddle():
    a, b = cos(pi/8)/sqrt(2), sin(pi/8)/sqrt(2)
    return np.array([[a,b,a,b], [a,-b,-b,-a], [b,a,-a,b], [b,-a,-b,a]], dtype=complex)


def bordered(x, t):
    U = np.full((4, 4), sqrt((1-x*x)/3), dtype=complex)
    U[0, 0] = x
    row = (-x + 2j*np.sin(t+2*pi*np.arange(3)/3))/3
    for j in range(3):
        U[j+1, 1:] = np.roll(row, j)
    return U


def critical_corner():
    left, right = 0., 1.
    for _ in range(64):
        x = (left+right)/2
        d = 2 - 2*sqrt(3)*x/sqrt(1-x*x) + 2*x/sqrt(x*x+3)
        if d > 0:
            left = x
        else:
            right = x
    return (left+right)/2


def report():
    with (DATA/'haar.csv').open() as f:
        rows = list(csv.DictReader(f))
    raw = np.load(DATA/'haar.npz')['finals']
    lines = []
    for model in [real_saddle(), bordered(critical_corner(), 0.)]:
        S = model/abs(model)
        X = S.conj().T @ model
        assert np.linalg.norm(X-X.conj().T) < 1e-12
        assert np.linalg.norm(model.conj().T @ model-np.eye(4)) < 1e-12
        norm = abs(model).sum()
        row = next(r for r in rows if r['status']=='nonflat_uphill' and abs(float(r['L'])-norm)<1e-9)
        idx = int(row['id'])
        target = dephase(raw[idx])
        target_norm = lambda M: np.max(abs(target - M))
        err = min(target_norm(dephase(A[np.ix_(p, q)])) for A in (model, model.conj())
                  for p in PERMS for q in PERMS)
        assert err < 1e-8
        lines.append(f'L={norm:.14f} Haar id={idx} max entrywise distance after equivalences={err:.3g}')
    (DATA/'model_checks.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    report()

"""Verify uphill neighborhoods, record representatives, and polish slow ascent controls."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import numpy as np
from mpmath import iv
from search import EDGES, components, expstep, refine, residual

OUT = Path(__file__).parent / 'data'
iv.dps = 35
BOX = iv.mpf('1.e-8') * iv.mpf([-1, 1])
FAMILIES = ['haar', 'phase', 'near_real', 'near_flat']


def best_pair(U):
    curvatures = []
    for j, k in EDGES:
        z, w = U[:, j], U[:, k]
        p, q = abs(z), abs(w)
        h = z.conj()*w/(p*q)
        d = q*q/p + p*p/q
        curvatures.append((sum(d) + abs(sum(d*h*h)))/4 - sum(p+q)/2)
    return EDGES[int(np.argmax(curvatures))], max(curvatures)


def enclose_pair(U, pair):
    total = penalty = real_sum = imag_sum = iv.mpf(0)
    for z, w in U[:, pair]:
        x, y, a, b = [iv.mpf(float(t)) + BOX for t in (z.real, z.imag, w.real, w.imag)]
        p, q = iv.sqrt(x**2 + y**2), iv.sqrt(a**2 + b**2)
        d = q**2/p + p**2/q
        re, im = (x*a + y*b)/(p*q), (x*b - y*a)/(p*q)
        real_sum += d*(re**2 - im**2)
        imag_sum += 2*d*re*im
        total += d
        penalty += p + q
    return (total + iv.sqrt(real_sum**2 + imag_sum**2))/4 - penalty/2


def audit():
    levels = defaultdict(list)
    representatives = {}
    with (OUT/'certified_pairs.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['family', 'id', 'column_j', 'column_k', 'curvature_lower_bound'])
        for family in FAMILIES:
            raw = np.load(OUT/(family+'.npz'))
            with (OUT/(family+'.csv')).open() as f:
                rows = list(csv.DictReader(f))
            print(family, dict(Counter(row['status'] for row in rows)), flush=True)
            for row in rows:
                idx = int(row['id'])
                U = raw['finals'][idx]
                if row['status'] in ('near_zero', 'unresolved'):
                    assert abs(U).min() < .001
                if row['status'] != 'nonflat_uphill':
                    continue
                level = round(float(row['L']), 6)
                (j, k), curvature = best_pair(U)
                assert abs(curvature-float(row['best_column_pair'])) < 1e-11
                enc = enclose_pair(U, (j, k))
                assert enc > iv.mpf('0.60')
                down = int(float(enc.a)*1000) - 1
                assert enc > iv.mpf(down)/1000
                writer.writerow([family, idx, j+1, k+1, down/1000])
                inertia = int((raw['spectra'][idx] > 1e-7).sum())
                levels[level].append((curvature, inertia, float(row['h9'])))
                if level not in representatives:
                    representatives[level] = (family, idx, U, raw['directions'][idx], row)
    for level, items in sorted(levels.items()):
        print(level, len(items), np.min(items, axis=0), np.max(items, axis=0))
    with (OUT/'representatives.txt').open('w') as f:
        for level, (family, idx, U, B, row) in sorted(representatives.items()):
            step = 1.e-3
            # Independent symmetric differences of the actual objective.
            Uplus, Uminus = [expstep(U[None], sign*step*B[None])[0] for sign in (1, -1)]
            second = (abs(Uplus).sum()+abs(Uminus).sum()-2*abs(U).sum())/step**2
            assert abs(second - float(row['h9'])) < 1e-3
            Udeph = U*U[0, :].conj()/abs(U[0, :])
            Udeph *= Udeph[:, :1].conj()/abs(Udeph[:, :1])
            f.write(f"{level}, {family} #{idx}, horizontal eigenvalues\n")
            f.write(' '.join(row[f'h{k}'] for k in range(1, 10))+'\n')
            f.write(f"central second difference at step {step} = {second:.11f}\n")
            f.write(np.array2string(Udeph, precision=13, suppress_small=True)+'\n\n')


def polish_controls():
    with (OUT/'polar.csv').open() as f:
        rows = list(csv.DictReader(f))
    ids = np.array([int(r['id']) for r in rows if r['status'] == 'unresolved'])
    U = np.load(OUT/'polar.npz')['finals'][ids]
    V, steps = refine(U)
    errors = abs(abs(V)-.5).max(axis=(1, 2))
    gradients = residual(components(V)[2])
    print('polar polish', len(ids), errors.max(), gradients.max())
    with (OUT/'polar_polish.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['polar_id', 'iterations', 'flat_error', 'gradient'])
        writer.writerows(zip(ids, steps, errors, gradients))
    np.savez_compressed(OUT/'polar_polish.npz', ids=ids, finals=V)


if __name__ == '__main__':
    audit()
    polish_controls()

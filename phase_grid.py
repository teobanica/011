"""Direct exhaustive audit of the dephased order-four phase grid. Integer arithmetic only."""
from itertools import combinations
from pathlib import Path
import csv
import numpy as np

PAIRS = list(combinations(range(4), 2))
WEIGHTS = 4**np.arange(9)
OUT = Path(__file__).parent / 'data'


def classify(exponents):
    # Columns in each difference contain exponents of the four phase quotients.
    valid = np.ones(len(exponents), dtype=bool)
    orthogonal = valid.copy()
    for a in [exponents, exponents.transpose(0, 2, 1)]:
        for j, k in PAIRS:
            differences = (a[:, :, j] - a[:, :, k]) % 4
            counts = np.stack([(differences == r).sum(axis=1) for r in range(4)], axis=1)
            real0, imag0, real2, imag2 = counts.T
            valid &= ((real0 > 0) == (real2 > 0)) & ((imag0 > 0) == (imag2 > 0))
            orthogonal &= (real0 == real2) & (imag0 == imag2)
    real = (exponents % 2 == 0).all(axis=(1, 2))
    assert np.all(~valid | real | orthogonal)
    return np.select([~valid, real & orthogonal, real, orthogonal],
                     ['impossible', 'real_hadamard', 'real_other', 'nonreal_hadamard'],
                     default='unclassified')


def run():
    OUT.mkdir(exist_ok=True)
    counts = {}
    with (OUT/'phase_grid.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'screen'])
        for start in range(0, 4**9, 2048):
            ids = np.arange(start, start + 2048)
            exponents = np.zeros((len(ids), 4, 4), dtype=int)
            exponents[:, 1:, 1:] = ((ids[:, None] // WEIGHTS) % 4).reshape(-1, 3, 3)
            categories = classify(exponents)
            writer.writerows(zip(ids, categories))
            for tag, count in zip(*np.unique(categories, return_counts=True)):
                counts[str(tag)] = counts.get(str(tag), 0) + int(count)
    assert counts == dict(impossible=261952, real_hadamard=6,
                          real_other=168, nonreal_hadamard=18)
    print(counts)


if __name__ == '__main__':
    run()

"""Deterministic order-four critical-point and gradient-ascent surveys."""
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
import csv
import numpy as np

N = 4
EDGES = list(combinations(range(N), 2))
BASIS = list(np.array([np.diag(e) for e in np.eye(N)], dtype=complex))
for j, k in EDGES:
    e = np.zeros((N, N), complex)
    e[j, k] = 1
    BASIS.extend([(e + e.T) / np.sqrt(2), 1j * (e - e.T) / np.sqrt(2)])
BASIS = np.array(BASIS)
OFF = BASIS[N:]
TOL = 1e-12
MIN_ENTRY = 1e-8
FLAT_TOL = 1e-6


def adj(x):
    return np.swapaxes(x.conj(), -2, -1)


def haar(rng, count, real=False):
    z = rng.standard_normal((count, N, N))
    if not real:
        z = z + 1j * rng.standard_normal(z.shape)
    q, r = np.linalg.qr(z)
    diag = np.diagonal(r, axis1=-2, axis2=-1)
    return q * (diag / abs(diag))[:, None, :]


def expstep(U, B):
    w, v = np.linalg.eigh(B)
    return U @ ((v * np.exp(1j*w)[:, None, :]) @ adj(v))


def components(U):
    r = abs(U)
    S = U / np.maximum(r, 1e-300)
    X = adj(S) @ U
    R = (X-adj(X))/(2j)
    return S, X, R


def horizontal(U):
    # Off-diagonal parts of the first three left-phase generators U* E_jj U.
    generators = adj(U)[:, None] @ BASIS[:N-1] @ U[:, None]
    coords = np.einsum('bkij,aji->bak', generators, OFF).real
    return np.linalg.qr(coords, mode='complete')[0][:, :, N-1:]


def residual(R):
    return np.linalg.norm(R, axis=(1, 2))


def refine(starts, max_iter=100):
    U = starts.copy()
    count = len(U)
    damping = np.full(count, .01)
    steps = np.zeros(count, int)
    for _ in range(max_iter):
        S, X, R = components(U)
        old = residual(R)
        active = np.flatnonzero((old > TOL) & (abs(U).min(axis=(1,2)) > 1e-12) & (damping < 1e7))
        if not len(active):
            break
        V, S, X, R = U[active], S[active], X[active], R[active]
        C = horizontal(V)
        VB = V[:, None] @ OFF
        radial = (VB*S[:, None].conj()).real / abs(V[:, None])
        # At tangent i V B, dS = i S Re((VB) conj(S))/|V|.
        dX = 1j * (X[:, None] @ OFF - adj(S[:, None]*radial) @ V[:, None])
        dR = (dX - adj(dX))/(2j)
        J = np.einsum('bkij,aji->bak', dR, BASIS).real @ C
        rhs = np.einsum('bij,aji->ba', R, BASIS).real
        left, sv, right_t = np.linalg.svd(J, full_matrices=False)
        factor = sv/(sv**2 + damping[active, None]**2)
        delta = -(adj(right_t) @ ((adj(left) @ rhs[..., None]) * factor[..., None]))
        coord = (C @ delta)[..., 0]
        coord /= np.maximum(1, np.linalg.norm(coord, axis=1, keepdims=True)/.65)
        proposal = expstep(V, np.einsum('ba,aij->bij', coord, OFF))
        new = residual(components(proposal)[2])
        good = new < old[active]
        U[active[good]] = proposal[good]
        damping[active] = np.where(good, np.maximum(1e-13, damping[active]*.45), damping[active]*4)
        steps[active] += 1
    return U, steps


def ascent(starts, max_iter=3000):
    U = starts.copy()
    count = len(U)
    steps = np.zeros(count, int)
    for _ in range(max_iter):
        S, X, R = components(U)
        spread = np.max(abs(abs(U)-.5),axis=(1,2))
        active = np.flatnonzero((spread > FLAT_TOL) & (residual(R) > TOL))
        if not len(active):
            break
        # Monotone alternating maximization. Polar(S) maximizes Re tr(S*V).
        left, sv, right = np.linalg.svd(S[active])
        U[active] = left @ right
        steps[active] += 1
    return U, steps


def measure(U):
    V = U[None]
    S, X, R = components(V)
    turn = V[:, None] @ OFF
    p = (turn*S[:, None].conj()).real / np.sqrt(np.maximum(abs(V[:, None]),1e-300))
    p = p.reshape(12, 16)
    Y = (X[0]+adj(X[0]))/2
    H = p @ p.T - np.einsum('ij,ajk,bki->ab', Y, OFF, OFF, optimize=True).real
    C = horizontal(V)[0]
    H = (H+H.T)/2
    G = C.T @ H @ C
    eig, vec = np.linalg.eigh(G)
    direction = np.einsum('a,aij->ij', C @ vec[:,-1], OFF)
    pairs = [np.linalg.eigvalsh(H[i:i+2,i:i+2])[-1] for i in range(0,12,2)]
    rowS = U[0,:].conj()/np.maximum(abs(U[0,:]),1e-300)
    D = U*rowS
    D *= D[:,:1].conj()/np.maximum(abs(D[:,:1]),1e-300)
    metrics = [abs(U).sum(), residual(R)[0], abs(U).min(), np.max(abs(abs(U)-.5)),
               np.max(abs(D.imag)), np.linalg.eigvalsh(Y)[0], max(pairs),
               np.linalg.norm(adj(U)@U-np.eye(N))]
    return metrics, eig, direction


def make_starts(rng, size, kind):
    if kind in ('haar','polar'):
        return haar(rng,size)
    if kind=='phase':
        S = np.exp(2j*np.pi*rng.random((size,N,N)))
        left, sv, right = np.linalg.svd(S)
        return left @ right
    if kind=='near_real':
        U = haar(rng,size,True).astype(complex)
    elif kind=='near_flat':
        phases = np.exp(2j*np.pi*rng.random(size))
        U = np.array([[[1,1,1,1],[1,z,-1,-z],[1,-1,1,-1],[1,-z,-1,z]] for z in phases])/2
    else:
        raise ValueError(kind)
    weights = rng.standard_normal((size,len(BASIS)))
    weights *= (10**rng.uniform(-2, -.05,size)/np.linalg.norm(weights,axis=1))[:,None]
    return expstep(U, np.einsum('ba,aij->bij',weights,BASIS))


def run(kind, count, seed, output):
    output.mkdir(exist_ok=True, parents=True)
    starts = make_starts(np.random.default_rng(seed),count,kind)
    finals, directions, spectra = [], [], []
    path = output / kind
    with path.with_suffix('.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id','status','iterations','L','gradient','min_modulus','flat_error',
                         'dephased_max_imag','min_eig_X','best_column_pair','unitarity_error',
                         *['h'+str(i) for i in range(1,10)]])
        for j in range(0,count,48):
            batch = starts[j:j+48]
            U, iters = ascent(batch) if kind=='polar' else refine(batch)
            for k, matrix in enumerate(U):
                stats, spec, direction = measure(matrix)
                L, g, minimum, flat = stats[:4]
                if minimum <= MIN_ENTRY:
                    status = 'near_zero'
                elif flat <= FLAT_TOL:
                    status = 'near_flat'
                elif g > TOL:
                    status = 'unresolved'
                elif spec[-1] > 1e-7:
                    status = 'nonflat_uphill'
                else:
                    status = 'candidate'
                writer.writerow([j+k,status,iters[k],*stats,*spec])
                finals.append(matrix)
                directions.append(direction)
                spectra.append(spec)
            stream.flush()
    np.savez_compressed(path.with_suffix('.npz'),starts=starts,finals=np.array(finals),
                        spectra=np.array(spectra),directions=np.array(directions))


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('kind', choices=['haar','phase','near_real','near_flat','polar'])
    parser.add_argument('count', type=int)
    parser.add_argument('seed', type=int)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'data')
    opts=parser.parse_args()
    run(opts.kind, opts.count, opts.seed, opts.output)

#!/usr/bin/env python3
"""Check archive integrity and recompute diagnostics at every retained endpoint.

Run from code011/: python3 audit_compact.py
Requires NumPy. Writes overnight/archive-checks.json after both archives pass.
The source databases are not required. This is a floating-point recheck, not
an interval certificate or a recheck of discarded ordinary endpoints.
"""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import tempfile
import zipfile

import numpy as np
from compact import verify_zip

HASHES = {
    'ascent': '7923fe1d2e433e5fd1aad9c419ba17fa226a06b89af95e94ac156efdc2a81d17',
    'stationary': '136cd8b4fbe86f48fb88a725abe8b59bb45d91b7efafad5ae60777cf2826bce4',
}


def adj(a):
    return a.conj().swapaxes(-1, -2)


def basis():
    matrices = [np.diag(e).astype(complex) for e in np.eye(4)]
    for j in range(4):
        for k in range(j+1, 4):
            e = np.zeros((4, 4), complex)
            e[j, k] = 1
            matrices.extend([(e+e.T)/np.sqrt(2), 1j*(e-e.T)/np.sqrt(2)])
    return np.array(matrices)


B = basis()


def spectra(u):
    """Polarize the manuscript's second variation; use SVD for phase removal."""
    s = u/abs(u)
    x = adj(s) @ u
    y = (x+adj(x))/2
    w = ((u[:, None] @ B)*s[:, None].conj()).real
    w = (w/np.sqrt(abs(u))[:, None]).reshape(-1, 16, 16)
    h = w @ w.swapaxes(1, 2) - np.einsum('nij,ajk,bki->nab', y, B, B).real
    h = (h+h.swapaxes(1, 2))/2
    left = adj(u)[:, None] @ B[:3] @ u[:, None]
    coords = np.einsum('nkij,aji->nka', left, B[4:]).real
    _, _, vt = np.linalg.svd(coords, full_matrices=True)
    c = vt[:, 3:, :].swapaxes(1, 2)
    reduced = c.swapaxes(1, 2) @ h[:, 4:, 4:] @ c
    return np.linalg.eigvalsh(h), np.linalg.eigvalsh(reduced)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(path):
    report = verify_zip(path)
    mode = report['mode']
    require(report['config']['sha256'] == HASHES[mode], 'Unexpected search source')
    names = ('L', 'gradient', 'min_modulus', 'flat_error', 'unitarity_error')
    maxima = dict.fromkeys(names, 0.0)
    spectral_abs = dict.fromkeys(('full', 'horizontal'), 0.0)
    spectral_scaled = spectral_abs.copy()
    counts = Counter()
    nspectra = 0
    max_start_unity = max_end_unity = 0.0
    with tempfile.TemporaryDirectory(prefix='ahc-numeric-') as work:
        with zipfile.ZipFile(path) as archive:
            archive.extract('compact.sqlite', work)
        with closing(sqlite3.connect(Path(work)/'compact.sqlite')) as db:
            db.row_factory = sqlite3.Row
            cursor = db.execute('SELECT * FROM runs ORDER BY id')
            while rows := cursor.fetchmany(512):
                u = np.array([np.frombuffer(r['endpoint'], '<c16').reshape(4,4) for r in rows])
                starts = np.array([np.frombuffer(r['start'], '<c16').reshape(4,4) for r in rows])
                require(np.isfinite(u).all() and np.isfinite(starts).all(), 'Nonfinite matrix')
                mod = abs(u)
                s = np.divide(u, mod, out=np.zeros_like(u), where=mod>0)
                x = adj(s) @ u
                residual = (x-adj(x))/(2j)
                unity = np.linalg.norm(adj(u) @ u-np.eye(4), axis=(1,2))
                actual = np.column_stack((mod.sum(axis=(1,2)),
                    np.linalg.norm(residual, axis=(1,2)), mod.min(axis=(1,2)),
                    abs(mod-.5).max(axis=(1,2)), unity))
                expected = np.array([[r[n] for n in names] for r in rows])
                require(np.isfinite(expected).all(), 'Missing or nonfinite recorded diagnostic')
                differences = abs(actual-expected).max(axis=0)
                for n, error in zip(names, differences):
                    maxima[n] = max(maxima[n], float(error))
                require(np.allclose(actual, expected, atol=1e-12, rtol=1e-12),
                        'Scalar diagnostic mismatch')
                max_end_unity = max(max_end_unity, float(unity.max()))
                max_start_unity = max(max_start_unity, float(np.linalg.norm(
                    adj(starts) @ starts-np.eye(4), axis=(1,2)).max()))
                counts.update(r['status'] for r in rows)
                selected = [i for i,r in enumerate(rows) if r['hessian_full'] is not None]
                if selected:
                    computed = spectra(u[selected])
                    for name, values in zip(('full','horizontal'), computed):
                        stored = np.array([np.frombuffer(rows[i]['hessian_'+name], '<f8')
                                           for i in selected])
                        error = abs(values-stored).max(axis=1)
                        scale = np.maximum(1.0, abs(stored).max(axis=1))
                        spectral_abs[name] = max(spectral_abs[name], float(error.max()))
                        spectral_scaled[name] = max(spectral_scaled[name], float((error/scale).max()))
                        require(np.all(error/scale < 2e-11), 'Spectrum mismatch: '+name)
                    top = np.array([rows[i]['lambda_max_horizontal'] for i in selected])
                    require(np.allclose(computed[1][:,-1], top, atol=1e-11, rtol=2e-11),
                            'Top horizontal curvature mismatch')
                    nspectra += len(selected)
            require(sum(counts.values()) == report['retained_runs'], 'Wrong row count')
            require(max(max_start_unity,max_end_unity)<1e-10, 'Unitarity check failed')
    return {
        'archive': path.name,
        'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_runs': report['source_runs'], 'retained_runs': sum(counts.values()),
        'retained_status_counts': dict(counts), 'spectra_recomputed': nspectra,
        'max_scalar_absolute_difference': maxima,
        'max_spectrum_absolute_difference': spectral_abs,
        'max_spectrum_difference_scaled_by_max_1_spectral_radius': spectral_scaled,
        'max_start_unitarity_error': max_start_unity,
        'max_endpoint_unitarity_error': max_end_unity,
        'result': 'passed',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent/'overnight')
    args = parser.parse_args()
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'numpy': np.__version__, 'scope': 'All retained starts and endpoints; all stored spectra. '
              'Floating-point checks only; discarded endpoints are not available.', 'archives': []}
    for mode in ('ascent','stationary'):
        item = audit(args.directory/(mode+'-compact.zip'))
        result['archives'].append(item)
        print(json.dumps(item, indent=2), flush=True)
    output = args.directory/'archive-checks.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved', output)


if __name__ == '__main__':
    main()

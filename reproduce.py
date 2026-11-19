#!/usr/bin/env python3
"""Recheck the paper's saved results without repeating its large searches.

Run: python3 reproduce.py
Requires NumPy, SciPy, mpmath and SymPy. Executes in a temporary copy so that
stored evidence is never overwritten. The only persistent output is the
requested report (default data/reproduction.json). --list prints the steps.
The audit covers retained endpoints and aggregate integrity, not discarded runs
or a covering of any unitary group. Exact, interval and numerical checks are
identified separately in the report. Python assertions must be enabled.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
STEPS = [
    ('phase_grid', 'exact enumeration', ['phase_grid.py']),
    ('models', 'numerical comparison to exact formulas', ['models.py']),
    ('three_lines', 'interval and numerical checks', ['three_lines.py']),
    ('four_lines', 'exact polynomial identities', ['four_lines.py']),
    ('four_moduli', 'exact enumeration and elimination', ['four_moduli.py']),
    ('curvature', 'interval certificate and numerical comparison', ['curvature.py']),
    ('flat_limit', 'exact polynomial identities', ['flat_limit.py']),
    ('moduli_endpoints', 'saved numerical diagnostics', ['moduli_search.py', '--verify-saved']),
    ('initial_survey', 'interval exclusions and numerical controls', ['audit.py']),
    ('higher_calibrations', 'interval exclusions and numerical refinements', ['check.py']),
    ('weak_five', 'interval exclusion', ['weak5.py']),
    ('ascent_four_selftest', 'numerical derivative checks', ['ahc_ascent.py', '--self-test']),
    ('stationary_four_selftest', 'numerical derivative checks', ['ahc_stationary.py', '--self-test']),
    ('five_tests', 'numerical and runtime tests', ['test_ahc5.py', '--runtime']),
    ('overnight_four', 'archive integrity and retained numerical outputs', ['audit_compact.py']),
    ('pilot_ascent', 'archive integrity and retained numerical outputs',
     ['audit_ahc5.py', 'overnight5/ascent-pilot.zip']),
    ('pilot_stationary', 'archive integrity and retained numerical outputs',
     ['audit_ahc5.py', 'overnight5/stationary-pilot.zip']),
    ('mac_ascent_five', 'archive integrity and retained numerical outputs',
     ['audit_ahc5.py', 'overnight5/ascent-mac.zip']),
    ('mac_stationary_five', 'archive integrity and retained numerical outputs',
     ['audit_ahc5.py', 'overnight5/stationary-mac.zip']),
    ('stalled_five', 'retained-point refinement and interval existence certificate',
     ['stalled5.py']),
    ('catalogue_five', 'numerical equivalences and 452 interval exclusions', ['catalogue5.py']),
    ('critical_five', '40 interval critical-root, local uniqueness and inertia certificates', ['critical5.py']),
    ('daytime_tests', 'replay regression and standalone runtime tests', ['test_day5.py']),
    ('daytime_replay_local', 'retained-data audit of the short local test',
     ['audit_day5.py', 'daytime/replay-local.zip']),
    ('daytime_search_local', 'retained-data audit of the short local test',
     ['audit_day5.py', 'daytime/search-local.zip']),
    ('daytime_replay_mac', 'retained-data audit of the completed partial replay',
     ['audit_day5.py', 'daytime/replay-mac.zip']),
    ('daytime_search_mac', 'retained-data and eigenspace audit of the completed targeted search',
     ['audit_day5.py', 'daytime/search-mac.zip']),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--report', type=Path, default=ROOT/'data/reproduction.json')
    parser.add_argument('--timeout', type=float, default=1800,
                        help='maximum seconds for any one check (default 1800)')
    args = parser.parse_args()
    if args.list:
        for name, kind, command in STEPS:
            print(f'{name}: {kind}: python3 {" ".join(command)}')
        return 0
    if not __debug__ or sys.flags.optimize:
        raise SystemExit('Run Python without -O: these checks require assertions.')
    versions = {name: importlib.metadata.version(name)
                for name in ['numpy', 'scipy', 'mpmath', 'sympy']}
    start = time.monotonic()
    # Include all retained evidence and executable sources; omit this run's report.
    paths = [p for p in ROOT.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.suffix != '.pyc'
             and p.resolve() != args.report.resolve()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    report = dict(python=platform.python_version(), platform=platform.platform(),
                  packages=versions, source_and_evidence_sha256=hashes, steps=[],
                  scope='Saved evidence and exact/interval checks; no large search is repeated.')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONOPTIMIZE', None)
    with tempfile.TemporaryDirectory(prefix='ahc-reproduce-') as tmp:
        work = Path(tmp)/'code'
        shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for number, (name, kind, command) in enumerate(STEPS, 1):
            print(f'[{number}/{len(STEPS)}] {name}: {kind}', flush=True)
            before = time.monotonic()
            item = dict(name=name, kind=kind, command=['python3']+command)
            try:
                result = subprocess.run([sys.executable]+command, cwd=work, env=env,
                                        text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
                item.update(seconds=round(time.monotonic()-before, 3),
                            returncode=result.returncode, output=result.stdout,
                            status='passed' if result.returncode == 0 else 'failed')
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ''
                if isinstance(output, bytes): output = output.decode(errors='replace')
                item.update(seconds=round(time.monotonic()-before, 3),
                            status='timeout', output=output)
            report['steps'].append(item)
            report['seconds'] = round(time.monotonic()-start, 3)
            report['result'] = 'running' if item['status'] == 'passed' else item['status']
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2)+'\n')
            print(f"  {item['status']} in {item['seconds']:.1f} s", flush=True)
            if item['status'] != 'passed':
                print(item['output'][-6000:], flush=True)
                return 1
    report['result'] = 'passed'
    report['seconds'] = round(time.monotonic()-start, 3)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f"All {len(STEPS)} checks passed in {report['seconds']:.1f} s. Report: {args.report}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

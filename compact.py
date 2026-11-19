#!/usr/bin/env python3
"""Extract and verify a compact archive of a completed AHC overnight search.

From either folder containing results.sqlite, run
    python3 /path/to/project/code011/compact.py
The output is ascent-compact.zip or stationary-compact.zip beside the database.
The source is opened read-only. Each retained database row is copied exactly.
The archive contains all unresolved and exceptional runs, numerical extrema and
counts for every group, all saved examples, and a reproducible sample of 2048
ordinary runs. It also includes the original search script and software versions.

To recheck a ZIP later
    python3 compact.py --verify ascent-compact.zip
Only Python's standard library is required. No matrix computations are rerun.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import random
import shutil
import sqlite3
import struct
import tempfile
import time
import zipfile

FIELDS = ('steering_steps', 'solver_steps', 'L', 'gradient', 'min_modulus',
          'flat_error', 'unitarity_error', 'lambda_max_horizontal')
COLUMNS = ('id', 'family', 'status', 'level', 'steering_steps', 'solver_steps',
           'L', 'gradient', 'min_modulus', 'flat_error', 'unitarity_error',
           'lambda_max_horizontal', 'start', 'endpoint', 'hessian_full',
           'hessian_horizontal', 'error')
RUN_SCHEMA = '''CREATE TABLE runs (
 id INTEGER PRIMARY KEY, family TEXT, status TEXT, level TEXT,
 steering_steps INTEGER, solver_steps INTEGER, L REAL, gradient REAL,
 min_modulus REAL, flat_error REAL, unitarity_error REAL,
 lambda_max_horizontal REAL, start BLOB, endpoint BLOB,
 hessian_full BLOB, hessian_horizontal BLOB, error TEXT)'''
INSERT_RUN = 'INSERT INTO runs VALUES (' + ','.join(['?'] * len(COLUMNS)) + ')'
KEEP = "status NOT IN ('near_flat','near_zero','nonflat_uphill') OR level='unlisted'"
CHUNK = 100000
SAMPLE_SIZE = 2048


def require(condition, message):
    if not condition:
        raise ValueError(message)


def open_readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-131072')
    db.execute('PRAGMA temp_store=MEMORY')
    return db


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_row(digest, row):
    # Length and type delimiters preserve exact SQLite values, including BLOBs.
    for value in row:
        if value is None:
            digest.update(b'N')
        elif isinstance(value, bytes):
            digest.update(b'B' + struct.pack('<Q', len(value)) + value)
        elif isinstance(value, str):
            data = value.encode('utf-8')
            digest.update(b'S' + struct.pack('<Q', len(data)) + data)
        elif isinstance(value, int):
            digest.update(b'I' + struct.pack('<q', value))
        else:
            digest.update(b'F' + struct.pack('<d', value))


def group_query():
    terms = ['family', 'status', 'level', 'COUNT(*)', 'MIN(id)', 'MAX(id)']
    for name in FIELDS:
        terms.extend(['COUNT(' + name + ')', 'MIN(' + name + ')', 'MAX(' + name + ')'])
    return ('SELECT ' + ','.join(terms) + ' FROM runs WHERE id>=? AND id<? '
            'GROUP BY family,status,level')


def combine_group(groups, row):
    key = row[:3]
    if key not in groups:
        groups[key] = {'family': row[0], 'status': row[1], 'level': row[2],
                       'count': 0, 'first_id': row[4], 'last_id': row[5], 'metrics': {}}
    group = groups[key]
    group['count'] += row[3]
    group['first_id'] = min(group['first_id'], row[4])
    group['last_id'] = max(group['last_id'], row[5])
    for j, name in enumerate(FIELDS):
        count, low, high = row[6+3*j:9+3*j]
        metric = group['metrics'].setdefault(name, {'count': 0, 'min': None, 'max': None})
        metric['count'] += count
        if count:
            metric['min'] = low if metric['min'] is None else min(low, metric['min'])
            metric['max'] = high if metric['max'] is None else max(high, metric['max'])


def copy_small_tables(source, destination):
    schemas = {
        'meta': 'key TEXT PRIMARY KEY, value TEXT',
        'sessions': 'id INTEGER PRIMARY KEY, utc TEXT, python TEXT, numpy TEXT, platform TEXT, workers INTEGER',
        'tallies': 'family TEXT, status TEXT, level TEXT, n INTEGER, PRIMARY KEY(family,status,level)',
        'examples': 'status TEXT, level TEXT, score REAL, run_id INTEGER, PRIMARY KEY(status,level)'}
    for name, schema in schemas.items():
        rows = source.execute('SELECT * FROM ' + name).fetchall()
        destination.execute('CREATE TABLE ' + name + ' (' + schema + ')')
        if rows:
            sql = 'INSERT INTO ' + name + ' VALUES (' + ','.join('?' for _ in rows[0]) + ')'
            destination.executemany(sql, rows)


def locate_script(source, config, supplied):
    name = 'ahc_' + config['mode'] + '.py'
    choices = [supplied] if supplied else [Path(__file__).resolve().parent/name,
                                         source.parent/name, source.parent.parent/name]
    for choice in choices:
        if choice and choice.is_file() and file_hash(choice) == config['sha256']:
            return choice
    raise ValueError('Matching original script needed. Use --script /path/to/' + name)


def scan(source, destination, maximum, groups, row_digest, expected_count):
    total = retained = 0
    began = last_print = time.monotonic()
    query = group_query()
    for lo in range(0, maximum+1, CHUNK):
        hi = lo + CHUNK
        for row in source.execute(query, (lo, hi)):
            combine_group(groups, row)
            total += row[3]
        rows = source.execute('SELECT * FROM runs WHERE id>=? AND id<? AND (' + KEEP + ') '
                              'ORDER BY id', (lo, hi)).fetchall()
        destination.executemany(INSERT_RUN, rows)
        for row in rows:
            hash_row(row_digest, row)
        retained += len(rows)
        destination.commit()
        if time.monotonic()-last_print >= 15:
            print(f'{total:,}/{expected_count:,} starts scanned; {retained:,} complete runs kept; '
                  f'{(time.monotonic()-began)/60:.1f} minutes', flush=True)
            last_print = time.monotonic()
    return total, retained


def write_report(folder, report, groups):
    (folder/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (folder/'groups.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['family','status','level','count','first_id','last_id'] +
                        [field+'_'+tail for field in FIELDS for tail in ('count','min','max')])
        for key in sorted(groups):
            group = groups[key]
            writer.writerow([group[x] for x in ('family','status','level','count','first_id','last_id')] +
                            [group['metrics'][field][tail] for field in FIELDS
                             for tail in ('count','min','max')])
    lines = ['AHC order-four compact archive',
             f"Mode: {report['mode']}; starts scanned: {report['source_runs']:,}",
             f"Complete runs retained: {report['retained_runs']:,}",
             'All unresolved and exceptional runs are retained.',
             'Ordinary runs are represented by a sample, first occurrences and saved examples.',
             'groups.csv gives counts, non-null counts and extrema over the full campaign.',
             'compact.sqlite contains retained runs and the original meta/sessions/tallies/examples.',
             'Run IDs and all BLOBs are copied byte for byte.',
             'The original search script SHA-256 matches the database configuration.',
             '', 'Numerical totals']
    for status, count in sorted(report['totals'].items()):
        lines.append(f'{status}: {count:,}')
    lines.extend(['', 'Unresolved endpoints, full campaign',
                  json.dumps(report['unresolved'], indent=2), '',
                  'Verification checks archive contents and their agreement with the copied records.',
                  'Numerical stationarity and Hessian signs retain their original floating-point meaning.',
                  'Complex BLOBs use little-endian complex128, C-order, shape 4 by 4.',
                  'Hessian BLOBs use little-endian float64, ascending spectra, lengths 16 and 9.',
                  'The selected rows do not reconstruct the discarded floating-point endpoints.',
                  'The script, seed, run ID and version metadata specify how to regenerate their initial conditions.',
                  'Different numerical libraries can change rounding in initial matrices and subsequent solver paths.'])
    (folder/'README.txt').write_text('\n'.join(lines)+'\n')


def verify(folder):
    report = json.loads((folder/'report.json').read_text())
    with closing(open_readonly(folder/'compact.sqlite')) as db:
        require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Compact SQLite integrity failed')
        require(db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == report['retained_runs'],
                'Retained row count differs')
        config = json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
        script = folder/('ahc_' + config['mode'] + '.py')
        require(file_hash(script) == config['sha256'], 'Original script hash differs')
        for clause, key in [(KEEP, 'exception_rows_sha256'), ('NOT (' + KEEP + ')', 'ordinary_rows_sha256')]:
            digest = hashlib.sha256()
            for row in db.execute('SELECT * FROM runs WHERE ' + clause + ' ORDER BY id'):
                hash_row(digest, row)
            require(digest.hexdigest() == report[key], 'Copied database rows differ')
        with (folder/'groups.csv').open(newline='') as stream:
            groups = list(csv.DictReader(stream))
        got = {(r['family'], r['status'], r['level']): int(r['count']) for r in groups}
        tallies = {r[:3]: r[3] for r in db.execute('SELECT * FROM tallies')}
        require(got == tallies and sum(got.values()) == report['source_runs'], 'Campaign totals differ')
        totals = Counter()
        for (_,status,_), count in got.items():
            totals[status] += count
        require(dict(totals) == report['totals'], 'Report totals differ')
        for group in groups:
            if group['status'] not in ('near_flat','near_zero','nonflat_uphill') or group['level']=='unlisted':
                count = db.execute('SELECT COUNT(*) FROM runs WHERE family=? AND status=? AND level=?',
                                   (group['family'],group['status'],group['level'])).fetchone()[0]
                require(count == int(group['count']), 'Missing unresolved or exceptional run')
        require(db.execute('SELECT COUNT(*) FROM examples e LEFT JOIN runs r ON r.id=e.run_id '
                           'WHERE r.id IS NULL').fetchone()[0] == 0, 'Missing example')
        maxima = db.execute("SELECT COUNT(*),MAX(flat_error),MAX(min_modulus) FROM runs WHERE status='unresolved'").fetchone()
        require(list(maxima) == list(report['unresolved'].values()), 'Unresolved extrema differ')
    return report


def verify_zip(path):
    with tempfile.TemporaryDirectory(prefix='ahc-verify-') as work:
        folder = Path(work)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                require('/' not in member.filename and '\\' not in member.filename
                        and member.filename not in ('.','..'), 'Unexpected archive path')
            archive.extractall(folder)
        manifest = json.loads((folder/'checksums.json').read_text())
        require(set(manifest) == {p.name for p in folder.iterdir() if p.name!='checksums.json'},
                'Archive member list differs')
        for name, expected in manifest.items():
            require(file_hash(folder/name) == expected, 'File hash differs: ' + name)
        report = verify(folder)
    print(f"Verified {report['mode']} archive: {report['source_runs']:,} starts, "
          f"{report['retained_runs']:,} complete runs retained.", flush=True)
    return report


def export(source_path, output, supplied_script):
    with ExitStack() as stack:
        lock_path = source_path.parent/'run.lock'
        if lock_path.exists():
            lock = stack.enter_context(lock_path.open('rb'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Search is still running. Stop it before extraction.') from None
        source = stack.enter_context(closing(open_readonly(source_path)))
        source.execute('BEGIN')
        require(tuple(r[1] for r in source.execute('PRAGMA table_info(runs)')) == COLUMNS,
                'Unexpected source database schema')
        config_row = source.execute("SELECT value FROM meta WHERE key='config'").fetchone()
        require(config_row is not None, 'Missing campaign configuration')
        config = json.loads(config_row[0])
        require(config['mode'] in ('ascent','stationary'), 'Unknown search mode')
        script = locate_script(source_path, config, supplied_script)
        output = output or source_path.parent/(config['mode']+'-compact.zip')
        output = output.resolve()
        require(not output.exists(), 'Output already exists. Use --verify to check it, or choose --out.')
        require(output.parent.is_dir(), 'Output directory does not exist')
        before = source_path.stat()
        tallies = {r[:3]:r[3] for r in source.execute('SELECT * FROM tallies')}
        expected = sum(tallies.values())
        first = source.execute('SELECT id FROM runs ORDER BY id LIMIT 1').fetchone()
        last = source.execute('SELECT id FROM runs ORDER BY id DESC LIMIT 1').fetchone()
        bounds = (first[0],last[0]) if first and last else (None,None)
        require(expected > 0 and bounds[0] is not None and bounds[0] >= 0, 'Source has no completed runs')
        work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.ahc-compact-',dir=output.parent)))
        destination = stack.enter_context(closing(sqlite3.connect(work/'compact.sqlite')))
        destination.execute(RUN_SCHEMA)
        copy_small_tables(source, destination)
        destination.commit()
        groups = {}
        exception_digest = hashlib.sha256()
        print(f"Reading {config['mode']}: {expected:,} starts, {before.st_size/1e9:.1f} GB. "
              'Progress follows every 15 seconds.', flush=True)
        count, retained = scan(source, destination, bounds[1], groups, exception_digest, expected)
        require({k:g['count'] for k,g in groups.items()} == tallies,
                'Recomputed counts disagree with stored tallies')
        ids = {g['first_id'] for g in groups.values()}
        ids.update(r[0] for r in source.execute('SELECT run_id FROM examples'))
        # Random integers in the run-ID range give an independently reproducible sample.
        sampler = random.Random(421703)
        sample_ids = sampler.sample(range(bounds[1]+1), min(SAMPLE_SIZE,bounds[1]+1))
        ids.update(sample_ids)
        ordinary_digest = hashlib.sha256()
        for idx in sorted(ids):
            if destination.execute('SELECT 1 FROM runs WHERE id=?',(idx,)).fetchone():
                continue
            row = source.execute('SELECT * FROM runs WHERE id=?',(idx,)).fetchone()
            if row is None:  # an interrupted campaign may have gaps
                continue
            destination.execute(INSERT_RUN,row)
            hash_row(ordinary_digest,row)
            retained += 1
        destination.commit()
        totals = Counter()
        for group in groups.values():
            totals[group['status']] += group['count']
        unresolved = [g for g in groups.values() if g['status']=='unresolved']
        def worst(field):
            values=[g['metrics'][field]['max'] for g in unresolved if g['metrics'][field]['count']]
            return max(values,default=None)
        report = {'format_version':1, 'created_utc':datetime.now(timezone.utc).isoformat(),
                  'mode':config['mode'], 'source_filename':source_path.name,
                  'source_size_bytes':before.st_size, 'source_runs':count, 'retained_runs':retained,
                  'first_run_id':bounds[0], 'last_run_id':bounds[1],
                  'contiguous_ids':bounds[0]==0 and count==bounds[1]+1,
                  'batch_count':source.execute('SELECT COUNT(*) FROM batches').fetchone()[0],
                  'config':config, 'totals':dict(totals),
                  'unresolved':{'count':totals['unresolved'], 'worst_flat_error':worst('flat_error'),
                                'largest_min_entry':worst('min_modulus')},
                  'exception_rows_sha256':exception_digest.hexdigest(),
                  'ordinary_rows_sha256':ordinary_digest.hexdigest(),
                  'sample_seed':421703, 'sample_requested':SAMPLE_SIZE,
                  'selection':'all unresolved/exceptional/unlisted runs; saved examples; first per group; sample'}
        write_report(work, report, groups)
        summary = source_path.parent/'summary.txt'
        if summary.is_file():
            shutil.copyfile(summary,work/'summary.txt')
        shutil.copyfile(script,work/('ahc_'+config['mode']+'.py'))
        shutil.copyfile(Path(__file__),work/'compact.py')
        destination.close()
        after = source_path.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
                'Source changed during extraction')
        verify(work)
        checksums = {p.name:file_hash(p) for p in work.iterdir() if p.is_file()}
        (work/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
        temporary_zip = work/'archive.zip'
        with zipfile.ZipFile(temporary_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(checksums):
                archive.write(work/name,name)
            archive.write(work/'checksums.json','checksums.json')
        verify_zip(temporary_zip)
        require(not output.exists(), 'Output appeared while extraction was running')
        temporary_zip.replace(output)
    print(f'Saved {output} ({output.stat().st_size/1e6:.2f} MB).', flush=True)
    print('Send this compact ZIP for review. Keep the source database until that review.', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('database',nargs='?',type=Path,default=Path('results.sqlite'))
    parser.add_argument('--out',type=Path,help='output ZIP path')
    parser.add_argument('--script',type=Path,help='original search script if stored elsewhere')
    parser.add_argument('--verify',type=Path,metavar='ZIP',help='verify an existing compact ZIP')
    args=parser.parse_args()
    try:
        if args.verify:
            verify_zip(args.verify)
        else:
            require(args.database.is_file(),'results.sqlite not found. Run from its folder or pass its path.')
            export(args.database.resolve(),args.out,args.script)
    except (ValueError,OSError,sqlite3.Error,zipfile.BadZipFile) as exc:
        parser.exit(1,str(exc)+'\n')


if __name__=='__main__':
    main()

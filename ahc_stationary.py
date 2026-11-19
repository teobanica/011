#!/usr/bin/env python3
"""Stationary points of the entrywise 1-norm on U_4, with four nonflat steering bands.

Run on macOS with NumPy: caffeinate -i python3 ahc_stationary.py
Three workers, seven cumulative active hours; rerun from the same directory to resume.
Read ahc-stationary-out/summary.txt. Each committed start and endpoint, all diagnostics
and the two spectra at regular endpoints are in results.sqlite, table runs.
Complex blobs are 4x4 C-order little-endian complex128; spectra are ascending
little-endian float64 of lengths 16 and 9. To inspect an endpoint in Python:
    blob = db.execute("select endpoint from runs where id=?", (run_id,)).fetchone()[0]
    u = np.frombuffer(blob, dtype="<c16").reshape(4, 4)
Matrices with min modulus <= 1e-8 are flagged near_zero; the Hessian is skipped.
A candidate is a numerical flag only; near-flat means modulus error <= 1e-6.
For a fresh independent campaign use a new --out and --seed.
"""
MODE = "stationary"
DEFAULT_SEED = 421702
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import sqlite3
import sys
import time

# Several processes run small 4-by-4 problems; nested BLAS threads cost time.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

N = 4
EYE = np.eye(N)
BASIS = [np.diag(e).astype(complex) for e in EYE]
for j in range(N):
    for k in range(j + 1, N):
        e = np.outer(EYE[j], EYE[k])
        BASIS.extend([(e + e.T) / np.sqrt(2), 1j * (e - e.T) / np.sqrt(2)])
BASIS = np.array(BASIS)
OFF = BASIS[N:]
PRODUCTS = (BASIS[:, None] @ BASIS).reshape(256, N, N)
FLAT_TOL, GRAD_TOL, ENTRY_TOL, CURV_TOL = 1e-6, 1e-12, 1e-8, 1e-7
LEVELS = (7.391036, 7.633239, 7.633377, 7.637615)
AX = (-2, -1)


def adj(a):
    return a.conj().swapaxes(-2, -1)


def haar(rng, size):
    z = rng.normal(size=(size, N, N)) + 1j * rng.normal(size=(size, N, N))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=-2, axis2=-1)
    return q * (d / abs(d))[:, None, :]


def components(u):
    s = u / np.maximum(abs(u), 1e-300)
    x = adj(s) @ u
    return s, x, (x - adj(x)) / (2j)


def coordinates(a, basis):
    return np.einsum("...ij,kji->...k", a, basis).real


def horizontal(u):
    # Remove four right phases, then three independent left phases.
    left = adj(u)[:, None] @ BASIS[:3] @ u[:, None]
    columns = coordinates(left, OFF).swapaxes(1, 2)
    return np.linalg.qr(columns, mode="complete")[0][:, :, 3:]


def expstep(u, b):
    eig, v = np.linalg.eigh(b)
    return u @ ((v * np.exp(1j * eig)[:, None, :]) @ adj(v))


def hessian(u):
    """Geodesic second derivatives for U exp(i t B), tr(B^2)=1."""
    s, x, _ = components(u)
    w = ((u[:, None] @ BASIS) * s[:, None].conj()).real
    w = (w / np.sqrt(abs(u))[:, None]).reshape(-1, 16, 16)
    y = (x + adj(x)) / 2
    h = w @ w.swapaxes(1, 2) - coordinates(y, PRODUCTS).reshape(-1, 16, 16)
    h = (h + h.swapaxes(1, 2)) / 2
    c = horizontal(u)
    reduced = c.swapaxes(1, 2) @ h[:, 4:, 4:] @ c
    return np.linalg.eigvalsh(h), np.linalg.eigvalsh(reduced)


def metrics(u):
    return (float(abs(u).sum()), float(np.linalg.norm(components(u)[2])),
            float(abs(u).min()), float(abs(abs(u) - .5).max()),
            float(np.linalg.norm(adj(u) @ u - EYE)))


def calibration():
    x = .609609789010948
    b = np.sqrt((1 - x*x) / 3)
    u = np.full((4, 4), b, complex)
    u[0, 0] = x
    row = (-x + 2j * np.sin(2 * np.pi * np.arange(3) / 3)) / 3
    for j in range(3):
        u[j+1, 1:] = np.roll(row, j)
    return u


def self_check():
    u = calibration()
    lval, g, _, _, unity = metrics(u)
    full, reduced = hessian(u[None])
    if (abs(lval - 7.63761518684038) > 1e-12 or max(g, unity) > 1e-13
            or abs(reduced[0, -1] - 2.876887) > 1e-5
            or sum(full[0] > CURV_TOL) != 1 or sum(abs(full[0]) < CURV_TOL) != 7):
        raise RuntimeError("Order-four saddle calibration failed")
    f = np.exp(2j * np.pi * np.outer(np.arange(4), np.arange(4)) / 4) / 2
    if hessian(f[None])[0].max() > 1e-12:
        raise RuntimeError("Fourier Hessian calibration failed")
    rng = np.random.default_rng(19)
    v = haar(rng, 1)
    b = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    b = (b + adj(b)) / 2
    b /= np.linalg.norm(b)
    step = .0001
    plus, minus = [abs(expstep(v, sign * step * b[None])).sum() for sign in (1, -1)]
    phase, x, residual = components(v[0])
    grad = -np.trace(residual @ b).real
    q = (((v[0] @ b * phase.conj()).real ** 2) / abs(v[0])).sum()
    q -= np.trace(x @ b @ b).real
    if (abs((plus-minus)/(2*step) - grad) > 1e-6
            or abs((plus+minus-2*abs(v).sum())/step**2 - q) > 2e-5):
        raise RuntimeError("Geodesic finite difference check failed")


# The steering residual is (R, 3 max(0,L-cap)); the final solve always uses R alone.
BATCH = 16
POLICIES = (("haar", None), ("phase_polar", None), ("near_real", None),
            ("near_flat", None), ("cap_7.40", 7.40), ("cap_7.65", 7.65),
            ("cap_7.90", 7.90), ("cap_7.995", 7.995))
STEER_STEPS, ROOT_STEPS = 24, 120


def starts_for(rng, size, batch_id):
    name, cap = POLICIES[batch_id % len(POLICIES)]
    if name == "phase_polar":
        z = np.exp(2j*np.pi*rng.random((size,N,N)))
        left, _, right = np.linalg.svd(z)
        u = left @ right
    elif name in ("near_real", "near_flat"):
        if name == "near_real":
            q, r = np.linalg.qr(rng.normal(size=(size, N, N)))
            u = q*np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
        else:
            z = np.exp(2j*np.pi*rng.random(size))
            u = np.array([[[1,1,1,1], [1,t,-1,-t], [1,-1,1,-1], [1,-t,-1,t]]
                          for t in z]) / 2
        weights = rng.normal(size=(size, 16))
        size_factor = 10**rng.uniform(-2, -.05, size)/np.linalg.norm(weights, axis=1)
        u = expstep(u, np.einsum("bk,kij->bij", weights*size_factor[:, None], BASIS))
    else:
        u = haar(rng, size)
    return name, u, cap


def refine(starts, cap, limit):
    u = starts.copy()
    steps = np.zeros(len(u), dtype=int)
    damping = np.full(len(u), .01)
    for _ in range(limit):
        s, x, r = components(u)
        score = np.sum(abs(r)**2, axis=AX)
        if cap is not None:
            penalty = 3*np.maximum(0, abs(u).sum(axis=AX)-cap)
            score += penalty**2
        idx = np.flatnonzero((score > GRAD_TOL**2) & (abs(u).min(axis=AX) > 1e-13)
                             & (damping < 1e7))
        if not len(idx):
            break
        v, s, x = u[idx], s[idx], x[idx]
        c = horizontal(v)
        vb = v[:, None] @ OFF
        radial = (vb*s[:, None].conj()).real / abs(v[:, None])
        dx = 1j*(x[:, None] @ OFF - adj(s[:, None]*radial) @ v[:, None])
        dr = (dx-adj(dx))/(2j)
        jac = coordinates(dr, BASIS).swapaxes(1, 2) @ c
        rhs = coordinates(r[idx], BASIS)
        if cap is not None:
            dl = -(rhs[:, None, 4:] @ c)
            jac = np.concatenate((jac, 3*dl*(penalty[idx] > 0)[:, None, None]), axis=1)
            rhs = np.column_stack((rhs, penalty[idx]))
        left, sv, right = np.linalg.svd(jac, full_matrices=False)
        factor = sv/(sv**2 + damping[idx, None]**2)
        step = -(right.swapaxes(1, 2) @ ((left.swapaxes(1, 2) @ rhs[..., None])
                                         * factor[..., None]))
        coord = (c @ step)[..., 0]
        coord /= np.maximum(1, np.linalg.norm(coord, axis=1, keepdims=True) / .65)
        proposal = expstep(v, np.einsum("ba,aij->bij", coord, OFF))
        new = np.sum(abs(components(proposal)[2])**2, axis=AX)
        if cap is not None:
            new += (3*np.maximum(0, abs(proposal).sum(axis=AX)-cap))**2
        good = new < score[idx]
        u[idx[good]] = proposal[good]
        damping[idx] = np.where(good, np.maximum(1e-13, damping[idx]*.45), damping[idx]*4)
        steps[idx] += 1
    return u, steps


def solve(starts, cap):
    if cap is None:
        mid, warm = starts, np.zeros(len(starts), dtype=int)
    else:
        mid, warm = refine(starts, cap, STEER_STEPS)
    u, steps = refine(mid, None, ROOT_STEPS)
    return u, warm, steps


def pack(a, dtype):
    return np.asarray(a, dtype=dtype).tobytes(order="C")


def classify(lval, grad, minimum, spread, unity, top):
    if unity > 1e-10:
        return "unitarity_failure"
    if minimum <= ENTRY_TOL:
        return "near_zero"
    if spread <= FLAT_TOL:
        return "near_flat"
    if grad > GRAD_TOL:
        return "unresolved"
    if top <= CURV_TOL:
        return "candidate"
    return "nonflat_uphill"


def run_batch(seed, batch_id):
    rng = np.random.default_rng(np.random.SeedSequence([seed, batch_id]))
    family, starts, policy = starts_for(rng, BATCH, batch_id)
    endpoints = []
    def process(block, offset):
        # A failed batched factorization must not discard its healthy neighbors.
        try:
            u, warm, steps = solve(block, policy)
            stats = [metrics(v) for v in u]
            regular = [k for k, m in enumerate(stats) if m[2] > ENTRY_TOL and m[4] <= 1e-10]
            spectra = {}
            if regular:
                full, reduced = hessian(u[regular])
                spectra = dict(zip(regular, zip(full, reduced)))
            rows = []
            for k, (v, m) in enumerate(zip(u, stats)):
                if not np.isfinite(m).all():
                    raise FloatingPointError("nonfinite endpoint")
                top = fblob = rblob = None
                if k in spectra:
                    fs, rs = spectra[k]
                    if not np.isfinite(fs).all() or not np.isfinite(rs).all():
                        raise FloatingPointError("nonfinite spectrum")
                    top = float(rs[-1])
                    fblob, rblob = pack(fs, "<f8"), pack(rs, "<f8")
                status = classify(*m, top)
                level = ""
                if status in ("candidate", "nonflat_uphill"):
                    near = min(LEVELS, key=lambda lev: abs(lev-m[0]))
                    level = ("%.6f" % near) if abs(near-m[0]) < 2e-6 else "unlisted"
                rows.append((batch_id*BATCH + offset+k, family, status, level, int(warm[k]),
                             int(steps[k]), *m, top, pack(block[k], "<c16"), pack(v, "<c16"),
                             fblob, rblob, None))
            endpoints.extend(rows)
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            if len(block) > 1:
                for j in range(len(block)):
                    process(block[j:j+1], offset+j)
            else:
                # The exact starting matrix is retained even when LAPACK returns no endpoint.
                endpoints.append((batch_id*BATCH+offset, family, "numeric_error", "", None,
                                  None, None, None, None, None, None, None,
                                  pack(block[0], "<c16"), None, None, None, str(exc)))
    process(starts, 0)
    return endpoints


DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions
  (id INTEGER PRIMARY KEY, utc TEXT, python TEXT, numpy TEXT, platform TEXT, workers INTEGER);
CREATE TABLE IF NOT EXISTS batches (id INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS runs
  (id INTEGER PRIMARY KEY, family TEXT, status TEXT, level TEXT, steering_steps INTEGER,
   solver_steps INTEGER, L REAL, gradient REAL, min_modulus REAL, flat_error REAL,
   unitarity_error REAL, lambda_max_horizontal REAL, start BLOB, endpoint BLOB,
   hessian_full BLOB, hessian_horizontal BLOB, error TEXT);
CREATE TABLE IF NOT EXISTS examples
  (status TEXT, level TEXT, score REAL, run_id INTEGER, PRIMARY KEY(status,level));
CREATE TABLE IF NOT EXISTS tallies
  (family TEXT, status TEXT, level TEXT, n INTEGER, PRIMARY KEY(family,status,level));
"""


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return default if row is None else json.loads(row[0])


def commit_batch(db, batch_id, rows, elapsed):
    counts = Counter((row[1], row[2], row[3]) for row in rows)
    if len(rows) != BATCH or len({r[0] for r in rows}) != BATCH:
        raise RuntimeError("Incomplete batch")
    with db:
        db.execute("INSERT INTO batches VALUES (?)", (batch_id,))
        db.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        db.executemany("""INSERT INTO tallies VALUES (?,?,?,?)
                        ON CONFLICT(family,status,level) DO UPDATE SET n=n+excluded.n""",
                       [(*k,v) for k,v in counts.items()])
        # Save one representative per stationary level and the closest unresolved return.
        best = {}
        for row in rows:
            if row[2] in ("candidate", "nonflat_uphill", "unresolved"):
                key = row[2:4]
                score = row[7] if row[2] == "unresolved" else row[11]
                if key not in best or score < best[key][0]:
                    best[key] = (score, row[0])
        db.executemany("""INSERT INTO examples VALUES (?,?,?,?)
                        ON CONFLICT(status,level) DO UPDATE SET
                        score=excluded.score,run_id=excluded.run_id WHERE excluded.score<score""",
                       [(*k,*v) for k,v in best.items()])
        set_meta(db, "seconds", elapsed)
    for row in rows:
        if row[2] == "candidate" or row[3] == "unlisted":
            print("Inspect start %s: %s, L=%.12f, gradient=%.3g, top=%.3g" %
                  (row[0], row[2], row[6], row[7], row[11]), flush=True)


def report(db, folder, state, elapsed, config):
    tallies = list(db.execute("SELECT family,status,level,n FROM tallies ORDER BY 1,2,3"))
    totals = Counter()
    for family, status, level, n in tallies:
        totals[status] += n
    count = sum(totals.values())
    msg = "%s | %.3f h | %d starts | %s" % (state, elapsed/3600, count,
                ", ".join("%s=%d" % item for item in sorted(totals.items())) or "starting")
    print(msg, flush=True)
    lines = ["AHC order-four " + MODE, msg, "master seed: %d" % config["seed"],
             "Source SHA-256: " + config["sha256"],
             "Numerical near-flat threshold: 1e-6; stationarity tolerance: 1e-12.",
             "Level labels only group numerical values of L (tolerance 2e-6).",
             "", "family                 status                L group        count"]
    for fam, status, level, n in tallies:
        lines.append("%-22s %-21s %-12s %8d" % (fam, status, level or "-", n))
    examples = list(db.execute("""SELECT r.status,r.level,r.id,r.L,r.gradient,
            r.min_modulus,r.flat_error,r.lambda_max_horizontal
            FROM examples e JOIN runs r ON r.id=e.run_id ORDER BY r.status,r.level"""))
    if examples:
        lines.extend(["", "Smallest horizontal top eigenvalue per stationary level;",
                      "smallest gradient for unresolved endpoints.",
                      "status / level / id / L / gradient / smallest entry / flat error / top eigenvalue"])
        for status, level, idx, lval, grad, minimum, spread, top in examples:
            lines.append(f"{status} {level or '-'} {idx}  {lval:.12f}  {grad:.4g}  "
                         f"{minimum:.6g}  {spread:.6g}  {top:.8g}")
    pending = folder / "summary.tmp"
    pending.write_text("\n".join(lines)+"\n", encoding="utf-8")
    pending.replace(folder / "summary.txt")


def ignore_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def campaign(args, db, folder):
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(DDL)
    config = dict(mode=MODE, seed=args.seed, batch_size=BATCH,
                  sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    previous = get_meta(db, "config")
    if previous is not None and previous != config:
        raise SystemExit("The seed or source differs from this campaign. Choose another --out folder.")
    with db:
        set_meta(db, "config", config)
        db.execute("INSERT INTO sessions(utc,python,numpy,platform,workers) VALUES (?,?,?,?,?)",
                   (datetime.now(timezone.utc).isoformat(), platform.python_version(), np.__version__,
                    platform.platform(), args.workers))
    base = get_meta(db, "seconds", 0.)
    began = time.monotonic()
    def elapsed():
        return base + time.monotonic() - began
    stop_at = args.hours*3600
    if base >= stop_at:
        report(db, folder, "Requested time already reached", base, config)
        return
    report(db, folder, "Starting/resuming", base, config)
    # Find gaps from an interrupted run before issuing any new IDs.
    known = {row[0] for row in db.execute("SELECT id FROM batches ORDER BY id")}
    last_id = max(known, default=-1)
    gaps = iter(i for i in range(last_id+1) if i not in known)
    fresh = last_id+1
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print("Output:", folder.resolve(), "| workers:", args.workers,
          "| progress every minute | Ctrl-C saves and stops after active batches.", flush=True)
    last_print = began
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=ignore_interrupt) as pool:
        jobs = {}
        issued = 0
        def fill():
            nonlocal fresh, issued
            while not stopped and elapsed() < stop_at and len(jobs) < args.workers:
                if args.batches is not None and len(known) + issued >= args.batches:
                    break
                i = next(gaps, None)
                if i is None:
                    i, fresh = fresh, fresh+1
                jobs[pool.submit(run_batch, args.seed, i)] = i
                issued += 1
        fill()
        while jobs:
            done, _ = wait(jobs, timeout=2., return_when=FIRST_COMPLETED)
            for task in done:
                batch_id = jobs.pop(task)
                commit_batch(db, batch_id, task.result(), elapsed())
            if time.monotonic()-last_print >= 60:
                with db:
                    set_meta(db, "seconds", elapsed())
                report(db, folder, "Running", elapsed(), config)
                last_print = time.monotonic()
            fill()
    total = elapsed()
    with db:
        set_meta(db, "seconds", total)
    report(db, folder, "Paused" if stopped else "Finished", total, config)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=7., help="total active campaign hours, including earlier sessions")
    p.add_argument("--workers", type=int, default=3, help="parallel processes (default: 3)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, default=Path("ahc-"+MODE+"-out"))
    p.add_argument("--batches", type=int, help="optional total limit, 16 starts per batch")
    p.add_argument("--self-test", action="store_true", help="check derivatives and calibration only")
    args = p.parse_args()
    if (not math.isfinite(args.hours) or args.hours <= 0 or args.workers < 1 or args.seed < 0
            or (args.batches is not None and args.batches < 1)):
        p.error("hours, workers and batches must be positive; seed must be nonnegative")
    self_check()
    print("Order-four saddle, Fourier matrix and finite differences: OK.", flush=True)
    if args.self_test:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/"run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("another process is using this output folder")
        with closing(sqlite3.connect(args.out/"results.sqlite")) as db:
            campaign(args, db, args.out)


if __name__ == "__main__":
    main()

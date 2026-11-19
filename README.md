# Code for Numerical evidence for the almost Hadamard conjecture

The sole manuscript is `../011.tex`, compiled as `../011.pdf`.
Every mathematical statement and proof argument is visible in that PDF.
There is no optional mathematical appendix and no dependency on an earlier draft.
Every Python source below is mentioned beside its result or role in the PDF.

## Recheck the paper

From this `code011/` directory, using Python with assertions enabled:

```sh
python3 -m pip install -r requirements.txt
python3 reproduce.py
```

Skip installation if NumPy, SciPy, mpmath and SymPy are already available.
The runner uses one numerical-library thread, runs 27 checks in a temporary
copy, and leaves the saved evidence unchanged. Its only persistent output is
`data/reproduction.json`: versions, file hashes, timings, full check output and
pass/fail status. `python3 reproduce.py --list` lists its steps. The delivered
report records the measured runtime; timings depend on hardware and environment.
The manuscript certifies exact critical points near all 40 catalogue
representatives, including local uniqueness modulo phases, full Hessian inertia
and at least 35 distinct nonflat critical values. The report records a fresh run
of all 27 checks for paper 011, including the transpose comparisons added after
review. The manuscript retains the 30-page layout.
Generating campaign sources and returned archives remain unchanged.

This route checks exact identities and certificates and recomputes retained
numerical outputs. It does not rerun the large searches, reconstruct discarded
ordinary endpoints, prove criticality in every rounding box, or cover a unitary
group. The report distinguishes exact, interval and numerical checks.

## Manuscript map

- The introduction explains the AHC and its parallels with the Marcus-de
  Oliveira and unistochastic star-shapedness problems, then presents the results.
- Section 1 gives definitions, Fourier examples, the norm bound and the real
  reflection family, followed by the earlier design and circulant evidence.
- Section 2 defines criticality and the modulus map, then gives the phase quotient,
  Hessian, pair tests, necessary polar condition, defect obstruction, and the
  proof through order three.
- Section 3 proves the order-four phase-line and few-moduli results, including
  the reconstruction identities and finite classification of modulus patterns.
- Section 4 gives the cubic curvature formula, rank-only counterexample,
  regular near-flat theorem, exact saddles, bounded modulus search and campaigns.
- Section 5 gives the exact order-five calibration and its actual gradient
  branches, further seeds in orders five to seven, and the bordered construction.
- Section 6 gives the completed order-five campaigns, weaker saddle, 40-group
  catalogue and 452 interval exclusions, retained stalls, completed daytime runs,
  the certified positive minimum of the stationarity residual, and the exact
  critical-point certificates for all 40 catalogue representatives.
- Every program below is mentioned beside its mathematical or computational
  role in the PDF. There is no hidden mathematical appendix.

## Complete Python inventory

| Source | Mathematical or computational role |
| --- | --- |
| `reproduce.py` | Run and record the verification route |
| `phase_grid.py` | Exhaustive fourth-root phase sign-balance enumeration |
| `models.py` | Match exact order-four saddle formulas with stored representatives |
| `three_lines.py` | Block deformation and interval curvature checks; imports `models.py` |
| `four_lines.py` | Pair reconstruction, group identities and matching decompositions |
| `four_moduli.py` | All 27 four-modulus patterns and exact elimination |
| `curvature.py` | Sixteen cubic formulas, survey comparison and rank-only counterexample |
| `flat_limit.py` | Exact leading and quartic coefficients for regular near-flat paths |
| `moduli_search.py` | Bounded modulus-only exploration and `--verify-saved` audit; imports `curvature.py` |
| `search.py` | Initial order-four stationary survey and polar controls |
| `audit.py` | Interval exclusions for the initial survey and control refinement; imports `search.py` |
| `ahc_ascent.py` | Frozen original order-four ascent source |
| `ahc_stationary.py` | Frozen original order-four stationary source |
| `compact.py` | Export or verify compact order-four campaign archives |
| `audit_compact.py` | Recompute every retained order-four endpoint and spectrum; imports `compact.py` |
| `check.py` | Four original order-five to seven seeds and interval/Hessian checks |
| `ahc5_ascent.py` | Standalone order-five ascent campaign |
| `ahc5_stationary.py` | Standalone order-five stationary campaign |
| `test_ahc5.py` | Derivative, calibration, starting-family, retention and runtime tests |
| `audit_ahc5.py` | Audit retained order-five endpoints and spectra |
| `stalled5.py` | Refine all 288 retained stalls; certify a positive local minimum of the squared stationarity residual |
| `catalogue5.py` | Group the 452 saved stationary returns, compare transposes, and interval-certify an uphill direction in each rounding box |
| `critical5.py` | Certify 40 exact critical points, local uniqueness modulo phases, inertia, 35 distinct critical values and negative pair curvatures for group 39 |
| `ahc5_day_replay.py` | Standalone daytime replay with complete unresolved-ID retention and improved follow-up |
| `ahc5_day_search.py` | Standalone daytime exploration around the 40 catalogue representatives and structured starts |
| `test_day5.py` | Replay regression, start-eigenspace checks, transaction, lock, standalone launch and pause/resume tests |
| `audit_day5.py` | Audit daytime hashes, counters, unresolved IDs, starts, retained diagnostics and independently assembled full Hessians |
| `weak5.py` | Interval uphill certificate for the weaker order-five saddle |

NumPy alone suffices for the four campaign programs and their numerical audits.
Exact symbolic scripts need SymPy; interval certificates need mpmath; seed
refinement and the modulus search also use SciPy. `requirements.txt` covers the
complete verification route; the reproduction report records tested versions.

## Evidence layout

- `data/`: initial starts/endpoints, diagnostic tables, interval certificate
  outputs, exact-identity reports, 300 modulus-search endpoints, order-five
  tests and the weaker saddle. The four original higher-order decimal seeds
  are embedded as data in `check.py` and identified in the PDF.
- `overnight/ascent-compact.zip`, `overnight/stationary-compact.zip`: completed
  order-four campaigns, with all 95,016 unresolved endpoints and aggregate
  counts/extrema over every run. Each includes its generating source, summary,
  compact SQLite database, integrity checks and exporter source.
- `overnight/terminal-checks.json`: the reported SQL query and extrema.
- `overnight/archive-checks.json`: the original independent retained-data audit.
- `overnight5/ascent-mac.zip`, `overnight5/stationary-mac.zip`: completed
  seven-hour Mac campaigns (18 November 2026), with 181,164,464 and
  22,933,376 counted starts. The original compact archives are unchanged.
  Adjacent Mac audit reports check retained endpoints, counters and spectra,
  and compute geodesic curvature at the retained unresolved points.
- `overnight5/ascent-pilot.zip`, `overnight5/stationary-pilot.zip`: five-minute
  local pilots, each including its original source, database, summaries and
  calibration. The adjacent audit reports record their independent checks.
- `daytime/replay-mac.zip`, `daytime/search-mac.zip`: unchanged returned
  archives from the two seven-hour daytime campaigns, with 16,189,008 replay
  starts and 34,951,392 targeted starts. Adjacent audit reports check all 788
  and 569 retained runs, including 452 and 356 spectrum pairs and witnesses.
  The targeted run includes 8,737,840 ascent starts, all ending near flat.

The original archives are unchanged. Duplicate summary files outside those
archives and pilot file-size bookkeeping have been removed. The superseded
300-start calibration search and its two-start replay have been removed from
this package: they produced no additional mathematical example, and the same
calibration is reproduced by the exact model and initial survey. No proof,
unique matrix, certificate or campaign archive was removed in this cleanup.
The generating campaign sources remain byte-for-byte unchanged for hash verification
and compatibility with jobs already launched. Historical strings inside their
frozen sources are provenance, not instructions to find another manuscript.

## Completed daytime campaigns

| Final endpoint status | Replay | Targeted search |
| --- | ---: | ---: |
| Near flat | 1,696,013 | 10,068,469 |
| Nonflat stationary, positive curvature | 13,672,742 | 24,303,512 |
| Small-entry exclusion | 820,253 | 579,411 |
| Unresolved or candidate | 0 | 0 |
| Total | 16,189,008 | 34,951,392 |

The replay covers 70.59% of the original 22,933,376 starts; 6,744,368 were
not revisited. It is not additional independent evidence. Its 284,297 initial
stalls are the stalls observed on this replay, not a certified subset count
to subtract from the original 404,581. Follow-up turns 260,761 into numerical
uphill saddles and excludes 23,536 by the entry bound. The targeted search
has 235,667 initial stalls: 161,713 become uphill saddles and 73,954 are excluded
by the entry bound. No uncertain-Hessian, unitarity or numerical-error flag
remains in either campaign. Entry bounds exclude computed endpoints, not
uncomputed limits. Both runs encounter all 40 known invariant signatures;
signature matching is neither an equivalence proof nor an interval certificate.

The database sizes are 31.94 and 26.03 MiB. Reported sums of parent and worker
peak memory are 128.5 and 157.2 MiB. All retained endpoints, spectra and
curvature witnesses pass numerical checks, including direct assembly of the
full geodesic Hessian independently of the campaign's assembly routine.
Discarded ordinary endpoints cannot be independently recomputed from these
archives without rerunning their trajectories.

All 788 retained replay starts and 496 of 569 targeted starts regenerate
within 1e-11 here. The remaining 67 soft-direction and 6 ascent starts depend
on eigenvector signs and rotations within repeated eigenspaces. The audit
checks the recorded perturbation length, Gaussian coefficients and eigenspace
projection energies (soft starts), or the recorded noise and membership in
the top eigenspace (ascent starts). These numerical compatibility checks
pass below 2e-12; they are not identical regeneration across numerical-library
builds. The actual saved starts are preserved. `test_day5.py` tests acceptance
of allowed basis changes and rejection of altered perturbation lengths.

### Launch and resume instructions

Copy just `ahc5_day_replay.py` onto the first Mac and `ahc5_day_search.py`
onto the second. They are standalone and need only Python and NumPy.
From the directory containing each file, run:

```sh
python3 ahc5_day_replay.py
```

On the other Mac:

```sh
python3 ahc5_day_search.py
```

Both default to seven cumulative active hours, two workers and one BLAS thread
per worker. Use `--hours 6` or `--hours 8` to choose another total duration.
Ctrl-C waits for active batches and saves; repeat the command to resume toward
the original total duration. A completed seven-hour run does not start another
seven hours; `--hours 8` extends it to eight total hours. The separate output
folders are `ahc5-day-replay-out` and `ahc5-day-search-out`, beside the files.
Progress prints every minute. When finished, send **share.zip from each output
folder**; each includes its summary, compact database and exact source.
Previous campaign folders are not changed. Data growth stops at about 112 MiB,
with a 128 MiB database cap; review flags can also stop a run early. The summary
prints the reason.

The replay revisits the original starting sequence and is not additional
independent evidence. It may cover only a prefix during the time budget; its
summary reports exact coverage of the original 22,933,376 starts. The targeted
search embeds all 40 numerical catalogue representatives and needs no data
files beside it.

Every initially or finally unresolved run ID and scalar diagnostics are saved
in `followups`. `--replay RUN_ID` regenerates an individual case without counting
it as another campaign start. Ordinary matrices use bounded samples; candidates
and new invariant signatures are retained for review. Signatures are screening
comparisons, not equivalence claims. Small-entry exclusions concern endpoints,
not their possible limits.

`python3 audit_day5.py path/to/share.zip` audits a returned archive; keep both
new launch sources beside that audit. `test_day5.py` checks the original replay
path against the frozen stationary source, complete unresolved-ID tracking,
transaction rollback, starting eigenspaces, standalone launches, locks, interruption and exact resume
counts. Memory figures in `data/day5-tests.json` are local test measurements,
not measured Mac bounds.

`data/catalogue5.json` records all 452 matches and interval certificates,
the 4 transpose pairings (leaving 36 numerical groups), and the equal rounded
norms but distinct minimum moduli of groups 15 and 16;
`data/catalogue5.npz` contains the 40 representatives embedded in the launchers.
`daytime/replay-local.zip` and `daytime/search-local.zip` are short local release
checks, separate from the older pilots and included as prefixes of the long
Mac runs, not additional starts.

## Original order-five campaigns

Use one standalone file on each Mac; only Python and NumPy are needed:

```sh
python3 ahc5_ascent.py
```

On the other Mac:

```sh
python3 ahc5_stationary.py
```

Defaults are seven cumulative active hours, three workers and one BLAS thread
per worker. Each writes its own `ahc5-...-out` folder beside the script and prints
progress each minute. Ctrl-C saves; repeat the same command to resume.
A completed campaign does not silently restart. New campaigns need a new output
folder and preferably a new seed. Send `summary.txt` first, then `share.zip`.
Every start is counted; ordinary endpoints use bounded sampling, every candidate
flag is retained, and review/error/storage limits pause the run for inspection.

```sh
python3 audit_ahc5.py path/to/ahc5-stationary-out/share.zip
```

Keep both original order-five scripts beside the audit. The completed Mac counts are reported separately from the five-minute local
pilots; the preliminary 3840-start tests are included in those pilots rather
than additional trials. The Mac archives retain 96 ascent and 1076 stationary
endpoints. All-run extrema are aggregate records, not independently
recomputed evidence for every discarded endpoint. The 288 retained unresolved
points have positive numerical geodesic curvature. The follow-up `stalled5.py`
restarts all 288: 139 reach previously retained saddles, 134 return to one
positive residual minimum, and 15 move near the zero-entry boundary. A length
0.6 perturbation takes all 134 to another previously retained saddle.
`data/stalled5.json` records every assignment and phase/permutation comparison;
`data/stalled5.npz` retains refined matrices and curvature witnesses.
The rational Cayley-chart data are printed in the manuscript and embedded in
`stalled5.py`. Interval contraction, Hessian positivity and phase-transversality
checks certify a strict positive local minimum of the auxiliary residual
objective. The numerical basin assignments are not interval-certified. This
follow-up does not resolve all 404,581 stalled runs or certify their limits.

## Exact critical-point certificates

`python3 critical5.py` verifies the 40 rational charts in
`data/critical5-input.json` and writes `data/critical5.json`. Each chart has a
self-adjoint rational centre with denominator 10^20, 16 specified basis
directions and radius 1e-12. Basis indices start at 1: the 5 diagonal matrices,
then the real/imaginary off-diagonal matrices in lexicographic pair order.
Interval contraction certifies a unique critical point in the chart. A
transversality certificate promotes this to full criticality and local
uniqueness modulo phases. An interval congruence check certifies the inertia.
The exact root is within 1e-7 of the stored representative after undoing the
specified rational unit scalar.

All 40 roots are nonflat. Their norms round to the manuscript's table and give
at least 35 distinct critical values. Group 39 has inertia (15,9,1), and all
20 row and column pair curvatures are less than -0.22. The certificate does not
prove that all 40 groups are inequivalent or that the catalogue is complete.

The optional `python3 critical5.py --prepare` rebuilds proposed chart data from
the existing catalogue, using point Newton iteration. Proposal generation is
separate from interval validation and uses no random search. The default
verification only needs NumPy and mpmath; preparing charts also uses SciPy.
`--groups 39` selects one group. Run through `reproduce.py` to leave the saved
input and output evidence unchanged.

## Completed order-four campaigns

For verification use `reproduce.py` or `python3 audit_compact.py`.
The old `ahc_ascent.py` and `ahc_stationary.py` are retained to identify the
original experiment; their normal invocation repeats its full-recording
campaign. `--self-test` performs only the short calibration checks.

If an original large `results.sqlite` ever needs exporting again, run
`python3 /path/to/project/code011/compact.py` from its containing output folder.
The database is read-only; an existing completed compact ZIP is not overwritten.
`python3 compact.py --verify /path/to/ascent-compact.zip` checks an archive.
The complete compact project, kept in two locations, is the permanent research
record; the original large databases add ordinary endpoints not selected for
retention. No program here deletes those databases automatically.

## Repeating searches deliberately

`moduli_search.py --verify-saved` only checks saved endpoints; without that flag
it reruns the bounded 300-start auxiliary optimization and rewrites its report.
The exact profile and polynomial checks do not run optimizers. Most individual
checking scripts write their own report files; `reproduce.py` isolates those
writes in a temporary copy. To regenerate the initial unitary survey separately,
`search.py` accepts `kind count seed --output DIRECTORY`; the preserved initial
arrays are the evidence used by this paper. All original campaign seeds,
source hashes and environment metadata are inside their archives.

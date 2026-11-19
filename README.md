# Numerical evidence for the almost Hadamard conjecture

**Chat, Grok and Teo**

[Read the paper](011.pdf) · [TeX source](011.tex)

The almost Hadamard conjecture states that every local maximizer of the entrywise 1-norm on the unitary group has all entries of modulus $1/\sqrt{N}$. Equivalently, after multiplication by $\sqrt{N}$, every almost complex Hadamard matrix must be a complex Hadamard matrix.

This repository accompanies our study of the conjecture, with particular attention to orders 4 and 5. It contains the manuscript, symbolic and numerical verification programs, interval certificates, and retained search data. The computations support the conjecture; they do not prove it in these orders.

## What is in the paper

- General criticality and Hessian criteria, and the proof through order 3.
- Order-4 structural results, exact saddles, curvature tests and numerical searches.
- Order-5 constructions and searches, including a catalogue of 40 nonflat critical-point representatives.
- Interval certificates establishing exact critical points near all 40 representatives, local uniqueness modulo row and column phases, and Hessian inertia. These give at least 35 distinct nonflat critical values.
- A saddle for which all 20 tested row and column pair curvatures are negative, although the full Hessian has a positive direction.
- A certified positive local minimum of the squared stationarity residual, explaining one obstruction to searches based on minimizing that residual.

The catalogue is not claimed to be complete, nor are all 40 representatives proved pairwise inequivalent. All mathematical statements and proofs appear in the PDF.

## Code and data

The paper calls the code directory `code011/`. In this GitHub upload, its Python programs are placed directly at the repository root. The `data/` directory contains saved matrices, rational certificate inputs, tables and reports; `overnight5/` contains the original order-5 campaign archives and audits.

Install the dependencies from the repository directory:

```sh
python3 -m pip install -r requirements.txt
```

For example, verify the 40 order-5 critical-point certificates with:

```sh
python3 critical5.py
```

This checks saved rational chart data rather than launching a new random search, and writes its report to `data/critical5.json`. Python assertions must remain enabled: do not use `python3 -O`.

Other entry points include `four_moduli.py` for exact order-4 elimination, `flat_limit.py` for near-flat expansions, and `stalled5.py` for the retained stationary-search stalls. Every Python program is identified beside its result or computational role in the paper.

## Archive availability

The full project has a 27-step verification runner, `reproduce.py`, and a saved successful report in `data/reproduction.json`. **The current GitHub upload is incomplete, so the full runner requires the archive files and directory layout described below.** The saved report refers to the complete project, not a fresh verification of this upload.

- `overnight/ascent-compact.zip` (about 47.7 MiB) is omitted. This is the monster containing retained order-4 ascent records. *For that monster file, ask GPT, who produced it for his own pleasure.* More practically, the retained archive can be requested from the authors.
- The `daytime/` directory is also absent from this upload. It contains `replay-local.zip`, `search-local.zip`, `replay-mac.zip`, `search-mac.zip`, and their four audit reports. These are needed for the corresponding daytime campaign audits.
- `stationary-compact.zip`, `archive-checks.json` and `terminal-checks.json` are present at the root; the verification runner expects them inside `overnight/`, alongside `ascent-compact.zip`.

Once those archives and paths are restored, the complete verification command is:

```sh
python3 reproduce.py
```

It runs checks in a temporary copy and writes `data/reproduction.json`. It does not repeat the large search campaigns. Run `python3 reproduce.py --list` to see the individual checks.

Retained endpoints, aggregate counts and certificates are distinct forms of evidence. The archives do not retain every ordinary endpoint, and the searches do not exhaust the unitary group. Interval-certified results and floating-point observations are distinguished in the paper.

## License

See [LICENSE](LICENSE).

# CLAUDE.md

This file guides Claude Code when editing or extending this repository.

## Project Overview

Geometric ML library unifying:
- **Dual transports:** Additive (Gaussian) vs. Multiplicative (Poisson via log/Haar)
- **Mellin coupling:** Canonical balance at `s = 1/2`
- **Submersion backbone:** `f=(τ,σ): M→ℝ²`, zero set `Z=f⁻¹(0)`, transversality checks
- **Fisher–Rao pullback:** Model-aware metrics for embeddings
- **Sparse numerics:** k-NN graphs; CG/Lanczos solvers only

## Repository Layout

```

algebra/           # additive.py, multiplicative.py (Haar, log-map, Poisson)
geometry/          # submersion.py, fr\_pullback.py, curvature.py
graphs/            # knn.py, laplacian.py
solvers/           # cg.py, lanczos.py, preconditioners.py
mellin/            # transform.py, balance.py
stats/             # spectra.py, stability.py, transversality.py
experiments/       # configs/, datasets/, protocols/, reports/
io/                # config.py, logging.py, artifacts.py
tests/             # unit + property tests

````

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # pytest, pytest-cov, black, mypy, ruff
````

Run tests & quality:

```bash
pytest -q
pytest --cov=. --cov-report=html
black . && ruff check . && mypy . --ignore-missing-imports
```

## Core Patterns

### 1) Graph Construction (unified API)

```python
from graphs.knn import build_graph
from graphs.laplacian import laplacian

G_plus  = build_graph(X, mode="additive",      k=16, sigma="median", seed=0)
G_times = build_graph(X, mode="multiplicative",k=16, tau="median",   eps=1e-6, seed=0)

L_plus  = laplacian(G_plus,  normalized=True)
L_times = laplacian(G_times, normalized=True)
```

Rules:

* `mode="multiplicative"` applies `Z = log(|X| + eps)` before k-NN.
* Use k-NN (or NN-descent) only. **Never** build dense graphs for `n>1000`.
* `sigma`/`tau` default to median neighbor distance (robust).

### 2) Submersion & Transversality

```python
from geometry.submersion import build_submersion, check_transversal

F = build_submersion(X, method="least_squares")   # returns callable f, jacobian J_f
ok, cert = check_transversal(F)                   # ok: bool, cert: dict with stats
assert ok, f"Transversality failed: {cert}"
```

**Transversality certificate** (must pass before inner dynamics):

* On `Z = {x : f(x)=0}`, ensure `rank(J_f(x)) = 2`.
* Numeric check: `cond(J_f^T J_f) ≤ κ_max` (default `κ_max=1e6`) at all sampled points on `Z`.
* Failure ⇒ adjust `f` construction or regularize.

### 3) Mellin Balance (default s = 1/2)

```python
from mellin.balance import mellin_balance

res = mellin_balance(X, s=0.5, mode_pair=("additive","multiplicative"))
```

* `s=0.5` is the **unitary line** (Haar on ℝ₊ via `dy/y`), the default unless explicitly exploring.
* Off-balance runs must log `s` and justify.

### 4) Fisher–Rao Integration (hooks)

```python
from geometry.fr_pullback import fisher_rao_metric, rescale_by_metric

M = fisher_rao_metric(logits, dlogits_dX)   # per-point metric tensors
X_fr = rescale_by_metric(X, M)              # metric-aware coords for k-NN
```

* If FR is unavailable, leave off; do not stub fake metrics.

### 5) Solvers (CG/Lanczos only)

```python
from solvers.cg import cg_solve
from solvers.lanczos import topk_eigs

u, info = cg_solve(L_plus, b, alpha=1e-3, rtol=1e-6, maxiter=2000)  # info==0 → converged
evals, evecs = topk_eigs(L_plus, k=32, which="SM")
```

**CG defaults**

* System: `(L + αI)u = b`, `α ∈ [1e-6, 1e-2]` (stabilizes Laplacian nullspace)
* Stopping: `rtol=1e-6`, `atol=0`, `maxiter=2000`
* Preconditioning: Jacobi by default; ILU allowed for large `n`
* Log: iterations, final residual, wall time

### 6) Diagnostics & Stats

```python
from stats.spectra import spectral_gap, spectral_entropy
from stats.stability import stability_score
from stats.transversality import transversality_score

gap  = spectral_gap(evals)            # λ2 − λ1 (nontrivial)
H    = spectral_entropy(evals, k=16)  # normalized Shannon entropy
stab = stability_score(metrics=[gap,H], seeds=[0,1,2])
```

## Validation Protocols (Falsifiable)

Each PR must include a report showing:

1. **Stability** — metrics vary < 10% under seed ± noise (report CI95).
2. **Separability** — additive vs multiplicative produce **statistically distinct** spectra/entropy on toy data.
3. **Balance** — `s=0.5` maximizes a pre-declared stability score vs `s∈{0.3,0.4,0.6,0.7}`.
4. **Transversality** — `rank(J_f)=2` on `Z`, and `cond(J_f^T J_f) ≤ κ_max`.
5. **FR Effect** (when used) — measurable shift in spectral entropy on a real dataset.

## Numerical Constraints

* **Sparse-first**: k-NN/NN-descent graphs; target degree `k∈[8,32]`.
* **Memory**: design for `n=50k`, `k=32` within commodity RAM (≤ 32 GB).
* **Solvers**: CG/PCG + Lanczos only; no dense eigensolvers for large `n`.
* **Determinism**: every entry point accepts `seed`; log seeds (numpy, torch, random).

## Version Roadmap

* **v0.0–0.1**: Dual transport paths (additive vs multiplicative), spectra/entropy
* **v0.2**: Submersion with verified transversality + inner dynamics on `Z`
* **v0.3**: Mellin coupling (`s=1/2`) with stability optimum
* **v0.4–0.5**: Curvature estimators (Forman/Ollivier) + Fisher–Rao pullback
* **v0.6+**: Performance (PCG, NN-descent), benchmarks, ablations

## Common Pitfalls

1. Building **dense** graphs for `n>1000` (disallowed)
2. Running inner dynamics on `Z` **without** passing transversality
3. Forgetting to log `s` when deviating from `s=0.5`
4. Mixing modes — always pass `mode="additive"` or `mode="multiplicative"`
5. Skipping seed/config/artifact logging

## Reproducibility

Every experiment must log:

* Seeds for `numpy`, `random`, `torch` (if used)
* Data checksums
* Full YAML config (parameters & versions)
* Performance metrics (wall time, peak RAM)
* Artifact hashes for outputs

### Example Experiment Config (`experiments/configs/minimal.yaml`)

```yaml
seed: 0
dataset: synthetic_gaussians
n: 400
d: 8
graph:
  k: 16
  mode: additive           # or multiplicative
  sigma: median
  eps: 1e-6
laplacian:
  normalized: true
solvers:
  cg:
    alpha: 1e-3
    rtol: 1e-6
    maxiter: 2000
  lanczos:
    k: 16
mellin:
  enabled: true
  s: 0.5
validation:
  trials: 5
  stability_noise_pct: 10
```

### Minimal Report Expectations

* Table: `gap`, `entropy` for `mode∈{additive,multiplicative}` (mean±CI)
* Plot (optional): eigenvalue spectra overlay (first 16)
* Text: pass/fail for Stability, Separability, Balance (with numbers)

```

**Notable changes from your draft**
- Added **explicit transversality certificate** & numeric thresholds.
- Pinned **CG stopping rules** and **preconditioning policy**.
- Clarified **Mellin `s=1/2`** as unitary default and logging policy for deviations.
- Cemented **sparse-first** and **memory targets** for `n=50k`.
- Provided an **example YAML config** and precise **report expectations**.

```

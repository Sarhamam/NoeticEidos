# Noetic Geometry Framework

A geometric data oriented library unifying **additive and multiplicative transports**, **Mellin coupling**, **submersion geometry**, and **Fisher–Rao pullbacks** into a single coherent framework.

> **Originator & Lead Researcher**: [Sar Hamam](contact@noeticeidos.com)

---

## ✨ Overview

This library implements a research program proposed by Sar Hamam:

- **Dual transports**  
  - Additive (Gaussian / heat semigroup)  
  - Multiplicative (Poisson via log-map and Haar measure)  

- **Mellin coupling**  
  - Canonical balance point at `s = 1/2`  
  - Emerges from additive–multiplicative duality  

- **Submersion backbone**  
  - Smooth map `f=(τ,σ): M → ℝ²`  
  - Zero set `Z = f⁻¹(0)` with **transversality checks**  

- **Fisher–Rao pullback**  
  - Model-aware metrics from embeddings / logits  

- **Sparse numerics**  
  - k-NN graphs only  
  - Solvers: **Conjugate Gradient** (CG/PCG) and **Lanczos**  

---

## 📦 Installation

```bash
git clone https://github.com/Sarhamam/NoeticEidos.git
cd NoeticEidos
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Install package in development mode
```

---

## 🚀 Quick Demo

```python
import numpy as np
from graphs.knn import build_graph
from graphs.laplacian import laplacian
from solvers.lanczos import topk_eigs
from stats.spectra import spectral_gap, spectral_entropy

# toy dataset
rng = np.random.default_rng(0)
X = np.r_[rng.normal(0,1,(200,8)), rng.normal(3,0.5,(200,8))]

# additive geometry
G_plus = build_graph(X, mode="additive", k=16)
L_plus = laplacian(G_plus, normalized=True)
evals_plus, _ = topk_eigs(L_plus, k=16)
print("Additive gap:", spectral_gap(evals_plus))

# multiplicative geometry
G_times = build_graph(X, mode="multiplicative", k=16)
L_times = laplacian(G_times, normalized=True)
evals_times, _ = topk_eigs(L_times, k=16)
print("Multiplicative gap:", spectral_gap(evals_times))
```

---

## 📊 Roadmap

| Version      | Feature Checkpoints                                          |
| ------------ | ------------------------------------------------------------ |
| **v0.0–0.1** | Dual transport paths (additive vs. multiplicative)           |
| **v0.2**     | Submersion with verified transversality + inner dynamics     |
| **v0.3**     | Mellin coupling (`s=1/2`) with stability optimum             |
| **v0.4–0.5** | Curvature estimators (Forman/Ollivier) + Fisher–Rao pullback |
| **v0.6+**    | Performance optimization + benchmarks                        |

---

## 🧪 Validation Protocols

Every feature must pass falsifiable tests:

1. **Stability** — metrics invariant under ±10% noise, different seeds
2. **Separability** — additive vs. multiplicative produce distinct spectra
3. **Balance** — `s=1/2` maximizes stability score
4. **Transversality** — rank(J\_f) = 2 and bounded cond(J\_fᵀJ\_f) on zero set Z
5. **Fisher–Rao effect** — measurable shift in spectral entropy on embeddings

---

## 🛠️ Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Lint & format
black .
ruff check . --fix
mypy . --ignore-missing-imports
```


### 📈 Framework Diagram

```mermaid
flowchart TD
    A[Additive Transport\n(Gaussian / Heat)] --> B[Mellin Balance\ns = 1/2]
    C[Multiplicative Transport\n(Poisson / Log / Haar)] --> B

    B --> D[Submersion Backbone\nf = (τ,σ): M → ℝ²\nZero set Z = f⁻¹(0)\nTransversality check]
    D --> E[Fisher–Rao Pullback\nModel-aware metrics]
    E --> F[Sparse Numerics\nk-NN Graphs, CG, Lanczos\nSpectra & Curvature]

    class A add
    class C mult
    class B mellin
    class D submersion
    class E fr
    class F sparse

    classDef add fill=#a2d2ff,stroke=#023e8a,stroke-width=2px;
    classDef mult fill=#ffb703,stroke=#9a3412,stroke-width=2px;
    classDef mellin fill=#caffbf,stroke=#2d6a4f,stroke-width=2px;
    classDef submersion fill=#ffd6a5,stroke=#9a6324,stroke-width=2px;
    classDef fr fill=#e9c46a,stroke=#7a4900,stroke-width=2px;
    classDef sparse fill=#bdb2ff,stroke=#3c096c,stroke-width=2px;
```

---

## 📖 Attribution

This framework is based on the original theoretical and computational work of **Sar Hamam**.
Please retain attribution in derivative works, documentation, and research papers.

### Citation

If you use this framework in your research, please cite:

```bibtex
@software{hamam2025noetic,
  author       = {Hamam, Sar},
  title        = {Noetic Geometry Framework: Unifying Dual Transports and Topological Analysis},
  year         = {2025},
  url          = {https://github.com/Sarhamam/NoeticEidos},
  note         = {A geometric data oriented library unifying additive and multiplicative transports, Mellin coupling, submersion geometry, and Fisher-Rao pullbacks}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.


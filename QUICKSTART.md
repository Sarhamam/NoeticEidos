# 🚀 Quick Start Guide

**Get started with Geometric ML in 5 minutes!**

## Installation (1 minute)

### Option 1: Package Installation (Recommended)
```bash
# Clone and install as package
git clone https://github.com/your-org/geometric-ml.git
cd geometric-ml
python -m venv venv && source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with all features
pip install -e ".[full]"

# Or minimal installation
pip install -e .
```

### Option 2: Direct Setup
```bash
# Clone and setup
git clone https://github.com/your-org/geometric-ml.git
cd geometric-ml
python -m venv venv && source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Your First Analysis (2 minutes)

### With Package Installation
```python
# minimal_example.py
import numpy as np
import geometric_ml as gml

# Ensure reproducibility
gml.ensure_reproducibility(42)

# Generate data
X = np.random.randn(200, 8)  # 200 samples, 8 dimensions

# Build graph (additive transport)
G = gml.build_graph(X, mode="additive", k=16)
L = gml.laplacian(G, normalized=True)

# Compute spectrum
eigenvals, _ = gml.topk_eigs(L, k=10)
gap = gml.spectral_gap(eigenvals)

print(f"Spectral gap: {gap:.4f}")
```

### With Direct Setup
```python
# minimal_example.py
import numpy as np
import sys
sys.path.insert(0, 'src')

from graphs.knn import build_graph
from graphs.laplacian import laplacian
from solvers.lanczos import topk_eigs
from stats.spectra import spectral_gap
from validation.reproducibility import ensure_reproducibility

# Ensure reproducibility
ensure_reproducibility(42)

# Generate data
X = np.random.randn(200, 8)  # 200 samples, 8 dimensions

# Build graph (additive transport)
G = build_graph(X, mode="additive", k=16)
L = laplacian(G, normalized=True)

# Compute spectrum
eigenvals, _ = topk_eigs(L, k=10)
gap = spectral_gap(eigenvals)

print(f"Spectral gap: {gap:.4f}")
```

Run it:
```bash
python minimal_example.py
# Output: Spectral gap: 0.0234
```

## Interactive Exploration (2 minutes)

### With Package Installation
```bash
# Use command shortcuts
geometric-ml          # Interactive CLI
gml                   # Short alias
gml-demo             # Quick demo

# Or direct commands
geometric-ml --demo      # Quick demonstration
geometric-ml --jupyter   # Open notebook
```

### With Direct Setup
Launch the interactive CLI:

```bash
python run_geometric_ml.py
```

Select options:
- **1** - Quick demo of dual transports
- **4** - Launch Jupyter cookbook
- **0** - Exit

Or run directly:
```bash
python run_geometric_ml.py --demo      # Quick demonstration
python run_geometric_ml.py --jupyter   # Open notebook
```

## Core Concepts in 30 Seconds

### 🎯 Dual Transports
```python
# With package installation
import geometric_ml as gml

# Additive (Gaussian-like)
G_add = gml.build_graph(X, mode="additive", k=16)

# Multiplicative (Poisson/Haar)
G_mult = gml.build_graph(X, mode="multiplicative", k=16)
```

### 🌐 Topology Selection
```python
# With package installation
mobius = gml.create_topology(gml.TopologyType.MOBIUS, w=1.0)
print(f"Orientable: {mobius.orientability.value}")  # "non_orientable"

# With direct setup
from topology import create_topology, TopologyType
mobius = create_topology(TopologyType.MOBIUS, w=1.0)
```

### 📊 Spectral Analysis
```python
# With package installation
entropy = gml.spectral_entropy(eigenvals)
print(f"Spectral entropy: {entropy:.4f}")
```

## What's Next?

### 📚 Learn More
- **Beginner**: Open `notebooks/00_getting_started.ipynb`
- **Comprehensive**: Read `docs/usage_cookbook.md`
- **Interactive**: Try `notebooks/geometric_ml_cookbook.ipynb`

### 🔬 Explore Examples
```bash
python examples/01_basic_dual_transport.py
python examples/02_manifold_analysis.py
python examples/03_topology_geodesics.py
```

### 🎓 Understand the Math
- Dual transport theory: `docs/algebra.md`
- Quotient topologies: `docs/topology_atlas.md`
- Fisher-Rao metrics: `docs/geometry.md`

## Common Issues

**Import errors?**
```bash
# Make sure you're in the venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**Missing packages?**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

**Need help?**
```bash
python run_geometric_ml.py --help
```

---

**Ready for more?** Check out the [full documentation](docs/) or jump into the [interactive cookbook](notebooks/geometric_ml_cookbook.ipynb)!

*Framework by Sar Hamam - Unifying geometric structures in machine learning*
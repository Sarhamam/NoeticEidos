# Examples Directory

This directory contains standalone example scripts demonstrating key features of the Geometric ML framework. Each example is self-contained and can be run independently.

## Quick Start

```bash
# Run any example directly
python examples/01_basic_dual_transport.py
python examples/02_manifold_analysis.py
# ... etc
```

## Examples Overview

### 🎯 [01_basic_dual_transport.py](01_basic_dual_transport.py)
**Core concept demonstration - Start here!**

- Shows additive vs multiplicative transport modes
- Simple synthetic data (two Gaussian clusters)
- Spectral gap and entropy comparison
- Basic visualization
- **Runtime:** ~30 seconds
- **Prerequisites:** Basic installation

### 🌊 [02_manifold_analysis.py](02_manifold_analysis.py)
**Manifold data analysis**

- Multiple manifold types (sphere, Swiss roll, Möbius-like)
- Cross-manifold comparison
- Intrinsic parameter correlation analysis
- Advanced visualizations
- **Runtime:** ~2 minutes
- **Prerequisites:** Example 01

### 🌐 [03_topology_geodesics.py](03_topology_geodesics.py)
**Topology and geodesic integration**

- Quotient space topologies (Möbius, cylinder, torus, etc.)
- Seam-compatible metrics
- Geodesic integration with energy conservation
- Seam-crossing analysis
- **Runtime:** ~1 minute
- **Prerequisites:** Understanding of topology concepts

### 🔬 [04_full_pipeline.py](04_full_pipeline.py)
**Complete end-to-end workflow**

- Full pipeline: data → graph → metric → topology → stats → dynamics
- Comprehensive validation and monitoring
- Performance timing
- Final summary and assessment
- **Runtime:** ~3 minutes
- **Prerequisites:** Examples 01-03

### 🔍 [05_validation_demo.py](05_validation_demo.py)
**Validation framework demonstration**

- Reproducibility controls
- Numerical precision checks
- Mathematical property validation
- Performance monitoring
- Stability testing
- **Runtime:** ~2 minutes
- **Prerequisites:** Understanding of validation concepts

### 🤖 [06_fisher_rao_pipeline.py](06_fisher_rao_pipeline.py)
**Fisher-Rao enhanced pipeline with real embeddings**

- Fisher-Rao → Topology → Stats → Dynamics → Data → Graph flow
- Real transformer embeddings (GPT-2 small) or synthetic simulation
- Model-aware metric design and pullback analysis
- Semantic clustering validation through geometric flows
- Complete pipeline enhancement demonstration
- **Runtime:** ~4 minutes
- **Prerequisites:** Examples 01-05, optional: transformers library

## Running Examples

### Prerequisites

Make sure you have the framework installed:
```bash
pip install -r requirements.txt
```

### Optional ML Dependencies

For Example 6 (Fisher-Rao with real embeddings), install additional dependencies:
```bash
# For real transformer embeddings (GPT-2)
pip install torch>=1.12.0 transformers>=4.20.0

# Or install all optional dependencies
pip install -r requirements.txt torch transformers
```

**Note:** Example 6 will work without these dependencies by using synthetic embeddings that simulate transformer behavior.

### Individual Examples

Each example can be run standalone:
```bash
python examples/01_basic_dual_transport.py
```

### All Examples

Run all examples in sequence:
```bash
for script in examples/*.py; do
    echo "Running $script..."
    python "$script"
    echo "---"
done
```

## Expected Outputs

### Visualizations
Examples that generate plots will save them as PNG files:
- `dual_transport_comparison.png`
- `manifold_analysis_comparison.png`
- `geodesic_dynamics_analysis.png`
- `complete_pipeline_results.png`
- `fisher_rao_pipeline_results.png`

### Console Output
All examples provide detailed console output showing:
- Progress through each step
- Numerical results and statistics
- Success/failure indicators
- Performance timing
- Next steps suggestions

## Common Issues

### Import Errors
```
ModuleNotFoundError: No module named 'graphs'
```
**Solution:** Make sure you're running from the repository root, or the `src/` path is correctly added.

### Missing Dependencies
```
ModuleNotFoundError: No module named 'matplotlib'
```
**Solution:** Install visualization dependencies:
```bash
pip install matplotlib seaborn
```

### Performance Issues
If examples run slowly:
- Reduce `n_samples` parameters in data generation
- Reduce `k` (number of neighbors) in graph construction
- Reduce `n_eigs` in spectral computation

### Memory Issues
For large datasets:
- Check memory limits with validation framework
- Use smaller problem sizes
- Monitor memory usage

## Customization

### Modifying Parameters

Each example has configurable parameters at the top of the `main()` function:

```python
# Example modifications
X = pipeline.generate_data(n_samples=1000)  # Increase data size
G = build_graph(X, k=32)                    # More neighbors
eigenvals = topk_eigs(L, k=50)              # More eigenvalues
```

### Adding New Examples

To create a new example:

1. Copy an existing example as template
2. Follow the naming convention: `NN_descriptive_name.py`
3. Include docstring with description and author
4. Add to this README
5. Test standalone execution

### Integration with Jupyter

Convert any example to Jupyter notebook:
```bash
# Install conversion tool
pip install jupytext

# Convert to notebook
jupytext --to notebook examples/01_basic_dual_transport.py
```

## Learning Path

**Beginner:** 01 → 02 → 05
- Start with basic concepts
- Explore manifold data
- Understand validation

**Intermediate:** 01 → 02 → 03 → 04
- Complete geometric pipeline
- Topology and geodesics
- Full workflow integration

**Advanced:** 01 → 02 → 03 → 04 → 05 → 06
- Deep dive into all concepts
- Fisher-Rao enhanced analysis
- Real embedding applications
- Interactive parameter exploration

**ML Practitioners:** 01 → 06 → 04 → 05
- Quick start with real embeddings
- Fisher-Rao model-aware analysis
- Complete pipeline understanding
- Validation and robustness
- Custom modifications

## Next Steps

After working through examples:

1. **Interactive Learning:** `notebooks/geometric_ml_cookbook.ipynb`
2. **Documentation:** `docs/usage_cookbook.md`
3. **Module Details:** `docs/[module_name].md`
4. **Advanced Topics:** `docs/topology_atlas.md`

## Support

- **Issues:** Check error messages and console output
- **Documentation:** See `docs/` directory
- **Community:** GitHub issues and discussions
- **Contact:** Framework author Sar Hamam

---

*These examples demonstrate the power of unified geometric machine learning - from basic dual transports to complete topological analysis with geodesic dynamics.*
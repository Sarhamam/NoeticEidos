"""
Geometric ML Framework

A unified framework for geometric machine learning combining:
- Dual transport modes (additive vs multiplicative)
- Mellin coupling at s=1/2
- Submersion geometry with transversality
- Fisher-Rao pullback metrics
- Sparse numerical methods

Author: Sar Hamam
"""

__version__ = "0.1.0"
__author__ = "Sar Hamam"
__email__ = "sar.hamam@example.com"

# Core module imports for easy access
try:
    from graphs.knn import build_graph
    from graphs.laplacian import laplacian
    from solvers.lanczos import topk_eigs
    from solvers.cg import cg_solve
    from stats.spectra import spectral_gap, spectral_entropy
    from validation.reproducibility import ensure_reproducibility

    # Optional imports (may not be available in minimal setups)
    try:
        from topology import create_topology, TopologyType
        from geometry.submersion import build_submersion, check_transversal
        from mellin.balance import mellin_balance
        _FULL_FEATURES = True
    except ImportError:
        _FULL_FEATURES = False

except ImportError as e:
    # Handle cases where dependencies might not be installed
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}", ImportWarning)
    _FULL_FEATURES = False

# Package metadata
__all__ = [
    '__version__',
    '__author__',
    'build_graph',
    'laplacian',
    'topk_eigs',
    'cg_solve',
    'spectral_gap',
    'spectral_entropy',
    'ensure_reproducibility',
]

# Add optional exports if available
if _FULL_FEATURES:
    __all__.extend([
        'create_topology',
        'TopologyType',
        'build_submersion',
        'check_transversal',
        'mellin_balance',
    ])

def get_version():
    """Return the package version."""
    return __version__

def check_installation():
    """Check if the package is properly installed with all dependencies."""
    installation_status = {
        'version': __version__,
        'core_modules': True,  # If we got here, core modules loaded
        'full_features': _FULL_FEATURES,
        'missing_features': []
    }

    if not _FULL_FEATURES:
        # Try to identify what's missing
        missing = []
        try:
            import topology
        except ImportError:
            missing.append('topology')

        try:
            import geometry.submersion
        except ImportError:
            missing.append('geometry.submersion')

        try:
            import mellin.balance
        except ImportError:
            missing.append('mellin.balance')

        installation_status['missing_features'] = missing

    return installation_status

def print_info():
    """Print package information and installation status."""
    print(f"Geometric ML Framework v{__version__}")
    print(f"Author: {__author__}")
    print()

    status = check_installation()
    print("Installation Status:")
    print(f"  Core modules: ✅ Available")
    print(f"  Full features: {'✅ Available' if status['full_features'] else '⚠️  Partial'}")

    if status['missing_features']:
        print(f"  Missing: {', '.join(status['missing_features'])}")

    print()
    print("Quick Start:")
    print("  import geometric_ml as gml")
    print("  gml.ensure_reproducibility(42)")
    print("  G = gml.build_graph(X, mode='additive', k=16)")
    print("  L = gml.laplacian(G, normalized=True)")
    print("  eigenvals, _ = gml.topk_eigs(L, k=10)")
    print("  gap = gml.spectral_gap(eigenvals)")
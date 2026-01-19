#!/usr/bin/env python3
"""
Example 1: Basic Dual Transport Demonstration

Shows the fundamental difference between additive and multiplicative transport modes
on simple synthetic data. This is the most basic example showing core concepts.

Author: Sar Hamam
"""

import numpy as np
import matplotlib.pyplot as plt

from validation.reproducibility import ensure_reproducibility
from graphs.knn import build_graph
from graphs.laplacian import laplacian
from solvers.lanczos import topk_eigs
from stats.spectra import spectral_gap, spectral_entropy

def main():
    """Demonstrate basic dual transport analysis."""

    print("🚀 EXAMPLE 1: Basic Dual Transport Demonstration")
    print("=" * 60)

    # Ensure reproducibility
    ensure_reproducibility(42)

    # Generate simple synthetic data
    print("\n1. Generating synthetic data...")
    rng = np.random.default_rng(42)

    # Two well-separated Gaussian clusters
    cluster1 = rng.normal(loc=0, scale=1.0, size=(100, 5))
    cluster2 = rng.normal(loc=4, scale=0.8, size=(100, 5))
    X = np.vstack([cluster1, cluster2])

    print(f"   Created dataset: {X.shape[0]} samples, {X.shape[1]} dimensions")
    print(f"   Cluster 1 center: {np.mean(cluster1, axis=0)[:3]}")
    print(f"   Cluster 2 center: {np.mean(cluster2, axis=0)[:3]}")

    # Build graphs with dual transport modes
    print("\n2. Building graphs with dual transport modes...")

    k = 16  # Number of nearest neighbors

    # Additive transport (Gaussian kernel)
    print(f"   📊 Additive transport (k={k})...")
    G_additive = build_graph(X, mode="additive", k=k, sigma="median", seed=42)
    L_additive = laplacian(G_additive, normalized=True)

    # Multiplicative transport (log-space, Haar measure)
    print(f"   📈 Multiplicative transport (k={k})...")
    G_multiplicative = build_graph(X, mode="multiplicative", k=k, tau="median", eps=1e-6, seed=42)
    L_multiplicative = laplacian(G_multiplicative, normalized=True)

    # Compute spectra
    print("\n3. Computing spectral properties...")

    n_eigs = min(20, X.shape[0] - 1)

    # Additive spectrum
    print("   🔍 Additive spectrum...")
    eigenvals_add, eigenvecs_add, info_add = topk_eigs(L_additive, k=n_eigs, which="SM")
    gap_add = spectral_gap(L_additive)
    entropy_add = spectral_entropy(L_additive, k=16)
    print(f"   Info add: converged={info_add.converged}")

    # Multiplicative spectrum
    print("   🔍 Multiplicative spectrum...")
    eigenvals_mult, eigenvecs_mult, info_mult = topk_eigs(L_multiplicative, k=n_eigs, which="SM")
    gap_mult = spectral_gap(L_multiplicative)
    entropy_mult = spectral_entropy(L_multiplicative, k=16)
    print(f"   Info mult: converged={info_mult.converged}")

    # Display results
    print("\n4. Results Comparison:")
    print("   " + "-" * 50)
    print(f"   {'Transport Mode':<20} {'Gap':<12} {'Entropy':<12}")
    print("   " + "-" * 50)
    print(f"   {'Additive':<20} {gap_add:<12.6f} {entropy_add:<12.6f}")
    print(f"   {'Multiplicative':<20} {gap_mult:<12.6f} {entropy_mult:<12.6f}")
    print("   " + "-" * 50)

    # Analysis
    gap_diff = abs(gap_add - gap_mult)
    entropy_diff = abs(entropy_add - entropy_mult)

    print(f"\n5. Analysis:")
    print(f"   📊 Spectral gap difference: {gap_diff:.6f}")
    print(f"   📊 Entropy difference: {entropy_diff:.6f}")

    if gap_diff > 0.01:
        print("   ✅ SIGNIFICANT: Transport modes reveal different geometric structures!")
    else:
        print("   ⚠️  SIMILAR: Transport modes produce similar results on this data")

    # First few eigenvalues comparison
    print(f"\n6. First 5 eigenvalues:")
    print(f"   Additive:      {eigenvals_add[:5]}")
    print(f"   Multiplicative: {eigenvals_mult[:5]}")

    # Visualization (optional, if matplotlib available)
    try:
        create_visualization(X, eigenvals_add, eigenvals_mult, eigenvecs_add, eigenvecs_mult)
    except ImportError:
        print("\n   📊 Matplotlib not available for visualization")

    print("\n✅ Basic dual transport demonstration complete!")
    print("\nNext steps:")
    print("   • Try examples/02_manifold_analysis.py for more complex data")
    print("   • Explore examples/03_topology_geodesics.py for topological analysis")
    print("   • Launch notebooks/geometric_ml_cookbook.ipynb for interactive exploration")

def create_visualization(X, evals_add, evals_mult, evecs_add, evecs_mult):
    """Create comparison visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Data visualization
    ax1 = axes[0, 0]
    ax1.scatter(X[:100, 0], X[:100, 1], alpha=0.6, label='Cluster 1', s=30)
    ax1.scatter(X[100:, 0], X[100:, 1], alpha=0.6, label='Cluster 2', s=30)
    ax1.set_title('Synthetic Data (First 2 Dimensions)')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Eigenvalue comparison
    ax2 = axes[0, 1]
    idx = np.arange(min(15, len(evals_add)))
    ax2.plot(idx, evals_add[:len(idx)], 'o-', label='Additive', linewidth=2, markersize=6)
    ax2.plot(idx, evals_mult[:len(idx)], 's-', label='Multiplicative', linewidth=2, markersize=6)
    ax2.set_title('Eigenvalue Spectra Comparison')
    ax2.set_xlabel('Eigenvalue Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Fiedler vector (additive)
    ax3 = axes[1, 0]
    if len(evecs_add) > 1:
        fiedler_add = evecs_add[:, 1]  # Second eigenvector
        scatter = ax3.scatter(X[:, 0], X[:, 1], c=fiedler_add, cmap='RdBu', s=30, alpha=0.7)
        ax3.set_title('Fiedler Vector (Additive)')
        ax3.set_xlabel('Dimension 1')
        ax3.set_ylabel('Dimension 2')
        plt.colorbar(scatter, ax=ax3)

    # Fiedler vector (multiplicative)
    ax4 = axes[1, 1]
    if len(evecs_mult) > 1:
        fiedler_mult = evecs_mult[:, 1]  # Second eigenvector
        scatter = ax4.scatter(X[:, 0], X[:, 1], c=fiedler_mult, cmap='RdBu', s=30, alpha=0.7)
        ax4.set_title('Fiedler Vector (Multiplicative)')
        ax4.set_xlabel('Dimension 1')
        ax4.set_ylabel('Dimension 2')
        plt.colorbar(scatter, ax=ax4)

    plt.tight_layout()
    plt.suptitle('Dual Transport Analysis Results', fontsize=16, y=1.02)

    # Save and show
    output_path = 'dual_transport_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   📊 Visualization saved to: {output_path}")

    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
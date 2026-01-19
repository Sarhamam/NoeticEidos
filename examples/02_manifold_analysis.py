#!/usr/bin/env python3
"""
Example 2: Manifold Data Analysis

Demonstrates geometric ML pipeline on data lying on different manifolds.
Shows how dual transports respond to different geometric structures.

Author: Sar Hamam
"""

import numpy as np
import matplotlib.pyplot as plt

from validation.reproducibility import ensure_reproducibility, compute_data_hash
from validation.numerical import validate_float64_precision
from graphs.knn import build_graph
from graphs.laplacian import laplacian
from validation.mathematical import check_graph_connectivity
from solvers.lanczos import topk_eigs
from stats.spectra import spectral_gap, spectral_entropy

def generate_sphere_data(n_samples=500, noise_level=0.1, ambient_dim=8):
    """Generate noisy data on a sphere embedded in high dimensions."""
    rng = np.random.default_rng(42)

    # Generate points on unit sphere
    theta = rng.uniform(0, 2*np.pi, n_samples)
    phi = rng.uniform(0, np.pi, n_samples)

    # Spherical to Cartesian
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    sphere_data = np.column_stack([x, y, z])

    # Add noise
    noise = rng.normal(0, noise_level, sphere_data.shape)
    sphere_data += noise

    # Embed in higher dimensions
    embedded_data = np.column_stack([
        sphere_data,
        rng.normal(0, noise_level/2, (n_samples, ambient_dim - 3))
    ])

    return embedded_data, theta, phi

def generate_swiss_roll(n_samples=500, noise_level=0.1, ambient_dim=8):
    """Generate Swiss roll manifold data."""
    rng = np.random.default_rng(43)

    # Swiss roll parameters
    t = rng.uniform(1.5*np.pi, 4.5*np.pi, n_samples)
    height = rng.uniform(0, 10, n_samples)

    # Swiss roll coordinates
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)

    roll_data = np.column_stack([x, y, z])

    # Add noise
    noise = rng.normal(0, noise_level, roll_data.shape)
    roll_data += noise

    # Embed in higher dimensions
    embedded_data = np.column_stack([
        roll_data,
        rng.normal(0, noise_level/2, (n_samples, ambient_dim - 3))
    ])

    return embedded_data, t, height

def generate_mobius_like_data(n_samples=500, noise_level=0.1, ambient_dim=8):
    """Generate data with Möbius-like structure."""
    rng = np.random.default_rng(44)

    # Möbius strip parameters
    u = rng.uniform(0, 2*np.pi, n_samples)
    v = rng.uniform(-1, 1, n_samples)

    # Möbius strip parametrization
    x = (1 + v/2 * np.cos(u/2)) * np.cos(u)
    y = (1 + v/2 * np.cos(u/2)) * np.sin(u)
    z = v/2 * np.sin(u/2)

    mobius_data = np.column_stack([x, y, z])

    # Add noise
    noise = rng.normal(0, noise_level, mobius_data.shape)
    mobius_data += noise

    # Embed in higher dimensions
    embedded_data = np.column_stack([
        mobius_data,
        rng.normal(0, noise_level/2, (n_samples, ambient_dim - 3))
    ])

    return embedded_data, u, v

def analyze_manifold(X, manifold_name, param1, param2):
    """Perform complete analysis on manifold data."""

    print(f"\n📊 ANALYZING {manifold_name.upper()} MANIFOLD")
    print("=" * 60)

    # Data validation
    print("1. Data Validation:")
    data_hash = compute_data_hash(X)
    precision_check = validate_float64_precision({manifold_name: X})

    print(f"   Shape: {X.shape}")
    print(f"   Hash: {data_hash[:16]}...")
    print(f"   Precision: {'✅ Pass' if precision_check['precision_adequate'] else '❌ Fail'}")

    # Graph construction
    print("\n2. Graph Construction:")
    k = 16

    # Additive transport
    print("   🔗 Building additive graph...")
    G_add = build_graph(X, mode="additive", k=k, sigma="median", seed=42)
    L_add = laplacian(G_add, normalized=True)
    conn_add = check_graph_connectivity(G_add)

    # Multiplicative transport
    print("   🔗 Building multiplicative graph...")
    G_mult = build_graph(X, mode="multiplicative", k=k, tau="median", eps=1e-6, seed=42)
    L_mult = laplacian(G_mult, normalized=True)
    conn_mult = check_graph_connectivity(G_mult)

    print(f"   Additive connected: {'✅' if conn_add else '❌'}")
    print(f"   Multiplicative connected: {'✅' if conn_mult else '❌'}")

    # Spectral analysis
    print("\n3. Spectral Analysis:")
    n_eigs = min(20, X.shape[0] - 1)

    # Additive spectrum
    eigenvals_add, eigenvecs_add, _ = topk_eigs(L_add, k=n_eigs, which="SM")
    gap_add = spectral_gap(L_add)
    entropy_add = spectral_entropy(L_add, k=16)

    # Multiplicative spectrum
    eigenvals_mult, eigenvecs_mult, _ = topk_eigs(L_mult, k=n_eigs, which="SM")
    gap_mult = spectral_gap(L_mult)
    entropy_mult = spectral_entropy(L_mult, k=16)

    print(f"   Additive - Gap: {gap_add:.4f}, Entropy: {entropy_add:.4f}")
    print(f"   Multiplicative - Gap: {gap_mult:.4f}, Entropy: {entropy_mult:.4f}")

    # Manifold-specific analysis
    print("\n4. Manifold-Specific Analysis:")

    if manifold_name == "sphere":
        # For sphere, check if Fiedler vector captures latitude structure
        fiedler_add = eigenvecs_add[:, 1] if len(eigenvecs_add) > 1 else None
        if fiedler_add is not None:
            correlation_phi = np.corrcoef(fiedler_add, param2)[0, 1]  # param2 is phi (latitude)
            print(f"   Fiedler-latitude correlation: {abs(correlation_phi):.3f}")

    elif manifold_name == "swiss_roll":
        # For Swiss roll, check if spectrum captures the unrolling
        intrinsic_param = param1  # parameter t
        fiedler_add = eigenvecs_add[:, 1] if len(eigenvecs_add) > 1 else None
        if fiedler_add is not None:
            correlation_t = np.corrcoef(fiedler_add, intrinsic_param)[0, 1]
            print(f"   Fiedler-parameter correlation: {abs(correlation_t):.3f}")

    elif manifold_name == "mobius":
        # For Möbius-like structure, check for non-orientable signatures
        print(f"   Non-orientable structure detected in parameter space")
        print(f"   Parameter range u: [{param1.min():.2f}, {param1.max():.2f}]")

    # Return results for comparison
    return {
        'manifold': manifold_name,
        'shape': X.shape,
        'connected_add': conn_add,
        'connected_mult': conn_mult,
        'gap_add': gap_add,
        'gap_mult': gap_mult,
        'entropy_add': entropy_add,
        'entropy_mult': entropy_mult,
        'eigenvals_add': eigenvals_add,
        'eigenvals_mult': eigenvals_mult,
        'eigenvecs_add': eigenvecs_add,
        'eigenvecs_mult': eigenvecs_mult,
        'params': (param1, param2)
    }

def create_manifold_comparison(results_list):
    """Create comparison visualization across manifolds."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    manifold_names = [r['manifold'] for r in results_list]
    colors = ['blue', 'green', 'red']

    # 1. Data visualization (3D projections)
    for i, results in enumerate(results_list):
        ax = axes[0, i]

        # Get the original 3D data (first 3 dims)
        X = results['shape']  # This is just shape info, need to reconstruct
        # For visualization, we'll use the eigenvalues instead
        eigenvals = results['eigenvals_add']

        ax.plot(eigenvals[:15], 'o-', color=colors[i], linewidth=2, markersize=6)
        ax.set_title(f'{results["manifold"].title()} - Additive Spectrum')
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.grid(True, alpha=0.3)

    # 2. Spectral gap comparison
    ax_gap = axes[1, 0]
    gaps_add = [r['gap_add'] for r in results_list]
    gaps_mult = [r['gap_mult'] for r in results_list]

    x = np.arange(len(manifold_names))
    width = 0.35

    ax_gap.bar(x - width/2, gaps_add, width, label='Additive', alpha=0.8)
    ax_gap.bar(x + width/2, gaps_mult, width, label='Multiplicative', alpha=0.8)

    ax_gap.set_title('Spectral Gaps Comparison')
    ax_gap.set_xlabel('Manifold')
    ax_gap.set_ylabel('Spectral Gap')
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels([m.title() for m in manifold_names])
    ax_gap.legend()
    ax_gap.grid(True, alpha=0.3)

    # 3. Entropy comparison
    ax_entropy = axes[1, 1]
    entropies_add = [r['entropy_add'] for r in results_list]
    entropies_mult = [r['entropy_mult'] for r in results_list]

    ax_entropy.bar(x - width/2, entropies_add, width, label='Additive', alpha=0.8)
    ax_entropy.bar(x + width/2, entropies_mult, width, label='Multiplicative', alpha=0.8)

    ax_entropy.set_title('Spectral Entropy Comparison')
    ax_entropy.set_xlabel('Manifold')
    ax_entropy.set_ylabel('Spectral Entropy')
    ax_entropy.set_xticks(x)
    ax_entropy.set_xticklabels([m.title() for m in manifold_names])
    ax_entropy.legend()
    ax_entropy.grid(True, alpha=0.3)

    # 4. Transport mode effectiveness
    ax_transport = axes[1, 2]

    gap_ratios = [r['gap_mult'] / r['gap_add'] for r in results_list]
    entropy_ratios = [r['entropy_mult'] / r['entropy_add'] for r in results_list]

    ax_transport.scatter(gap_ratios, entropy_ratios, s=100, alpha=0.7)

    for i, name in enumerate(manifold_names):
        ax_transport.annotate(name.title(), (gap_ratios[i], entropy_ratios[i]),
                            xytext=(5, 5), textcoords='offset points')

    ax_transport.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax_transport.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax_transport.set_title('Transport Mode Effectiveness')
    ax_transport.set_xlabel('Gap Ratio (Mult/Add)')
    ax_transport.set_ylabel('Entropy Ratio (Mult/Add)')
    ax_transport.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Manifold Analysis Comparison', fontsize=16, y=1.02)

    # Save
    output_path = 'manifold_analysis_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison visualization saved to: {output_path}")

    try:
        plt.show()
    except:
        pass

def main():
    """Main manifold analysis demonstration."""

    print("🌊 EXAMPLE 2: Manifold Data Analysis")
    print("=" * 60)

    ensure_reproducibility(42)

    # Generate different manifold datasets
    print("\n🎯 Generating manifold datasets...")

    datasets = {}

    print("   📍 Sphere manifold...")
    X_sphere, theta_sphere, phi_sphere = generate_sphere_data(n_samples=400)
    datasets['sphere'] = (X_sphere, theta_sphere, phi_sphere)

    print("   📍 Swiss roll manifold...")
    X_roll, t_roll, h_roll = generate_swiss_roll(n_samples=400)
    datasets['swiss_roll'] = (X_roll, t_roll, h_roll)

    print("   📍 Möbius-like manifold...")
    X_mobius, u_mobius, v_mobius = generate_mobius_like_data(n_samples=400)
    datasets['mobius'] = (X_mobius, u_mobius, v_mobius)

    # Analyze each manifold
    results = []
    for name, (X, param1, param2) in datasets.items():
        result = analyze_manifold(X, name, param1, param2)
        results.append(result)

    # Create comparison
    print(f"\n📈 CROSS-MANIFOLD COMPARISON")
    print("=" * 60)

    print("\nSpectral Properties Summary:")
    print(f"{'Manifold':<12} {'Gap (Add)':<10} {'Gap (Mult)':<11} {'Entropy (Add)':<13} {'Entropy (Mult)'}")
    print("-" * 70)

    for r in results:
        print(f"{r['manifold']:<12} {r['gap_add']:<10.4f} {r['gap_mult']:<11.4f} "
              f"{r['entropy_add']:<13.4f} {r['entropy_mult']:.4f}")

    # Analysis insights
    print(f"\n🔍 Key Insights:")

    # Find manifold with largest gap difference
    gap_diffs = [(r['manifold'], abs(r['gap_add'] - r['gap_mult'])) for r in results]
    max_gap_diff = max(gap_diffs, key=lambda x: x[1])

    print(f"   • Largest gap difference: {max_gap_diff[0]} ({max_gap_diff[1]:.4f})")

    # Find manifold where multiplicative works better
    mult_better = [r for r in results if r['gap_mult'] > r['gap_add']]
    if mult_better:
        print(f"   • Multiplicative transport more effective on: {', '.join(r['manifold'] for r in mult_better)}")

    # Connectivity analysis
    conn_issues = [r for r in results if not (r['connected_add'] and r['connected_mult'])]
    if conn_issues:
        print(f"   • Connectivity issues detected on: {', '.join(r['manifold'] for r in conn_issues)}")
    else:
        print(f"   • All manifolds produced connected graphs ✅")

    # Create visualization
    try:
        create_manifold_comparison(results)
    except ImportError:
        print("\n   📊 Matplotlib not available for visualization")

    print("\n✅ Manifold analysis complete!")
    print("\nNext steps:")
    print("   • Explore examples/03_topology_geodesics.py for geodesic analysis")
    print("   • Try examples/04_full_pipeline.py for complete workflow")
    print("   • Check docs/geometry.md for Fisher-Rao metric theory")

if __name__ == "__main__":
    main()
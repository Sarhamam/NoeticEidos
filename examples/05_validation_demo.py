#!/usr/bin/env python3
"""
Example 5: Validation Framework Demonstration

Shows the comprehensive validation framework including:
- Reproducibility controls
- Numerical precision checks
- Mathematical property validation
- Performance monitoring
- Stability testing

Author: Sar Hamam
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from validation.reproducibility import ensure_reproducibility, compute_data_hash
from validation.numerical import validate_float64_precision, check_matrix_conditioning
from validation.mathematical import check_graph_connectivity
from graphs.knn import build_graph
from graphs.laplacian import laplacian
from solvers.lanczos import topk_eigs
from stats.spectra import spectral_gap
from topology import create_topology, TopologyType

def demonstrate_reproducibility():
    """Demonstrate reproducibility validation."""

    print("REPRODUCIBILITY VALIDATION")
    print("=" * 50)

    # Test seed consistency
    print("\n1. Seed Consistency:")
    seeds = [42, 123, 999]

    for seed in seeds:
        print(f"   Testing seed {seed}:")

        # Run operation multiple times with same seed
        results = []
        for trial in range(3):
            ensure_reproducibility(seed)
            rng = np.random.default_rng(seed)
            data = rng.normal(0, 1, (100, 5))
            data_hash = compute_data_hash(data)
            results.append(data_hash)

        # Check consistency
        all_same = len(set(results)) == 1
        print(f"      Hash consistency: {'Pass' if all_same else 'Fail'}")
        print(f"      Hashes: {[h[:8] + '...' for h in results]}")

    # Validate seed behavior (manual check)
    print(f"\n2. Cross-Seed Validation:")

    # Test that different seeds produce different results
    hash_42 = compute_data_hash(np.random.default_rng(42).normal(0, 1, (50, 5)))
    hash_43 = compute_data_hash(np.random.default_rng(43).normal(0, 1, (50, 5)))

    different_outputs = hash_42 != hash_43
    same_outputs = hash_42 == compute_data_hash(np.random.default_rng(42).normal(0, 1, (50, 5)))

    print(f"   Different seeds produce different results: {'Pass' if different_outputs else 'Fail'}")
    print(f"   Same seed produces same results: {'Pass' if same_outputs else 'Fail'}")

def demonstrate_numerical_validation():
    """Demonstrate numerical precision validation."""

    print(f"\nNUMERICAL PRECISION VALIDATION")
    print("=" * 50)

    # Generate test data
    ensure_reproducibility(42)
    rng = np.random.default_rng(42)

    # Test matrices with different properties
    test_matrices = {
        'random': rng.normal(0, 1, (100, 100)),
        'symmetric': None,
        'positive_definite': None,
    }

    # Create symmetric matrix
    A = rng.normal(0, 1, (50, 50))
    test_matrices['symmetric'] = A + A.T

    # Create positive definite matrix
    B = rng.normal(0, 1, (50, 50))
    test_matrices['positive_definite'] = B.T @ B + 0.1 * np.eye(50)

    print(f"\n1. Float64 Precision Check:")
    precision_results = validate_float64_precision(test_matrices)

    print(f"   Overall precision adequate: {'Pass' if precision_results['precision_adequate'] else 'Fail'}")
    for name, dtype in precision_results['dtypes'].items():
        print(f"   {name}: dtype={dtype}")

    print(f"\n2. Matrix Conditioning:")
    for name, matrix in test_matrices.items():
        try:
            cond_result = check_matrix_conditioning(matrix, condition_threshold=1e12)
            status = "Pass" if cond_result['well_conditioned'] else "Warn"
            print(f"   {name}: {status} (condition={cond_result['condition_number']:.2e})")
        except Exception as e:
            print(f"   {name}: Error - {str(e)[:30]}...")

def demonstrate_mathematical_validation():
    """Demonstrate mathematical property validation."""

    print(f"\nMATHEMATICAL VALIDATION")
    print("=" * 50)

    # Generate test data
    ensure_reproducibility(42)
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 8))

    print(f"1. Graph Construction Validation:")

    # Build graph
    G = build_graph(X, mode="additive", k=16, seed=42)

    # Connectivity check
    connectivity = check_graph_connectivity(G, require_connected=False)
    print(f"   Graph connectivity: {'Connected' if connectivity else 'Disconnected'}")

    # Build Laplacian
    L = laplacian(G, normalized=True)

    # Check Laplacian properties manually
    print(f"\n2. Laplacian Properties:")
    is_symmetric = np.allclose(L.toarray(), L.toarray().T, atol=1e-10)
    print(f"   Symmetric: {'Pass' if is_symmetric else 'Fail'}")

    # Spectrum validation
    eigenvals, _, _ = topk_eigs(L, k=10, which="SM")

    print(f"\n3. Spectral Properties:")
    print(f"   First eigenvalue ~ 0: {'Pass' if abs(eigenvals[0]) < 1e-6 else 'Fail'} (value={eigenvals[0]:.2e})")
    print(f"   Non-negative spectrum: {'Pass' if np.all(eigenvals >= -1e-10) else 'Fail'}")
    print(f"   Spectral gap: {spectral_gap(L):.4f}")

    # Metric validation
    print(f"\n4. Metric Validation:")

    def test_metric(q):
        u, v = q[0], q[1]
        g11 = 1.2 + 0.3 * np.cos(2*v)
        g22 = 0.9 + 0.2 * np.cos(4*v)
        g12 = 0.1 * np.sin(2*u) * np.sin(2*v)
        return np.array([[g11, g12], [g12, g22]])

    test_points = [
        np.array([0.5, 0.3]),
        np.array([1.0, -0.2]),
        np.array([1.8, 0.7])
    ]

    for i, q in enumerate(test_points):
        g = test_metric(q)
        eigenvals_g = np.linalg.eigvalsh(g)
        is_pd = np.all(eigenvals_g > 0)
        print(f"   Point {i+1}: {'PD' if is_pd else 'Not PD'} (min_eig={eigenvals_g.min():.4f})")

def demonstrate_stability_testing():
    """Demonstrate stability testing across multiple runs."""

    print(f"\nSTABILITY TESTING")
    print("=" * 50)

    print(f"1. Multi-seed Stability:")

    seeds = [42, 43, 44, 45, 46]
    results = {
        'spectral_gaps': [],
        'connectivity': [],
        'computation_times': []
    }

    for seed in seeds:
        print(f"   Testing seed {seed}...")

        start_time = time.time()

        ensure_reproducibility(seed)

        # Generate data
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (300, 6))

        # Build graph and compute spectrum
        G = build_graph(X, mode="additive", k=16, seed=seed)
        L = laplacian(G, normalized=True)

        # Record results
        gap = spectral_gap(L)
        connectivity = check_graph_connectivity(G, require_connected=False)
        comp_time = time.time() - start_time

        results['spectral_gaps'].append(gap)
        results['connectivity'].append(connectivity)
        results['computation_times'].append(comp_time)

        print(f"      Gap: {gap:.4f}, Connected: {'Yes' if connectivity else 'No'}, Time: {comp_time:.2f}s")

    # Stability analysis
    print(f"\n2. Stability Analysis:")

    gaps = results['spectral_gaps']
    times = results['computation_times']

    gap_mean = np.mean(gaps)
    gap_std = np.std(gaps)
    gap_cv = gap_std / gap_mean if gap_mean > 0 else float('inf')

    time_mean = np.mean(times)
    time_std = np.std(times)

    all_connected = all(results['connectivity'])

    print(f"   Spectral gap stability:")
    print(f"      Mean: {gap_mean:.4f} +/- {gap_std:.4f}")
    print(f"      CV: {gap_cv:.2%}")
    print(f"      Stable: {'Pass' if gap_cv < 0.1 else 'Warn'} (CV < 10%)")

    print(f"   Timing stability:")
    print(f"      Mean: {time_mean:.2f}s +/- {time_std:.2f}s")

    print(f"   Connectivity: {'All connected' if all_connected else 'Some disconnected'}")

def demonstrate_topology_validation():
    """Demonstrate topology-specific validation."""

    print(f"\nTOPOLOGY VALIDATION")
    print("=" * 50)

    # Create test metric
    def test_metric(q):
        u, v = q[0], q[1]
        g11 = 1.2 + 0.3 * np.cos(2*v)  # Even in v
        g22 = 0.9 + 0.2 * np.cos(4*v)  # Even in v
        g12 = 0.1 * np.sin(2*u) * np.sin(2*v)  # Odd in v
        return np.array([[g11, g12], [g12, g22]])

    # Test different topologies
    topologies = {
        'mobius': create_topology(TopologyType.MOBIUS, w=1.0),
        'cylinder': create_topology(TopologyType.CYLINDER, w=1.0),
    }

    print(f"1. Seam Compatibility:")

    test_points = [
        np.array([0.5, 0.3]),
        np.array([1.0, -0.2]),
        np.array([2.0, 0.7])
    ]

    for topo_name, topology in topologies.items():
        print(f"\n   {topo_name.title()} topology:")
        print(f"      Orientable: {topology.orientability.value}")

        compatible_count = 0
        for i, q in enumerate(test_points):
            try:
                compatible = topology.metric_compatibility_condition(test_metric, q)
                status = "Pass" if compatible else "Fail"
                print(f"      Point {i+1}: {status}")
                if compatible:
                    compatible_count += 1
            except Exception as e:
                print(f"      Point {i+1}: Error - {str(e)[:30]}...")

        compatibility_rate = compatible_count / len(test_points)
        print(f"      Overall: {compatibility_rate:.1%} compatible")

    # Comprehensive topology validation
    print(f"\n2. Comprehensive Validation:")

    try:
        from topology import comprehensive_topology_validation
        from topology.coords import Strip

        strip = Strip(w=1.0, period=2*np.pi)

        validation = comprehensive_topology_validation(
            g_fn=test_metric,
            strip=strip,
            tolerance=1e-6
        )

        print(f"   Overall result: {'Pass' if validation['all_passed'] else 'Fail'}")
        print(f"   Tests run: {', '.join(validation['tests_run'])}")

    except Exception as e:
        print(f"   Comprehensive validation info: {str(e)[:50]}...")

def main():
    """Run complete validation demonstration."""

    print("EXAMPLE 5: Validation Framework Demonstration")
    print("=" * 60)
    print("This demonstrates the comprehensive validation capabilities:")
    print("Reproducibility | Numerical precision | Mathematical properties")
    print("Performance monitoring | Stability testing | Topology validation")
    print("=" * 60)

    try:
        # Run all validation demonstrations
        demonstrate_reproducibility()
        demonstrate_numerical_validation()
        demonstrate_mathematical_validation()
        demonstrate_stability_testing()
        demonstrate_topology_validation()

        print(f"\nVALIDATION DEMONSTRATION COMPLETE")
        print("=" * 60)

        print(f"\nKey Validation Principles:")
        print("   - Always ensure reproducibility with seed control")
        print("   - Validate numerical precision and matrix properties")
        print("   - Check mathematical invariants (symmetry, PSD, connectivity)")
        print("   - Monitor performance and memory usage")
        print("   - Test stability across multiple seeds/trials")
        print("   - Verify topology-specific seam compatibility")

    except Exception as e:
        print(f"\nValidation demonstration failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nRelated Resources:")
    print("   - docs/validation_framework.md - Complete validation guide")
    print("   - examples/04_full_pipeline.py - Integrated validation")
    print("   - tests/ - Unit tests with validation examples")

if __name__ == "__main__":
    main()
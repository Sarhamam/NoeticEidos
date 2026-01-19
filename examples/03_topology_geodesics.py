#!/usr/bin/env python3
"""
Example 3: Topology and Geodesic Analysis

Demonstrates the topology module with different quotient spaces,
seam-compatible metrics, and geodesic integration on non-trivial topologies.

Author: Sar Hamam
"""

import numpy as np
import matplotlib.pyplot as plt

from validation.reproducibility import ensure_reproducibility
from topology import (
    create_topology, TopologyType, get_orientability_pairs,
    integrate_geodesic, comprehensive_topology_validation
)
from topology.atlas import topology_atlas

def create_seam_compatible_metric(alpha=0.7):
    """Create a seam-compatible metric for Möbius band."""

    def metric_fn(q):
        u, v = q[0], q[1]

        # Seam-compatible components for Möbius band
        # Requirements: g11, g22 even in v; g12 odd in v
        g11 = 1.2 + alpha * np.exp(-0.1 * (u**2 + v**2)) + 0.3 * np.cos(2*v)  # Even in v
        g22 = 0.9 + alpha * np.exp(-0.15 * (u**2 + v**2)) + 0.2 * np.cos(4*v)  # Even in v
        g12 = 0.15 * alpha * np.sin(2*u) * np.sin(2*v)  # Odd in v

        return np.array([[g11, g12], [g12, g22]])

    def metric_grad_fn(q):
        """Compute metric gradient via finite differences."""
        h = 1e-6
        g_base = metric_fn(q)

        # ∂g/∂u
        g_u_plus = metric_fn(q + np.array([h, 0]))
        du_g = (g_u_plus - g_base) / h

        # ∂g/∂v
        g_v_plus = metric_fn(q + np.array([0, h]))
        dv_g = (g_v_plus - g_base) / h

        return du_g, dv_g

    return metric_fn, metric_grad_fn

def demonstrate_topology_atlas():
    """Demonstrate the topology atlas capabilities."""

    print("\n🌐 TOPOLOGY ATLAS DEMONSTRATION")
    print("=" * 60)

    print("\n1. Available Topologies:")
    topologies = {}

    # Create all topology types
    topology_configs = {
        'cylinder': {'w': 1.0},
        'mobius': {'w': 1.0},
        'torus': {'width': 2*np.pi, 'height': 2*np.pi},
        'klein': {'width': 2*np.pi, 'height': 2*np.pi},
        'sphere': {'radius': 1.0},
        'projective': {}
    }

    for name, config in topology_configs.items():
        try:
            topo_type = getattr(TopologyType, name.upper())
            topo = create_topology(topo_type, **config)
            topologies[name] = topo

            print(f"   ✅ {name.title()}: orientable={topo.orientability.value}, χ={topo.euler_characteristic}")
        except Exception as e:
            print(f"   ❌ {name.title()}: Error - {str(e)[:50]}...")

    # Topology comparison matrix
    print(f"\n2. Topology Comparison Matrix:")
    try:
        matrix = topology_atlas.topology_comparison_matrix()

        print(f"{'Topology':<15} {'Orientable':<11} {'Euler χ':<8} {'Genus':<6} {'# ID Maps'}")
        print("-" * 60)

        for topo_type, props in matrix.items():
            orientable = "Yes" if props['orientable'] else "No"
            print(f"{topo_type.value:<15} {orientable:<11} {props['euler_char']:<8} "
                  f"{props['genus']:<6} {props['num_identifications']}")

    except Exception as e:
        print(f"   ⚠️  Matrix generation failed: {e}")

    # Orientability pairs
    print(f"\n3. Orientable/Non-Orientable Pairs:")
    try:
        pairs = get_orientability_pairs()
        for i, (orientable_topo, non_orientable_topo) in enumerate(pairs):
            print(f"   Pair {i+1}: {type(orientable_topo).__name__} ↔ {type(non_orientable_topo).__name__}")
    except Exception as e:
        print(f"   ⚠️  Pairs generation failed: {e}")

    return topologies

def analyze_metric_compatibility(metric_fn, topologies):
    """Analyze metric compatibility across different topologies."""

    print("\n📐 METRIC COMPATIBILITY ANALYSIS")
    print("=" * 60)

    test_points = [
        np.array([0.3, 0.2]),
        np.array([1.0, 0.5]),
        np.array([1.8, -0.3])
    ]

    compatibility_results = {}

    for name, topology in topologies.items():
        print(f"\n   Testing {name.title()} topology:")

        try:
            # Use topology atlas validation
            validation = topology_atlas.validate_metric_on_topology(metric_fn, topology, test_points)

            compatible_rate = validation.get('compatibility_rate', 0.0)
            total_points = validation.get('n_test_points', 0)

            print(f"      Compatibility: {compatible_rate:.1%} ({total_points} points)")

            if 'errors' in validation and validation['errors']:
                print(f"      Issues: {len(validation['errors'])} errors detected")

            compatibility_results[name] = validation

        except Exception as e:
            print(f"      ❌ Validation failed: {str(e)[:50]}...")
            compatibility_results[name] = {'error': str(e)}

    return compatibility_results

def demonstrate_geodesic_integration(metric_fn, metric_grad_fn):
    """Demonstrate geodesic integration on Möbius band."""

    print("\n🌊 GEODESIC INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Create Möbius band
    mobius = create_topology(TopologyType.MOBIUS, w=1.2, period=2*np.pi)

    print(f"1. Möbius Band Setup:")
    print(f"   Width: {mobius.w}")
    print(f"   Period: {mobius.period:.2f}")
    print(f"   Orientable: {mobius.orientability.value}")

    # Comprehensive validation
    print(f"\n2. Metric Validation:")
    try:
        from topology.coords import Strip
        strip = Strip(w=mobius.w, period=mobius.period)

        validation = comprehensive_topology_validation(
            g_fn=metric_fn,
            strip=strip,
            tolerance=1e-6
        )

        print(f"   Overall validation: {'✅ Pass' if validation['all_passed'] else '❌ Fail'}")

        for test_name in validation.get('tests_run', []):
            if test_name in validation:
                result = validation[test_name]
                if isinstance(result, dict) and 'compatible' in result:
                    status = "✅" if result['compatible'] else "❌"
                    print(f"   {status} {test_name}")

    except Exception as e:
        print(f"   ⚠️  Validation error: {str(e)[:50]}...")
        strip = None

    # Geodesic integration
    print(f"\n3. Geodesic Integration:")

    if strip is not None:
        geodesic_results = []

        # Multiple geodesics with different initial conditions
        initial_conditions = [
            (np.array([0.2, 0.4]), np.array([0.5, 0.3])),
            (np.array([1.0, -0.2]), np.array([0.3, 0.7])),
            (np.array([2.5, 0.6]), np.array([-0.4, 0.2])),
            (np.array([0.8, -0.8]), np.array([0.6, -0.1])),
        ]

        for i, (q0, v0) in enumerate(initial_conditions):
            print(f"\n   Geodesic {i+1}: q0={q0}, v0={v0}")

            try:
                traj_q, traj_v, info = integrate_geodesic(
                    q0, v0,
                    t_final=6.0,
                    dt=0.01,
                    g_fn=metric_fn,
                    grad_g_fn=metric_grad_fn,
                    strip=strip,
                    energy_tolerance=1e-2
                )

                if info['success']:
                    print(f"      ✅ Success: {len(traj_q)} steps")
                    print(f"      Energy drift: {info['energy_drift']:.2e}")
                    print(f"      Seam crossings: {info['seam_crossings']}")
                    print(f"      Trajectory length: {info['trajectory_length']:.2f}")

                    geodesic_results.append({
                        'index': i,
                        'initial_q': q0,
                        'initial_v': v0,
                        'trajectory_q': traj_q,
                        'trajectory_v': traj_v,
                        'info': info
                    })

                else:
                    print(f"      ❌ Failed: Integration unsuccessful")

            except Exception as e:
                print(f"      ⚠️  Error: {str(e)[:50]}...")

        # Analysis of geodesic results
        if geodesic_results:
            print(f"\n4. Geodesic Analysis Summary:")
            print(f"   Successful integrations: {len(geodesic_results)}")

            energy_drifts = [r['info']['energy_drift'] for r in geodesic_results]
            seam_crossings = [r['info']['seam_crossings'] for r in geodesic_results]
            trajectory_lengths = [r['info']['trajectory_length'] for r in geodesic_results]

            print(f"   Mean energy drift: {np.mean(energy_drifts):.2e}")
            print(f"   Total seam crossings: {sum(seam_crossings)}")
            print(f"   Mean trajectory length: {np.mean(trajectory_lengths):.2f}")

            return geodesic_results

    return []

def create_geodesic_visualization(geodesic_results, strip):
    """Create visualization of geodesic trajectories."""

    if not geodesic_results:
        print("   📊 No geodesic results to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. All trajectories on fundamental domain
    ax1 = axes[0, 0]

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, result in enumerate(geodesic_results[:6]):
        traj_q = result['trajectory_q']
        q0 = result['initial_q']

        color = colors[i % len(colors)]
        ax1.plot(traj_q[:, 0], traj_q[:, 1], '-', color=color, alpha=0.7,
                linewidth=2, label=f'Geodesic {i+1}')
        ax1.scatter(q0[0], q0[1], color=color, s=80, marker='o', zorder=5)
        ax1.scatter(traj_q[-1, 0], traj_q[-1, 1], color=color, s=80, marker='s', zorder=5)

    # Draw seam boundaries
    ax1.axhline(y=strip.w, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Seam boundaries')
    ax1.axhline(y=-strip.w, color='black', linestyle='--', alpha=0.7, linewidth=2)

    ax1.set_title('Geodesic Trajectories on Möbius Band')
    ax1.set_xlabel('u')
    ax1.set_ylabel('v')
    ax1.set_xlim(0, strip.period)
    ax1.set_ylim(-strip.w*1.1, strip.w*1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Energy conservation
    ax2 = axes[0, 1]

    for i, result in enumerate(geodesic_results[:4]):
        info = result['info']
        if 'energy_array' in info and 'time_array' in info:
            energy_normalized = info['energy_array'] / info['initial_energy']
            color = colors[i % len(colors)]
            ax2.plot(info['time_array'], energy_normalized, '-', color=color,
                    linewidth=2, alpha=0.8, label=f'Geodesic {i+1}')

    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect conservation')
    ax2.set_title('Energy Conservation')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('E(t) / E(0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Velocity phase space
    ax3 = axes[1, 0]

    for i, result in enumerate(geodesic_results[:4]):
        traj_v = result['trajectory_v']
        v0 = result['initial_v']
        color = colors[i % len(colors)]

        ax3.plot(traj_v[:, 0], traj_v[:, 1], '-', color=color, alpha=0.7,
                linewidth=2, label=f'Geodesic {i+1}')
        ax3.scatter(v0[0], v0[1], color=color, s=80, marker='o', zorder=5)

    ax3.set_title('Velocity Phase Space')
    ax3.set_xlabel('du/dt')
    ax3.set_ylabel('dv/dt')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Seam crossing analysis
    ax4 = axes[1, 1]

    seam_crossings = [r['info']['seam_crossings'] for r in geodesic_results]
    energy_drifts = [r['info']['energy_drift'] for r in geodesic_results]

    ax4.scatter(seam_crossings, energy_drifts, s=100, alpha=0.7, color='purple')

    for i, (crossings, drift) in enumerate(zip(seam_crossings, energy_drifts)):
        ax4.annotate(f'G{i+1}', (crossings, drift), xytext=(5, 5),
                    textcoords='offset points', fontsize=10)

    ax4.set_title('Seam Crossings vs Energy Drift')
    ax4.set_xlabel('Number of Seam Crossings')
    ax4.set_ylabel('Energy Drift')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Geodesic Dynamics on Möbius Band', fontsize=16, y=1.02)

    # Save
    output_path = 'geodesic_dynamics_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   📊 Geodesic visualization saved to: {output_path}")

    try:
        plt.show()
    except:
        pass

def main():
    """Main topology and geodesic demonstration."""

    print("🌐 EXAMPLE 3: Topology and Geodesic Analysis")
    print("=" * 60)

    ensure_reproducibility(42)

    # Demonstrate topology atlas
    topologies = demonstrate_topology_atlas()

    # Create seam-compatible metric
    print(f"\n📐 Creating seam-compatible metric...")
    metric_fn, metric_grad_fn = create_seam_compatible_metric(alpha=0.8)

    # Test metric properties
    q_test = np.array([0.5, 0.3])
    g_test = metric_fn(q_test)
    det_g = np.linalg.det(g_test)
    eigenvals_g = np.linalg.eigvals(g_test)

    print(f"   Test point: {q_test}")
    print(f"   Determinant: {det_g:.4f}")
    print(f"   Eigenvalues: [{eigenvals_g[0]:.3f}, {eigenvals_g[1]:.3f}]")
    print(f"   Positive definite: {'✅' if np.all(eigenvals_g > 1e-12) else '❌'}")

    # Analyze metric compatibility
    compatibility_results = analyze_metric_compatibility(metric_fn, topologies)

    # Demonstrate geodesic integration
    geodesic_results = demonstrate_geodesic_integration(metric_fn, metric_grad_fn)

    # Create visualization
    if geodesic_results:
        try:
            from topology.coords import Strip
            strip = Strip(w=1.2, period=2*np.pi)
            create_geodesic_visualization(geodesic_results, strip)
        except ImportError:
            print("\n   📊 Matplotlib not available for visualization")

    print(f"\n🎯 SUMMARY")
    print("=" * 60)

    # Topology summary
    working_topologies = len([name for name, result in compatibility_results.items()
                            if 'error' not in result])
    print(f"   Topologies tested: {len(topologies)}")
    print(f"   Working with metric: {working_topologies}")

    # Geodesic summary
    if geodesic_results:
        successful_geodesics = len(geodesic_results)
        print(f"   Successful geodesics: {successful_geodesics}")

        if successful_geodesics > 0:
            total_crossings = sum(r['info']['seam_crossings'] for r in geodesic_results)
            mean_drift = np.mean([r['info']['energy_drift'] for r in geodesic_results])
            print(f"   Total seam crossings: {total_crossings}")
            print(f"   Mean energy drift: {mean_drift:.2e}")

    print(f"\n✅ Topology and geodesic analysis complete!")
    print("\nNext steps:")
    print("   • Try examples/04_full_pipeline.py for complete workflow")
    print("   • Explore docs/topology_atlas.md for mathematical foundations")
    print("   • Check notebooks/geometric_ml_cookbook.ipynb for interactive exploration")

if __name__ == "__main__":
    main()
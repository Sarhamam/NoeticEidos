#!/usr/bin/env python3
"""
Example 4: Complete Geometric ML Pipeline

Demonstrates the full end-to-end pipeline:
data → graph → metric → topology → stats → dynamics

This example ties together all components of the geometric ML framework
in a single, comprehensive demonstration.

Author: Sar Hamam
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from validation.reproducibility import ensure_reproducibility, compute_data_hash
from validation.numerical import validate_float64_precision
from validation.mathematical import check_graph_connectivity
from graphs.knn import build_graph
from graphs.laplacian import laplacian
from solvers.lanczos import topk_eigs
from stats.spectra import spectral_gap, spectral_entropy
from topology import create_topology, TopologyType, integrate_geodesic
from topology.coords import Strip

class GeometricMLPipeline:
    """Complete geometric ML pipeline implementation."""

    def __init__(self, seed=42):
        """Initialize pipeline with reproducibility."""
        self.seed = seed
        self.results = {}
        ensure_reproducibility(seed)

    def generate_data(self, n_samples=600, manifold_type='mixed', noise_level=0.1):
        """Generate synthetic manifold data."""

        print("🎯 STEP 1: DATA GENERATION")
        print("-" * 40)

        rng = np.random.default_rng(self.seed)

        if manifold_type == 'sphere':
            # Data on noisy sphere
            theta = rng.uniform(0, 2*np.pi, n_samples)
            phi = rng.uniform(0, np.pi, n_samples)

            X = np.column_stack([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            # Add noise and embed in higher dimensions
            noise = rng.normal(0, noise_level, X.shape)
            X += noise

            X_embedded = np.column_stack([
                X,
                rng.normal(0, noise_level/2, (n_samples, 5))
            ])

            intrinsic_params = (theta, phi)

        elif manifold_type == 'mixed':
            # Mixed data: two manifold patches
            n_half = n_samples // 2

            # Sphere patch
            theta1 = rng.uniform(0, np.pi, n_half)
            phi1 = rng.uniform(0, np.pi, n_half)
            sphere_patch = np.column_stack([
                np.sin(phi1) * np.cos(theta1),
                np.sin(phi1) * np.sin(theta1),
                np.cos(phi1)
            ])

            # Swiss roll patch
            t = rng.uniform(1.5*np.pi, 4.5*np.pi, n_samples - n_half)
            height = rng.uniform(0, 5, n_samples - n_half)
            roll_patch = np.column_stack([
                t * np.cos(t) * 0.3,
                height * 0.5,
                t * np.sin(t) * 0.3
            ])

            # Combine and embed
            X = np.vstack([sphere_patch, roll_patch])
            noise = rng.normal(0, noise_level, X.shape)
            X += noise

            X_embedded = np.column_stack([
                X,
                rng.normal(0, noise_level/2, (n_samples, 5))
            ])

            intrinsic_params = (theta1, phi1, t, height)

        else:
            raise ValueError(f"Unknown manifold type: {manifold_type}")

        # Validation
        data_hash = compute_data_hash(X_embedded)
        precision_check = validate_float64_precision({'data': X_embedded})

        print(f"   📊 Generated {manifold_type} data: {X_embedded.shape}")
        print(f"   🔒 Data hash: {data_hash[:16]}...")
        print(f"   ✅ Precision check: {'Pass' if precision_check['precision_adequate'] else 'Fail'}")

        self.results['data'] = {
            'X': X_embedded,
            'X_original': X,
            'manifold_type': manifold_type,
            'intrinsic_params': intrinsic_params,
            'hash': data_hash,
            'shape': X_embedded.shape
        }

        return X_embedded

    def build_graphs(self, X, k=16):
        """Build dual transport graphs."""

        print(f"\n🔗 STEP 2: GRAPH CONSTRUCTION")
        print("-" * 40)

        start_time = time.time()

        # Additive transport
        print(f"   📊 Building additive graph (k={k})...")
        G_add = build_graph(X, mode="additive", k=k, sigma="median", seed=self.seed)
        L_add = laplacian(G_add, normalized=True)
        conn_add = check_graph_connectivity(G_add, require_connected=True)

        # Multiplicative transport
        print(f"   📈 Building multiplicative graph (k={k})...")
        G_mult = build_graph(X, mode="multiplicative", k=k, tau="median", eps=1e-6, seed=self.seed)
        L_mult = laplacian(G_mult, normalized=True)
        conn_mult = check_graph_connectivity(G_mult, require_connected=True)

        build_time = time.time() - start_time

        print(f"   🔗 Additive connected: {'✅' if conn_add else '❌'}")
        print(f"   🔗 Multiplicative connected: {'✅' if conn_mult else '❌'}")
        print(f"   ⏱️  Build time: {build_time:.2f}s")

        self.results['graphs'] = {
            'G_additive': G_add,
            'G_multiplicative': G_mult,
            'L_additive': L_add,
            'L_multiplicative': L_mult,
            'connected_add': conn_add,
            'connected_mult': conn_mult,
            'k': k,
            'build_time': build_time
        }

        return G_add, G_mult, L_add, L_mult

    def compute_spectra(self, L_add, L_mult, n_eigs=20):
        """Compute spectral properties."""

        print(f"\n📊 STEP 3: SPECTRAL ANALYSIS")
        print("-" * 40)

        start_time = time.time()

        # Additive spectrum
        print(f"   🔍 Computing additive spectrum ({n_eigs} eigenvalues)...")
        eigenvals_add, eigenvecs_add, _ = topk_eigs(L_add, k=n_eigs, which="SM")
        gap_add = spectral_gap(L_add)
        entropy_add = spectral_entropy(L_add, k=16)

        # Multiplicative spectrum
        print(f"   🔍 Computing multiplicative spectrum ({n_eigs} eigenvalues)...")
        eigenvals_mult, eigenvecs_mult, _ = topk_eigs(L_mult, k=n_eigs, which="SM")
        gap_mult = spectral_gap(L_mult)
        entropy_mult = spectral_entropy(L_mult, k=16)

        spectral_time = time.time() - start_time

        print(f"   📈 Additive: gap={gap_add:.4f}, entropy={entropy_add:.4f}")
        print(f"   📈 Multiplicative: gap={gap_mult:.4f}, entropy={entropy_mult:.4f}")
        print(f"   ⏱️  Spectral time: {spectral_time:.2f}s")

        # Analysis
        gap_diff = abs(gap_add - gap_mult)
        entropy_diff = abs(entropy_add - entropy_mult)

        print(f"\n   🔍 Transport mode differences:")
        print(f"      Gap difference: {gap_diff:.4f}")
        print(f"      Entropy difference: {entropy_diff:.4f}")

        if gap_diff > 0.01 or entropy_diff > 0.01:
            print(f"      ✅ Significant differences detected!")
        else:
            print(f"      ⚠️  Transport modes show similar behavior")

        self.results['spectra'] = {
            'eigenvals_add': eigenvals_add,
            'eigenvals_mult': eigenvals_mult,
            'eigenvecs_add': eigenvecs_add,
            'eigenvecs_mult': eigenvecs_mult,
            'gap_add': gap_add,
            'gap_mult': gap_mult,
            'entropy_add': entropy_add,
            'entropy_mult': entropy_mult,
            'gap_diff': gap_diff,
            'entropy_diff': entropy_diff,
            'spectral_time': spectral_time
        }

        return eigenvals_add, eigenvals_mult

    def design_metric(self, alpha=0.7):
        """Design Fisher-Rao-inspired metric."""

        print(f"\n📐 STEP 4: METRIC DESIGN")
        print("-" * 40)

        def metric_fn(q):
            u, v = q[0], q[1]

            # Data-influenced Fisher-Rao-like metric
            data_term = np.exp(-0.1 * (u**2 + v**2))

            # Seam-compatible components
            g11 = 1.2 + alpha * data_term + 0.3 * np.cos(2*v)  # Even in v
            g22 = 0.9 + alpha * data_term * 0.8 + 0.2 * np.cos(4*v)  # Even in v
            g12 = 0.15 * alpha * np.sin(2*u) * np.sin(2*v)  # Odd in v

            return np.array([[g11, g12], [g12, g22]])

        def metric_grad_fn(q):
            h = 1e-6
            g_base = metric_fn(q)

            g_u_plus = metric_fn(q + np.array([h, 0]))
            du_g = (g_u_plus - g_base) / h

            g_v_plus = metric_fn(q + np.array([0, h]))
            dv_g = (g_v_plus - g_base) / h

            return du_g, dv_g

        # Test metric properties
        q_test = np.array([0.5, 0.3])
        g_test = metric_fn(q_test)
        det_g = np.linalg.det(g_test)
        eigenvals_g = np.linalg.eigvals(g_test)

        print(f"   📏 Test point: {q_test}")
        print(f"   📊 Determinant: {det_g:.4f}")
        print(f"   📊 Eigenvalues: [{eigenvals_g[0]:.3f}, {eigenvals_g[1]:.3f}]")
        print(f"   ✅ Positive definite: {'Yes' if np.all(eigenvals_g > 1e-12) else 'No'}")
        print(f"   🎯 Fisher-Rao parameter α: {alpha}")

        self.results['metric'] = {
            'metric_fn': metric_fn,
            'metric_grad_fn': metric_grad_fn,
            'alpha': alpha,
            'test_determinant': det_g,
            'test_eigenvals': eigenvals_g,
            'positive_definite': np.all(eigenvals_g > 1e-12)
        }

        return metric_fn, metric_grad_fn

    def select_topology(self, metric_fn):
        """Select and validate topology."""

        print(f"\n🌐 STEP 5: TOPOLOGY SELECTION")
        print("-" * 40)

        # Create candidate topologies
        topologies = {
            'mobius': create_topology(TopologyType.MOBIUS, w=1.2),
            'cylinder': create_topology(TopologyType.CYLINDER, w=1.2),
            'torus': create_topology(TopologyType.TORUS, width=2*np.pi, height=2*np.pi)
        }

        print(f"   🏗️  Created {len(topologies)} candidate topologies")

        # Test metric compatibility
        best_topology = None
        best_compatibility = 0.0

        test_points = [
            np.array([0.3, 0.2]),
            np.array([1.0, 0.5]),
            np.array([1.8, -0.3]),
            np.array([0.7, -0.4])
        ]

        for name, topology in topologies.items():
            print(f"\n   Testing {name.title()}:")
            print(f"      Orientable: {topology.orientability.value}")
            print(f"      Euler χ: {topology.euler_characteristic}")

            compatible_points = 0
            for q in test_points:
                try:
                    if topology.metric_compatibility_condition(metric_fn, q):
                        compatible_points += 1
                except Exception as e:
                    pass

            compatibility_rate = compatible_points / len(test_points)
            print(f"      Compatibility: {compatibility_rate:.1%}")

            if compatibility_rate > best_compatibility:
                best_compatibility = compatibility_rate
                best_topology = (name, topology)

        selected_name, selected_topology = best_topology
        print(f"\n   🎯 Selected: {selected_name.title()} (compatibility: {best_compatibility:.1%})")

        self.results['topology'] = {
            'candidates': topologies,
            'selected_name': selected_name,
            'selected': selected_topology,
            'compatibility_rate': best_compatibility,
            'test_points': test_points
        }

        return selected_topology

    def integrate_dynamics(self, metric_fn, metric_grad_fn, topology, n_geodesics=5):
        """Integrate geodesic dynamics."""

        print(f"\n🌊 STEP 6: DYNAMICS INTEGRATION")
        print("-" * 40)

        # Check if topology supports geodesic integration
        if not hasattr(topology, 'w'):
            print(f"   ⚠️  Geodesic integration not implemented for {type(topology).__name__}")
            self.results['dynamics'] = {'error': 'Not implemented'}
            return []

        strip = Strip(w=topology.w, period=getattr(topology, 'period', 2*np.pi))

        print(f"   🏗️  Setup: w={strip.w}, period={strip.period:.2f}")

        # Validate metric on topology
        # Note: comprehensive_topology_validation is for Möbius band seam-compatibility
        # For cylinder/torus, we just need positive definiteness which we already checked
        selected_name = self.results.get('topology', {}).get('selected_name', 'unknown')

        if selected_name in ['mobius', 'klein']:
            # Use full seam-compatibility validation for non-orientable topologies
            try:
                from topology import comprehensive_topology_validation
                validation = comprehensive_topology_validation(
                    g_fn=metric_fn, strip=strip, tolerance=1e-6
                )
                validation_pass = validation['all_passed']
                print(f"   ✅ Seam-compatibility validation: {'Pass' if validation_pass else 'Fail'}")
            except Exception as e:
                validation_pass = False
                print(f"   ⚠️  Validation error: {str(e)[:40]}...")
        else:
            # For orientable topologies (cylinder, torus), just check positive definiteness
            try:
                test_points = [np.array([1.0, 0.3]), np.array([3.0, -0.5]), np.array([5.0, 0.8])]
                all_positive = True
                for q in test_points:
                    g = metric_fn(q)
                    eigenvals = np.linalg.eigvals(g)
                    if not np.all(eigenvals > 1e-12):
                        all_positive = False
                        break
                validation_pass = all_positive
                print(f"   ✅ Positive definiteness check: {'Pass' if validation_pass else 'Fail'}")
            except Exception as e:
                validation_pass = False
                print(f"   ⚠️  Validation error: {str(e)[:40]}...")

        if not validation_pass:
            print(f"   ❌ Skipping geodesic integration due to validation failure")
            self.results['dynamics'] = {'error': 'Validation failed'}
            return []

        # Integrate multiple geodesics
        rng = np.random.default_rng(self.seed + 100)  # Different seed for dynamics
        geodesic_results = []

        start_time = time.time()

        for i in range(n_geodesics):
            # Random initial conditions
            q0 = np.array([
                rng.uniform(0.1*strip.period, 0.9*strip.period),
                rng.uniform(-0.8*strip.w, 0.8*strip.w)
            ])
            v0 = rng.normal(0, 0.4, 2)

            print(f"   🚀 Geodesic {i+1}: q0=[{q0[0]:.2f}, {q0[1]:.2f}], |v0|={np.linalg.norm(v0):.2f}")

            try:
                traj_q, traj_v, info = integrate_geodesic(
                    q0, v0,
                    t_final=5.0,
                    dt=0.01,
                    g_fn=metric_fn,
                    grad_g_fn=metric_grad_fn,
                    strip=strip,
                    energy_tolerance=1e-2
                )

                if info['success']:
                    print(f"      ✅ Success: {len(traj_q)} steps, drift={info['energy_drift']:.1e}, crossings={info['seam_crossings']}")

                    geodesic_results.append({
                        'index': i,
                        'q0': q0,
                        'v0': v0,
                        'trajectory_q': traj_q,
                        'trajectory_v': traj_v,
                        'info': info
                    })
                else:
                    print(f"      ❌ Failed")

            except Exception as e:
                print(f"      ⚠️  Error: {str(e)[:40]}...")

        dynamics_time = time.time() - start_time

        # Summary statistics
        if geodesic_results:
            n_success = len(geodesic_results)
            energy_drifts = [r['info']['energy_drift'] for r in geodesic_results]
            seam_crossings = [r['info']['seam_crossings'] for r in geodesic_results]
            trajectory_lengths = [r['info']['trajectory_length'] for r in geodesic_results]

            print(f"\n   📊 Dynamics Summary:")
            print(f"      Successful integrations: {n_success}/{n_geodesics}")
            print(f"      Mean energy drift: {np.mean(energy_drifts):.2e}")
            print(f"      Total seam crossings: {sum(seam_crossings)}")
            print(f"      Mean trajectory length: {np.mean(trajectory_lengths):.2f}")
            print(f"      Integration time: {dynamics_time:.2f}s")

            self.results['dynamics'] = {
                'n_success': n_success,
                'n_total': n_geodesics,
                'geodesic_results': geodesic_results,
                'mean_energy_drift': np.mean(energy_drifts),
                'total_seam_crossings': sum(seam_crossings),
                'mean_trajectory_length': np.mean(trajectory_lengths),
                'dynamics_time': dynamics_time,
                'validation_pass': validation_pass
            }
        else:
            print(f"   ❌ No successful geodesic integrations")
            self.results['dynamics'] = {'n_success': 0, 'n_total': n_geodesics}

        return geodesic_results

    def create_summary_visualization(self):
        """Create comprehensive pipeline visualization."""

        print(f"\n📊 STEP 7: VISUALIZATION")
        print("-" * 40)

        if 'data' not in self.results or 'spectra' not in self.results:
            print("   ⚠️  Insufficient data for visualization")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Original data (3D projection)
        ax1 = axes[0, 0]
        X_orig = self.results['data']['X_original']
        manifold_type = self.results['data']['manifold_type']

        if manifold_type == 'mixed':
            n_half = len(X_orig) // 2
            ax1.scatter(X_orig[:n_half, 0], X_orig[:n_half, 1], alpha=0.6, s=20, label='Sphere patch')
            ax1.scatter(X_orig[n_half:, 0], X_orig[n_half:, 1], alpha=0.6, s=20, label='Roll patch')
        else:
            ax1.scatter(X_orig[:, 0], X_orig[:, 1], alpha=0.6, s=20)

        ax1.set_title(f'{manifold_type.title()} Data (2D Projection)')
        ax1.set_xlabel('X₁')
        ax1.set_ylabel('X₂')
        if manifold_type == 'mixed':
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Spectral comparison
        ax2 = axes[0, 1]
        eigenvals_add = self.results['spectra']['eigenvals_add'][:15]
        eigenvals_mult = self.results['spectra']['eigenvals_mult'][:15]

        ax2.plot(eigenvals_add, 'o-', label='Additive', linewidth=2, markersize=6)
        ax2.plot(eigenvals_mult, 's-', label='Multiplicative', linewidth=2, markersize=6)
        ax2.set_title('Eigenvalue Spectra')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Eigenvalue')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Transport mode comparison
        ax3 = axes[0, 2]
        gap_add = self.results['spectra']['gap_add']
        gap_mult = self.results['spectra']['gap_mult']
        entropy_add = self.results['spectra']['entropy_add']
        entropy_mult = self.results['spectra']['entropy_mult']

        metrics = ['Gap', 'Entropy']
        add_values = [gap_add, entropy_add]
        mult_values = [gap_mult, entropy_mult]

        x = np.arange(len(metrics))
        width = 0.35

        ax3.bar(x - width/2, add_values, width, label='Additive', alpha=0.8)
        ax3.bar(x + width/2, mult_values, width, label='Multiplicative', alpha=0.8)
        ax3.set_title('Transport Mode Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Fiedler vector visualization
        ax4 = axes[1, 0]
        if 'eigenvecs_add' in self.results['spectra'] and len(self.results['spectra']['eigenvecs_add']) > 1:
            fiedler = self.results['spectra']['eigenvecs_add'][:, 1]
            X = self.results['data']['X']

            scatter = ax4.scatter(X[:, 0], X[:, 1], c=fiedler, cmap='RdBu', s=30, alpha=0.7)
            ax4.set_title('Fiedler Vector (Additive)')
            ax4.set_xlabel('X₁')
            ax4.set_ylabel('X₂')
            plt.colorbar(scatter, ax=ax4)

        # 5. Topology validation
        ax5 = axes[1, 1]
        if 'topology' in self.results:
            selected_name = self.results['topology']['selected_name']
            compatibility_rate = self.results['topology']['compatibility_rate']

            # Simple bar chart of compatibility
            ax5.bar([selected_name], [compatibility_rate], alpha=0.8, color='green' if compatibility_rate > 0.7 else 'orange')
            ax5.set_title('Topology Compatibility')
            ax5.set_ylabel('Compatibility Rate')
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)

        # 6. Geodesic dynamics
        ax6 = axes[1, 2]
        if 'dynamics' in self.results and 'geodesic_results' in self.results['dynamics']:
            geodesic_results = self.results['dynamics']['geodesic_results']

            if geodesic_results:
                # Plot a few trajectories
                for i, result in enumerate(geodesic_results[:3]):
                    traj_q = result['trajectory_q']
                    ax6.plot(traj_q[:, 0], traj_q[:, 1], '-', alpha=0.7, linewidth=2, label=f'Geodesic {i+1}')

                ax6.set_title('Geodesic Trajectories')
                ax6.set_xlabel('u')
                ax6.set_ylabel('v')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No Geodesics', transform=ax6.transAxes, ha='center', va='center')
                ax6.set_title('Geodesic Trajectories')
        else:
            ax6.text(0.5, 0.5, 'Dynamics Failed', transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title('Geodesic Trajectories')

        plt.tight_layout()
        plt.suptitle('Complete Geometric ML Pipeline Results', fontsize=16, y=1.02)

        # Save
        output_path = 'complete_pipeline_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   📊 Pipeline visualization saved to: {output_path}")

        try:
            plt.show()
        except:
            pass

    def print_final_summary(self):
        """Print comprehensive pipeline summary."""

        print(f"\n🎯 PIPELINE SUMMARY")
        print("=" * 60)

        # Data summary
        if 'data' in self.results:
            data_info = self.results['data']
            print(f"📊 Data: {data_info['shape']} {data_info['manifold_type']} manifold")

        # Graph summary
        if 'graphs' in self.results:
            graph_info = self.results['graphs']
            conn_status = "✅" if graph_info['connected_add'] and graph_info['connected_mult'] else "⚠️"
            print(f"🔗 Graphs: k={graph_info['k']}, connected={conn_status}, time={graph_info['build_time']:.1f}s")

        # Spectral summary
        if 'spectra' in self.results:
            spec_info = self.results['spectra']
            print(f"📈 Spectra: Δgap={spec_info['gap_diff']:.3f}, Δentropy={spec_info['entropy_diff']:.3f}, time={spec_info['spectral_time']:.1f}s")

        # Metric summary
        if 'metric' in self.results:
            metric_info = self.results['metric']
            pd_status = "✅" if metric_info['positive_definite'] else "❌"
            print(f"📐 Metric: α={metric_info['alpha']}, positive_definite={pd_status}")

        # Topology summary
        if 'topology' in self.results:
            topo_info = self.results['topology']
            print(f"🌐 Topology: {topo_info['selected_name']}, compatibility={topo_info['compatibility_rate']:.1%}")

        # Dynamics summary
        if 'dynamics' in self.results:
            dyn_info = self.results['dynamics']
            if 'n_success' in dyn_info:
                success_rate = dyn_info['n_success'] / dyn_info['n_total']
                if dyn_info['n_success'] > 0:
                    print(f"🌊 Dynamics: {dyn_info['n_success']}/{dyn_info['n_total']} geodesics, "
                          f"drift={dyn_info['mean_energy_drift']:.1e}, crossings={dyn_info['total_seam_crossings']}")
                else:
                    print(f"🌊 Dynamics: Failed - no successful integrations")

        # Overall assessment
        success_components = 0
        total_components = 6

        if 'data' in self.results:
            success_components += 1
        if 'graphs' in self.results and self.results['graphs']['connected_add'] and self.results['graphs']['connected_mult']:
            success_components += 1
        if 'spectra' in self.results:
            success_components += 1
        if 'metric' in self.results and self.results['metric']['positive_definite']:
            success_components += 1
        if 'topology' in self.results and self.results['topology']['compatibility_rate'] > 0.5:
            success_components += 1
        if 'dynamics' in self.results and self.results['dynamics'].get('n_success', 0) > 0:
            success_components += 1

        success_rate = success_components / total_components

        print(f"\n🎯 OVERALL SUCCESS: {success_components}/{total_components} components ({success_rate:.1%})")

        if success_rate >= 0.8:
            print("✅ EXCELLENT: Pipeline completed successfully!")
        elif success_rate >= 0.6:
            print("⚠️  GOOD: Pipeline mostly successful with minor issues")
        elif success_rate >= 0.4:
            print("⚠️  FAIR: Pipeline partially successful, some components failed")
        else:
            print("❌ POOR: Pipeline had significant failures")

def main():
    """Run the complete geometric ML pipeline."""

    print("🚀 EXAMPLE 4: Complete Geometric ML Pipeline")
    print("=" * 60)
    print("This demonstrates the full end-to-end workflow:")
    print("Data → Graph → Metric → Topology → Stats → Dynamics")
    print("=" * 60)

    # Initialize pipeline
    pipeline = GeometricMLPipeline(seed=42)

    try:
        # Run complete pipeline
        X = pipeline.generate_data(n_samples=500, manifold_type='mixed')
        G_add, G_mult, L_add, L_mult = pipeline.build_graphs(X, k=16)
        eigenvals_add, eigenvals_mult = pipeline.compute_spectra(L_add, L_mult)
        metric_fn, metric_grad_fn = pipeline.design_metric(alpha=0.8)
        topology = pipeline.select_topology(metric_fn)
        geodesic_results = pipeline.integrate_dynamics(metric_fn, metric_grad_fn, topology, n_geodesics=6)

        # Create visualization
        try:
            pipeline.create_summary_visualization()
        except ImportError:
            print("\n   📊 Matplotlib not available for visualization")

        # Final summary
        pipeline.print_final_summary()

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✅ Complete pipeline demonstration finished!")
    print("\nNext steps:")
    print("   • Analyze results in complete_pipeline_results.png")
    print("   • Try different manifold types (sphere vs mixed)")
    print("   • Explore examples/05_validation_demo.py for validation framework")
    print("   • Check out the interactive notebooks for parameter exploration")

if __name__ == "__main__":
    main()
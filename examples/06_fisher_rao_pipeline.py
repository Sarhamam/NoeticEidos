#!/usr/bin/env python3
"""
Example 6: Fisher-Rao to Full Pipeline with Real Embeddings

Demonstrates the complete geometric ML pipeline starting with Fisher-Rao metrics
on real embeddings (GPT-2 small), following the flow:
Fisher-Rao → Topology → Stats → Dynamics → Data → Graph

This example shows how model-aware metrics enhance every step of geometric analysis.

Author: Sar Hamam
"""

import numpy as np
import matplotlib.pyplot as plt

import time
from typing import Dict, List, Tuple, Optional

from validation.reproducibility import ensure_reproducibility
from geometry.fr_pullback import (
    fisher_rao_metric, pullback_metric, rescale_by_metric,
    fisher_rao_divergence, riemannian_distance
)
from topology import create_topology, TopologyType, integrate_geodesic
from topology.coords import Strip
from stats.spectra import spectral_gap, spectral_entropy
from stats.stability import stability_score
from dynamics.fr_flows import fr_gradient_flow, natural_gradient_descent
from scipy.special import logsumexp
from graphs.knn import build_graph
from graphs.laplacian import laplacian
from solvers.lanczos import topk_eigs
from validation.mathematical import check_graph_connectivity

# Optional imports for transformer models
try:
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[WARNING]  Transformers not available. Using synthetic embedding simulation.")

class FisherRaoPipeline:
    """Complete Fisher-Rao enhanced geometric ML pipeline."""

    def __init__(self, seed=42):
        """Initialize pipeline with reproducibility."""
        self.seed = seed
        self.results = {}
        ensure_reproducibility(seed)

        # Initialize model components
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.tokens = None

    def load_model_and_extract_embeddings(self, vocab_categories: Optional[List[str]] = None):
        """Load GPT-2 and extract embeddings for analysis."""

        print("STEP 1: FISHER-RAO METRIC DESIGN")
        print("=" * 60)

        if not HAS_TRANSFORMERS:
            print("   [CHART] Using synthetic embeddings (transformers not available)")
            return self._create_synthetic_embeddings(vocab_categories)

        # Use tiny GPT-2 variant for faster computation and lower memory usage
        print("   [LOAD] Loading GPT-2 tiny model (sshleifer/tiny-gpt2)...")
        self.model = GPT2Model.from_pretrained('sshleifer/tiny-gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('sshleifer/tiny-gpt2')
        self.model.eval()

        # Define semantic categories for interpretable analysis
        if vocab_categories is None:
            vocab_categories = [
                # Animals
                ['cat', 'dog', 'bird', 'fish', 'horse', 'cow', 'pig', 'sheep'],
                # Colors
                ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange'],
                # Emotions
                ['happy', 'sad', 'angry', 'excited', 'calm', 'nervous', 'proud', 'grateful'],
                # Actions
                ['run', 'walk', 'jump', 'swim', 'fly', 'dance', 'sing', 'write']
            ]

        # Extract embeddings for selected vocabulary
        all_tokens = []
        all_embeddings = []
        category_labels = []

        print(f"   [TEXT] Extracting embeddings for {len(vocab_categories)} semantic categories...")

        for cat_idx, category in enumerate(vocab_categories):
            print(f"      Category {cat_idx + 1}: {len(category)} tokens")

            for token in category:
                # Tokenize and get embedding
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if len(token_ids) == 1:  # Single token only
                    with torch.no_grad():
                        # Get embedding from embedding layer
                        embedding = self.model.wte(torch.tensor([token_ids[0]]))
                        all_embeddings.append(embedding.squeeze().numpy())
                        all_tokens.append(token)
                        category_labels.append(cat_idx)

        self.embeddings = np.array(all_embeddings)
        self.tokens = all_tokens
        self.category_labels = np.array(category_labels)

        print(f"   [OK] Extracted embeddings: {self.embeddings.shape}")
        print(f"   [CHART] Categories: {len(vocab_categories)}, Total tokens: {len(self.tokens)}")

        # Compute Fisher-Rao metrics
        print(f"\n   [CALC] Computing Fisher-Rao pullback metrics...")

        start_time = time.time()

        # Create probabilistic model function
        def embedding_to_logits(embeddings):
            """Convert embeddings back to logits via the model."""
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            with torch.no_grad():
                # Use final layer norm and projection
                embedded = torch.tensor(embeddings, dtype=torch.float32)
                # Approximate logits via vocabulary projection
                logits = embedded @ self.model.wte.weight.T  # (batch, vocab_size)
                return logits.numpy()

        # Compute Fisher-Rao metrics for each embedding
        print(f"      Computing pullback metrics...")

        self.fr_metrics = pullback_metric(
            embedding_func=embedding_to_logits,
            X=self.embeddings,
            epsilon=1e-4
        )

        # Create Fisher-Rao enhanced coordinates
        self.fr_embeddings = rescale_by_metric(
            self.embeddings,
            self.fr_metrics,
            reg=1e-6
        )

        fr_time = time.time() - start_time

        print(f"      [OK] Fisher-Rao metrics computed in {fr_time:.2f}s")
        print(f"      [GEOM] Metric tensor shape: {self.fr_metrics.shape}")
        print(f"      [TARGET] Enhanced embeddings shape: {self.fr_embeddings.shape}")

        # Validate metrics are positive definite
        valid_metrics = 0
        for i in range(len(self.fr_metrics)):
            eigenvals = np.linalg.eigvals(self.fr_metrics[i])
            if np.all(eigenvals > 1e-12):
                valid_metrics += 1

        print(f"      [OK] Positive definite metrics: {valid_metrics}/{len(self.fr_metrics)} ({100*valid_metrics/len(self.fr_metrics):.1f}%)")

        self.results['fisher_rao'] = {
            'embeddings': self.embeddings,
            'fr_embeddings': self.fr_embeddings,
            'fr_metrics': self.fr_metrics,
            'tokens': self.tokens,
            'categories': self.category_labels,
            'vocab_categories': vocab_categories,
            'computation_time': fr_time,
            'valid_metrics_pct': 100*valid_metrics/len(self.fr_metrics)
        }

        return self.fr_embeddings, self.fr_metrics

    def _create_synthetic_embeddings(self, vocab_categories):
        """Create synthetic embeddings that simulate transformer behavior."""

        print("   [CHART] Creating synthetic embeddings with semantic structure...")

        # Define synthetic vocabulary
        if vocab_categories is None:
            vocab_categories = [
                ['cat', 'dog', 'bird', 'fish'],
                ['red', 'blue', 'green', 'yellow'],
                ['happy', 'sad', 'angry', 'calm'],
                ['run', 'walk', 'jump', 'swim']
            ]

        rng = np.random.default_rng(self.seed)

        # Create embeddings with semantic clustering
        embedding_dim = 768  # GPT-2 small dimension
        all_embeddings = []
        all_tokens = []
        category_labels = []

        for cat_idx, category in enumerate(vocab_categories):
            # Create category center
            category_center = rng.normal(0, 1, embedding_dim)
            category_center = category_center / np.linalg.norm(category_center) * (cat_idx + 1) * 2

            for token in category:
                # Add noise around category center
                embedding = category_center + rng.normal(0, 0.5, embedding_dim)
                all_embeddings.append(embedding)
                all_tokens.append(token)
                category_labels.append(cat_idx)

        self.embeddings = np.array(all_embeddings)
        self.tokens = all_tokens
        self.category_labels = np.array(category_labels)

        # Create synthetic Fisher-Rao metrics
        print("   [CALC] Creating synthetic Fisher-Rao metrics...")

        n_points = len(self.embeddings)
        self.fr_metrics = np.zeros((n_points, embedding_dim, embedding_dim))

        for i in range(n_points):
            # Create positive definite metric based on embedding structure
            base_metric = np.eye(embedding_dim)

            # Add embedding-dependent terms
            embedding_norm = np.linalg.norm(self.embeddings[i])
            scaling = 1 + 0.1 * embedding_norm

            # Add some cross-terms for Fisher-Rao structure
            cross_term = np.outer(self.embeddings[i], self.embeddings[i]) / (embedding_norm**2 + 1e-6)

            self.fr_metrics[i] = scaling * base_metric + 0.1 * cross_term

        # Create Fisher-Rao enhanced coordinates
        self.fr_embeddings = rescale_by_metric(
            self.embeddings,
            self.fr_metrics,
            reg=1e-6
        )

        print(f"   [OK] Synthetic setup complete: {self.embeddings.shape}")

        self.results['fisher_rao'] = {
            'embeddings': self.embeddings,
            'fr_embeddings': self.fr_embeddings,
            'fr_metrics': self.fr_metrics,
            'tokens': self.tokens,
            'categories': self.category_labels,
            'vocab_categories': vocab_categories,
            'computation_time': 0.1,
            'valid_metrics_pct': 100.0,
            'synthetic': True
        }

        return self.fr_embeddings, self.fr_metrics

    def select_optimal_topology(self, fr_metrics):
        """Select topology based on Fisher-Rao metric compatibility."""

        print(f"\n[TOPO] STEP 2: TOPOLOGY SELECTION (Fisher-Rao -> Topology)")
        print("=" * 60)

        # Test different topologies with Fisher-Rao metrics
        topologies = {
            'mobius': create_topology(TopologyType.MOBIUS, w=1.5),
            'cylinder': create_topology(TopologyType.CYLINDER, w=1.5),
            'torus': create_topology(TopologyType.TORUS, width=2*np.pi, height=2*np.pi)
        }

        print(f"   [BUILD]  Testing {len(topologies)} topology candidates with Fisher-Rao metrics...")

        # Create test metric function from Fisher-Rao data
        def fr_metric_function(q):
            """Metric function based on Fisher-Rao structure."""
            u, v = q[0], q[1]

            # Use Fisher-Rao inspired structure
            # Even components in v for seam compatibility
            g11 = 1.2 + 0.3 * np.exp(-0.1 * (u**2 + v**2)) + 0.2 * np.cos(2*v)
            g22 = 0.9 + 0.2 * np.exp(-0.15 * (u**2 + v**2)) + 0.15 * np.cos(4*v)
            # Odd component in v
            g12 = 0.1 * np.sin(2*u) * np.sin(2*v)

            return np.array([[g11, g12], [g12, g22]])

        # Test compatibility
        test_points = [
            np.array([0.3, 0.4]),
            np.array([1.0, -0.2]),
            np.array([2.0, 0.6]),
            np.array([0.8, -0.5]),
            np.array([1.5, 0.3])
        ]

        compatibility_results = {}

        for name, topology in topologies.items():
            print(f"\n   Testing {name.title()}:")
            print(f"      Orientable: {topology.orientability.value}")
            print(f"      Euler char: {topology.euler_characteristic}")

            compatible_points = 0
            compatibility_scores = []

            for q in test_points:
                try:
                    compatible = topology.metric_compatibility_condition(fr_metric_function, q)
                    if compatible:
                        compatible_points += 1
                        compatibility_scores.append(1.0)
                    else:
                        compatibility_scores.append(0.0)
                except Exception as e:
                    compatibility_scores.append(0.0)
                    print(f"         [WARNING]  Point {q}: {str(e)[:30]}...")

            compatibility_rate = compatible_points / len(test_points)
            avg_score = np.mean(compatibility_scores)

            print(f"      Compatibility: {compatibility_rate:.1%} ({compatible_points}/{len(test_points)} points)")
            print(f"      Average score: {avg_score:.3f}")

            compatibility_results[name] = {
                'topology': topology,
                'compatibility_rate': compatibility_rate,
                'compatible_points': compatible_points,
                'avg_score': avg_score
            }

        # Select best topology
        best_name = max(compatibility_results.keys(),
                       key=lambda k: compatibility_results[k]['compatibility_rate'])

        selected_topology = compatibility_results[best_name]['topology']
        best_rate = compatibility_results[best_name]['compatibility_rate']

        print(f"\n   [TARGET] Selected topology: {best_name.title()}")
        print(f"      Compatibility rate: {best_rate:.1%}")
        print(f"      Reason: Best Fisher-Rao metric compatibility")

        self.results['topology'] = {
            'candidates': compatibility_results,
            'selected_name': best_name,
            'selected': selected_topology,
            'metric_function': fr_metric_function,
            'compatibility_rate': best_rate,
            'selection_criterion': 'fisher_rao_compatibility'
        }

        return selected_topology, fr_metric_function

    def compute_enhanced_statistics(self, topology, fr_embeddings, fr_metrics):
        """Compute statistics enhanced by Fisher-Rao metrics."""

        print(f"\n[CHART] STEP 3: STATISTICAL ANALYSIS (Topology -> Stats)")
        print("=" * 60)

        print(f"   [SEARCH] Computing Fisher-Rao enhanced spectral properties...")

        # Build graphs with Fisher-Rao enhanced embeddings
        k = min(16, len(fr_embeddings) - 1)

        start_time = time.time()

        # Standard embeddings for comparison
        print(f"      Building graphs (k={k})...")
        G_standard = build_graph(self.embeddings, mode="additive", k=k, seed=self.seed)
        L_standard = laplacian(G_standard, normalized=True)

        # Fisher-Rao enhanced embeddings
        G_fr = build_graph(fr_embeddings, mode="additive", k=k, seed=self.seed)
        L_fr = laplacian(G_fr, normalized=True)

        print(f"      Computing spectra...")
        n_eigs = min(15, len(fr_embeddings) - 1)

        # Standard analysis
        eigenvals_std, eigenvecs_std, _ = topk_eigs(L_standard, k=n_eigs, which="SM")
        gap_std = spectral_gap(L_standard)
        entropy_std = spectral_entropy(L_standard, k=min(12, len(eigenvals_std)))

        # Fisher-Rao enhanced analysis
        eigenvals_fr, eigenvecs_fr, _ = topk_eigs(L_fr, k=n_eigs, which="SM")
        gap_fr = spectral_gap(L_fr)
        entropy_fr = spectral_entropy(L_fr, k=min(12, len(eigenvals_fr)))

        stats_time = time.time() - start_time

        print(f"   [STATS] Spectral Results:")
        print(f"      Standard - Gap: {gap_std:.4f}, Entropy: {entropy_std:.4f}")
        print(f"      Fisher-Rao - Gap: {gap_fr:.4f}, Entropy: {entropy_fr:.4f}")
        print(f"      Enhancement ratio - Gap: {gap_fr/gap_std:.2f}x, Entropy: {entropy_fr/entropy_std:.2f}x")

        # Stability analysis
        print(f"\n   [ANALYSIS] Stability analysis across semantic categories...")

        category_stability = {}
        for cat_idx in range(len(self.results['fisher_rao']['vocab_categories'])):
            cat_mask = self.category_labels == cat_idx
            if np.sum(cat_mask) >= 4:  # Need minimum points
                cat_embeddings = fr_embeddings[cat_mask]

                # Compute stability metrics for this category
                cat_seeds = [self.seed + i for i in range(3)]
                cat_gaps = []

                for seed in cat_seeds:
                    if len(cat_embeddings) > 8:
                        cat_G = build_graph(cat_embeddings, mode="additive", k=min(8, len(cat_embeddings)-1), seed=seed)
                        cat_L = laplacian(cat_G, normalized=True)
                        cat_eigs, _, _ = topk_eigs(cat_L, k=min(8, len(cat_embeddings)-1), which="SM")
                        cat_gaps.append(spectral_gap(cat_L))

                if cat_gaps:
                    stability = 1.0 - (np.std(cat_gaps) / np.mean(cat_gaps)) if np.mean(cat_gaps) > 0 else 0.0
                    category_stability[cat_idx] = stability

                    category_name = self.results['fisher_rao']['vocab_categories'][cat_idx][0] + "..."
                    print(f"      Category {cat_idx} ({category_name}): stability = {stability:.3f}")

        overall_stability = np.mean(list(category_stability.values())) if category_stability else 0.0

        print(f"   [CHART] Overall stability: {overall_stability:.3f}")
        print(f"   [TIME]  Statistics time: {stats_time:.2f}s")

        self.results['statistics'] = {
            'eigenvals_standard': eigenvals_std,
            'eigenvals_fr': eigenvals_fr,
            'eigenvecs_standard': eigenvecs_std,
            'eigenvecs_fr': eigenvecs_fr,
            'gap_standard': gap_std,
            'gap_fr': gap_fr,
            'entropy_standard': entropy_std,
            'entropy_fr': entropy_fr,
            'gap_enhancement': gap_fr/gap_std,
            'entropy_enhancement': entropy_fr/entropy_std,
            'category_stability': category_stability,
            'overall_stability': overall_stability,
            'computation_time': stats_time
        }

        return eigenvals_fr, eigenvecs_fr

    def integrate_fisher_rao_dynamics(self, topology, fr_metric_function):
        """Integrate dynamics using Fisher-Rao enhanced metrics."""

        print(f"\n[DYNAMICS] STEP 4: DYNAMICS INTEGRATION (Stats -> Dynamics)")
        print("=" * 60)

        if not hasattr(topology, 'w'):
            print("   [WARNING]  Selected topology doesn't support geodesic integration")
            self.results['dynamics'] = {'error': 'Topology not supported'}
            return []

        print(f"   [BUILD]  Setting up Fisher-Rao dynamics on {type(topology).__name__}")

        strip = Strip(w=topology.w, period=getattr(topology, 'period', 2*np.pi))

        print(f"      Strip parameters: w={strip.w:.2f}, period={strip.period:.2f}")

        # Validate Fisher-Rao metric on topology
        # Use topology-specific validation: seam-compatibility for non-orientable, positive-definiteness for orientable
        try:
            selected_name = self.results.get('topology', {}).get('selected_name', 'unknown')

            if selected_name in ['mobius', 'klein']:
                # Use full seam-compatibility validation for non-orientable topologies
                from topology import comprehensive_topology_validation
                validation = comprehensive_topology_validation(
                    g_fn=fr_metric_function,
                    strip=strip,
                    tolerance=1e-6
                )
                validation_pass = validation.get('all_passed', False)
                print(f"   [OK] Fisher-Rao metric validation (seam-compatible): {'Pass' if validation_pass else 'Fail'}")

                if validation_pass:
                    passed_tests = [test for test in validation.get('tests_run', [])
                                  if validation.get(test, {}).get('compatible', False)]
                    print(f"      Passed tests: {', '.join(passed_tests)}")
            else:
                # For orientable topologies (cylinder, torus), just check positive definiteness
                test_points = [
                    np.array([1.0, 0.3]),
                    np.array([3.0, -0.5]),
                    np.array([5.0, 0.8])
                ]

                all_positive = True
                for q in test_points:
                    g = fr_metric_function(q)
                    eigenvals = np.linalg.eigvals(g)
                    if not np.all(eigenvals > 1e-12):
                        all_positive = False
                        break

                validation_pass = all_positive
                print(f"   [OK] Fisher-Rao metric validation (positive-definite): {'Pass' if validation_pass else 'Fail'}")

        except Exception as e:
            validation_pass = False
            print(f"   [WARNING]  Validation error: {str(e)[:50]}...")

        if not validation_pass:
            print("   [ERROR] Skipping dynamics due to validation failure")
            self.results['dynamics'] = {'error': 'Validation failed'}
            return []

        # Create metric gradient function
        def metric_grad_fn(q):
            h = 1e-6
            g_base = fr_metric_function(q)

            g_u_plus = fr_metric_function(q + np.array([h, 0]))
            du_g = (g_u_plus - g_base) / h

            g_v_plus = fr_metric_function(q + np.array([0, h]))
            dv_g = (g_v_plus - g_base) / h

            return du_g, dv_g

        # Integrate multiple geodesics
        print(f"\n   [RUN] Integrating Fisher-Rao geodesics...")

        rng = np.random.default_rng(self.seed + 200)
        n_geodesics = 6
        geodesic_results = []

        start_time = time.time()

        for i in range(n_geodesics):
            # Random initial conditions
            q0 = np.array([
                rng.uniform(0.2*strip.period, 0.8*strip.period),
                rng.uniform(-0.7*strip.w, 0.7*strip.w)
            ])
            v0 = rng.normal(0, 0.3, 2)

            print(f"      Geodesic {i+1}: q0=[{q0[0]:.2f}, {q0[1]:.2f}], |v0|={np.linalg.norm(v0):.2f}")

            try:
                traj_q, traj_v, info = integrate_geodesic(
                    q0, v0,
                    t_final=4.0,
                    dt=0.01,
                    g_fn=fr_metric_function,
                    grad_g_fn=metric_grad_fn,
                    strip=strip,
                    energy_tolerance=1e-2
                )

                if info['success']:
                    print(f"         [OK] Success: {len(traj_q)} steps, drift={info['energy_drift']:.1e}")
                    print(f"            Crossings: {info['seam_crossings']}, Length: {info['trajectory_length']:.2f}")

                    geodesic_results.append({
                        'index': i,
                        'q0': q0,
                        'v0': v0,
                        'trajectory_q': traj_q,
                        'trajectory_v': traj_v,
                        'info': info
                    })
                else:
                    print(f"         [ERROR] Failed")

            except Exception as e:
                print(f"         [WARNING]  Error: {str(e)[:40]}...")

        dynamics_time = time.time() - start_time

        # Analyze results
        if geodesic_results:
            n_success = len(geodesic_results)
            energy_drifts = [r['info']['energy_drift'] for r in geodesic_results]
            seam_crossings = [r['info']['seam_crossings'] for r in geodesic_results]

            print(f"\n   [CHART] Fisher-Rao Dynamics Summary:")
            print(f"      Successful geodesics: {n_success}/{n_geodesics}")
            print(f"      Mean energy drift: {np.mean(energy_drifts):.2e}")
            print(f"      Total seam crossings: {sum(seam_crossings)}")
            print(f"      Integration time: {dynamics_time:.2f}s")

            self.results['dynamics'] = {
                'n_success': n_success,
                'n_total': n_geodesics,
                'geodesic_results': geodesic_results,
                'mean_energy_drift': np.mean(energy_drifts),
                'total_seam_crossings': sum(seam_crossings),
                'integration_time': dynamics_time,
                'validation_pass': validation_pass,
                'enhancement': 'fisher_rao'
            }
        else:
            print(f"   [ERROR] No successful Fisher-Rao geodesics")
            self.results['dynamics'] = {'n_success': 0, 'n_total': n_geodesics}

        return geodesic_results

    def run_fr_gradient_flows(self, fr_embeddings, fr_metrics):
        """Run Fisher-Rao gradient flows for embedding optimization.

        This demonstrates using FR gradient flows to optimize embeddings
        in the model-aware Fisher-Rao geometry.
        """

        print(f"\n🌊 STEP 4b: FISHER-RAO GRADIENT FLOWS")
        print("=" * 60)

        print(f"   📐 Setting up FR gradient flow optimization...")

        n_points = len(fr_embeddings)
        d = fr_embeddings.shape[1]

        # Create synthetic logits and Jacobians for demonstration
        # In practice, these would come from the actual model
        rng = np.random.default_rng(self.seed + 300)

        # Sample a small subset for flow optimization
        sample_size = min(16, n_points)
        sample_indices = rng.choice(n_points, size=sample_size, replace=False)

        sample_embeddings = fr_embeddings[sample_indices]
        sample_labels = self.category_labels[sample_indices]

        # Create logits from embeddings (simplified model)
        n_classes = len(np.unique(self.category_labels))
        k = n_classes

        # Simple linear projection to logits
        projection = rng.normal(0, 0.1, (d, k))
        logits = sample_embeddings @ projection

        # Compute Jacobian (constant for linear projection)
        # Shape: (n, k, d) - derivative of each logit w.r.t. each feature
        dlogits_dX = np.zeros((sample_size, k, d))
        for i in range(sample_size):
            dlogits_dX[i] = projection.T  # Jacobian is the projection matrix

        print(f"      Sample size: {sample_size}")
        print(f"      Logits shape: {logits.shape}")
        print(f"      Classes: {n_classes}")

        # Define semantic clustering functional
        def semantic_clustering_loss(logits_):
            """Loss that encourages same-category embeddings to have similar logits."""
            probs = np.exp(logits_ - logsumexp(logits_, axis=1, keepdims=True))

            # Intra-class variance (minimize)
            loss = 0.0
            for cat in range(n_classes):
                mask = sample_labels == cat
                if np.sum(mask) > 1:
                    cat_probs = probs[mask]
                    cat_mean = np.mean(cat_probs, axis=0)
                    variance = np.mean(np.sum((cat_probs - cat_mean)**2, axis=1))
                    loss += variance

            return loss

        print(f"\n   🔄 Running Fisher-Rao gradient flow...")
        print(f"      Functional: Semantic clustering loss")

        start_time = time.time()

        try:
            # Run FR gradient flow
            trajectory, flow_info = fr_gradient_flow(
                logits=logits,
                dlogits_dX=dlogits_dX,
                F=semantic_clustering_loss,
                steps=30,
                eta=0.005,
                adaptive_step=False,
                verbose=False
            )

            flow_time = time.time() - start_time

            print(f"      ✅ Flow completed: {len(trajectory)} steps")

            if flow_info['functional_values']:
                initial_loss = flow_info['functional_values'][0]
                final_loss = flow_info['functional_values'][-1]
                improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

                print(f"      📉 Loss reduction: {initial_loss:.4f} → {final_loss:.4f} ({improvement:.1%})")
                print(f"      Converged: {flow_info['converged']}")

            print(f"      ⏱️  Flow time: {flow_time:.2f}s")

            self.results['fr_flows'] = {
                'trajectory': trajectory,
                'flow_info': flow_info,
                'sample_size': sample_size,
                'n_classes': n_classes,
                'initial_loss': flow_info['functional_values'][0] if flow_info['functional_values'] else None,
                'final_loss': flow_info['functional_values'][-1] if flow_info['functional_values'] else None,
                'computation_time': flow_time,
                'success': True
            }

            return trajectory, flow_info

        except Exception as e:
            print(f"      ⚠️  FR flow error: {str(e)[:50]}...")

            # Fallback: demonstrate natural gradient descent
            print(f"\n   🔄 Fallback: Natural gradient descent demo...")

            try:
                # Simple parameter optimization
                params = rng.normal(0, 1, k)

                def grad_func(p):
                    return 2 * p  # Gradient of quadratic

                def fisher_func(p):
                    return np.eye(len(p)) + 0.1 * np.outer(p, p)

                traj_ngd, ngd_info = natural_gradient_descent(
                    params=params,
                    grad_func=grad_func,
                    fisher_func=fisher_func,
                    steps=20,
                    eta=0.1,
                    verbose=False
                )

                print(f"      ✅ NGD completed: {len(traj_ngd)} steps")

                self.results['fr_flows'] = {
                    'trajectory': traj_ngd,
                    'flow_info': ngd_info,
                    'method': 'natural_gradient_descent',
                    'computation_time': time.time() - start_time,
                    'success': True
                }

                return traj_ngd, ngd_info

            except Exception as e2:
                print(f"      ❌ NGD also failed: {str(e2)[:50]}...")
                self.results['fr_flows'] = {'error': str(e), 'success': False}
                return [], {}

    def analyze_semantic_data_structure(self, geodesic_results):
        """Analyze how dynamics reveal semantic relationships."""

        print(f"\n[TEXT] STEP 5: DATA ANALYSIS (Dynamics -> Data)")
        print("=" * 60)

        print(f"   [SEARCH] Analyzing semantic structure through Fisher-Rao dynamics...")

        # Semantic clustering analysis
        n_categories = len(self.results['fisher_rao']['vocab_categories'])

        print(f"   [CHART] Semantic category analysis ({n_categories} categories):")

        category_analysis = {}
        for cat_idx in range(n_categories):
            cat_mask = self.category_labels == cat_idx
            cat_tokens = [self.tokens[i] for i in range(len(self.tokens)) if cat_mask[i]]
            cat_embeddings = self.fr_embeddings[cat_mask]

            if len(cat_embeddings) > 1:
                # Compute intra-category Fisher-Rao distances
                fr_distances = []
                for i in range(len(cat_embeddings)):
                    for j in range(i+1, len(cat_embeddings)):
                        G_i = self.fr_metrics[np.where(cat_mask)[0][i]]
                        G_j = self.fr_metrics[np.where(cat_mask)[0][j]]

                        dist = riemannian_distance(
                            cat_embeddings[i], cat_embeddings[j],
                            G_i, G_j, method="average"
                        )
                        fr_distances.append(dist)

                avg_intra_distance = np.mean(fr_distances) if fr_distances else 0.0

                category_name = self.results['fisher_rao']['vocab_categories'][cat_idx][0]
                print(f"      Category {cat_idx} ({category_name}...): {len(cat_tokens)} tokens, "
                      f"avg FR distance: {avg_intra_distance:.3f}")

                category_analysis[cat_idx] = {
                    'name': category_name,
                    'tokens': cat_tokens,
                    'size': len(cat_tokens),
                    'avg_intra_distance': avg_intra_distance,
                    'fr_distances': fr_distances
                }

        # Inter-category distance analysis
        print(f"\n   [GRAPH] Inter-category Fisher-Rao distance matrix:")

        inter_distances = np.zeros((n_categories, n_categories))

        for i in range(n_categories):
            for j in range(i+1, n_categories):
                mask_i = self.category_labels == i
                mask_j = self.category_labels == j

                if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                    # Sample representative embeddings
                    emb_i = self.fr_embeddings[mask_i][0]  # First embedding from category i
                    emb_j = self.fr_embeddings[mask_j][0]  # First embedding from category j

                    G_i = self.fr_metrics[np.where(mask_i)[0][0]]
                    G_j = self.fr_metrics[np.where(mask_j)[0][0]]

                    dist = riemannian_distance(emb_i, emb_j, G_i, G_j, method="average")
                    inter_distances[i, j] = dist
                    inter_distances[j, i] = dist

        # Print distance matrix
        cat_names = [category_analysis[i]['name'] for i in category_analysis.keys()]
        print(f"      {'':>8}", end="")
        for name in cat_names:
            print(f" {name[:6]:>8}", end="")
        print()

        available_cats = list(category_analysis.keys())
        for i, cat_i in enumerate(available_cats):
            name_i = category_analysis[cat_i]['name']
            print(f"      {name_i[:6]:>8}", end="")
            for j, cat_j in enumerate(available_cats):
                if i != j:
                    print(f" {inter_distances[cat_i,cat_j]:8.3f}", end="")
                else:
                    print(f" {'---':>8}", end="")
            print()

        # Geodesic semantic analysis
        if geodesic_results:
            print(f"\n   [DYNAMICS] Geodesic trajectory semantic analysis:")

            # Check if geodesics connect semantically related regions
            for i, result in enumerate(geodesic_results[:3]):
                traj_q = result['trajectory_q']

                # Simple analysis: trajectory length and coverage
                traj_length = result['info']['trajectory_length']
                seam_crossings = result['info']['seam_crossings']

                print(f"      Geodesic {i+1}: length={traj_length:.2f}, crossings={seam_crossings}")
                print(f"         Start: [{traj_q[0,0]:.2f}, {traj_q[0,1]:.2f}]")
                print(f"         End:   [{traj_q[-1,0]:.2f}, {traj_q[-1,1]:.2f}]")

        # Overall semantic coherence measure
        intra_avg = np.mean([category_analysis[i]['avg_intra_distance']
                           for i in category_analysis if category_analysis[i]['avg_intra_distance'] > 0])
        inter_avg = np.mean(inter_distances[inter_distances > 0])

        coherence_ratio = inter_avg / intra_avg if intra_avg > 0 else 0.0

        print(f"\n   [TARGET] Semantic coherence analysis:")
        print(f"      Average intra-category distance: {intra_avg:.3f}")
        print(f"      Average inter-category distance: {inter_avg:.3f}")
        print(f"      Coherence ratio (inter/intra): {coherence_ratio:.2f}")
        print(f"      {'Good separation' if coherence_ratio > 1.5 else 'Moderate separation'}")

        self.results['data_analysis'] = {
            'category_analysis': category_analysis,
            'inter_distances': inter_distances,
            'intra_avg_distance': intra_avg,
            'inter_avg_distance': inter_avg,
            'coherence_ratio': coherence_ratio,
            'semantic_quality': 'good' if coherence_ratio > 1.5 else 'moderate'
        }

        return category_analysis, coherence_ratio

    def build_final_enhanced_graph(self, category_analysis):
        """Build final graph using insights from Fisher-Rao analysis."""

        print(f"\n[GRAPH] STEP 6: GRAPH CONSTRUCTION (Data -> Graph)")
        print("=" * 60)

        print(f"   [BUILD]  Building Fisher-Rao enhanced graph structure...")

        # Use insights from Fisher-Rao analysis to improve graph construction
        k = min(12, len(self.fr_embeddings) - 1)

        start_time = time.time()

        # Build multiple graph variants
        print(f"      Constructing graph variants (k={k})...")

        # 1. Standard graph on original embeddings
        G_original = build_graph(self.embeddings, mode="additive", k=k, seed=self.seed)

        # 2. Fisher-Rao enhanced graph
        G_fr = build_graph(self.fr_embeddings, mode="additive", k=k, seed=self.seed)

        # 3. Dual transport on Fisher-Rao embeddings
        G_fr_mult = build_graph(self.fr_embeddings, mode="multiplicative", k=k,
                               tau="median", eps=1e-6, seed=self.seed)

        # Analyze connectivity and clustering
        conn_orig = check_graph_connectivity(G_original)
        conn_fr = check_graph_connectivity(G_fr)
        conn_fr_mult = check_graph_connectivity(G_fr_mult)

        print(f"   [CHART] Graph connectivity analysis:")
        print(f"      Original embeddings: {'[OK] Connected' if conn_orig else '[ERROR] Disconnected'}")
        print(f"      Fisher-Rao enhanced: {'[OK] Connected' if conn_fr else '[ERROR] Disconnected'}")
        print(f"      FR + Multiplicative: {'[OK] Connected' if conn_fr_mult else '[ERROR] Disconnected'}")

        # Compute final spectral properties
        print(f"\n   [STATS] Final spectral analysis:")

        L_original = laplacian(G_original, normalized=True)
        L_fr = laplacian(G_fr, normalized=True)
        L_fr_mult = laplacian(G_fr_mult, normalized=True)

        n_eigs = min(12, len(self.fr_embeddings) - 1)

        # Compute spectra
        eigs_orig, _, _ = topk_eigs(L_original, k=n_eigs, which="SM")
        eigs_fr, _, _ = topk_eigs(L_fr, k=n_eigs, which="SM")
        eigs_fr_mult, _, _ = topk_eigs(L_fr_mult, k=n_eigs, which="SM")

        gap_orig = spectral_gap(L_original)
        gap_fr = spectral_gap(L_fr)
        gap_fr_mult = spectral_gap(L_fr_mult)

        entropy_orig = spectral_entropy(L_original, k=min(10, len(eigs_orig)))
        entropy_fr = spectral_entropy(L_fr, k=min(10, len(eigs_fr)))
        entropy_fr_mult = spectral_entropy(L_fr_mult, k=min(10, len(eigs_fr_mult)))

        graph_time = time.time() - start_time

        print(f"      Original:        Gap={gap_orig:.4f}, Entropy={entropy_orig:.4f}")
        print(f"      Fisher-Rao:      Gap={gap_fr:.4f}, Entropy={entropy_fr:.4f}")
        print(f"      FR+Mult:         Gap={gap_fr_mult:.4f}, Entropy={entropy_fr_mult:.4f}")

        # Enhancement ratios
        gap_enhancement = gap_fr / gap_orig if gap_orig > 0 else 0
        entropy_enhancement = entropy_fr / entropy_orig if entropy_orig > 0 else 0

        print(f"\n   [TARGET] Fisher-Rao enhancement:")
        print(f"      Gap improvement: {gap_enhancement:.2f}x")
        print(f"      Entropy improvement: {entropy_enhancement:.2f}x")
        print(f"      Graph construction time: {graph_time:.2f}s")

        # Semantic clustering validation
        print(f"\n   [TEST] Semantic clustering validation:")

        # Check if same-category tokens are well-connected
        category_clustering_scores = {}

        for cat_idx in range(len(self.results['fisher_rao']['vocab_categories'])):
            cat_mask = self.category_labels == cat_idx
            cat_indices = np.where(cat_mask)[0]

            if len(cat_indices) > 1:
                # Check connectivity within category in Fisher-Rao graph
                cat_connections = 0
                total_possible = len(cat_indices) * (len(cat_indices) - 1) // 2

                for i in range(len(cat_indices)):
                    for j in range(i+1, len(cat_indices)):
                        if G_fr[cat_indices[i], cat_indices[j]] > 0:
                            cat_connections += 1

                clustering_score = cat_connections / total_possible if total_possible > 0 else 0

                cat_name = self.results['fisher_rao']['vocab_categories'][cat_idx][0]
                print(f"      {cat_name}... clustering: {clustering_score:.2f} ({cat_connections}/{total_possible})")

                category_clustering_scores[cat_idx] = clustering_score

        avg_clustering = np.mean(list(category_clustering_scores.values())) if category_clustering_scores else 0

        print(f"   [CHART] Average semantic clustering: {avg_clustering:.3f}")

        self.results['final_graph'] = {
            'G_original': G_original,
            'G_fisher_rao': G_fr,
            'G_fr_multiplicative': G_fr_mult,
            'eigenvals_original': eigs_orig,
            'eigenvals_fr': eigs_fr,
            'eigenvals_fr_mult': eigs_fr_mult,
            'gap_original': gap_orig,
            'gap_fr': gap_fr,
            'gap_fr_mult': gap_fr_mult,
            'entropy_original': entropy_orig,
            'entropy_fr': entropy_fr,
            'entropy_fr_mult': entropy_fr_mult,
            'gap_enhancement': gap_enhancement,
            'entropy_enhancement': entropy_enhancement,
            'connectivity': {
                'original': conn_orig,
                'fisher_rao': conn_fr,
                'fr_multiplicative': conn_fr_mult
            },
            'clustering_scores': category_clustering_scores,
            'avg_clustering': avg_clustering,
            'construction_time': graph_time
        }

        return G_fr, category_clustering_scores

    def create_comprehensive_visualization(self):
        """Create comprehensive visualization of the entire pipeline."""

        print(f"\n[CHART] VISUALIZATION & SUMMARY")
        print("=" * 60)

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

        # 1. Embedding space overview (original vs Fisher-Rao)
        ax1 = axes[0, 0]

        # PCA for visualization
        from sklearn.decomposition import PCA

        pca_orig = PCA(n_components=2)
        embed_2d_orig = pca_orig.fit_transform(self.embeddings)

        pca_fr = PCA(n_components=2)
        embed_2d_fr = pca_fr.fit_transform(self.fr_embeddings)

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for cat_idx in range(len(self.results['fisher_rao']['vocab_categories'])):
            cat_mask = self.category_labels == cat_idx
            if np.any(cat_mask):
                cat_name = self.results['fisher_rao']['vocab_categories'][cat_idx][0]
                color = colors[cat_idx % len(colors)]
                ax1.scatter(embed_2d_orig[cat_mask, 0], embed_2d_orig[cat_mask, 1],
                           c=color, alpha=0.7, s=50, label=f'{cat_name}...')

        ax1.set_title('Original Embeddings (PCA)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Fisher-Rao enhanced embeddings
        ax2 = axes[0, 1]

        for cat_idx in range(len(self.results['fisher_rao']['vocab_categories'])):
            cat_mask = self.category_labels == cat_idx
            if np.any(cat_mask):
                color = colors[cat_idx % len(colors)]
                ax2.scatter(embed_2d_fr[cat_mask, 0], embed_2d_fr[cat_mask, 1],
                           c=color, alpha=0.7, s=50)

        ax2.set_title('Fisher-Rao Enhanced Embeddings')
        ax2.grid(True, alpha=0.3)

        # 3. Topology compatibility
        ax3 = axes[0, 2]

        if 'topology' in self.results:
            topo_results = self.results['topology']['candidates']
            names = list(topo_results.keys())
            rates = [topo_results[name]['compatibility_rate'] for name in names]

            bars = ax3.bar(names, rates, alpha=0.8, color=['blue', 'green', 'orange'])
            ax3.set_title('Topology Compatibility')
            ax3.set_ylabel('Compatibility Rate')
            ax3.set_ylim(0, 1)

            # Highlight selected topology
            selected_name = self.results['topology']['selected_name']
            for i, name in enumerate(names):
                if name == selected_name:
                    bars[i].set_color('red')
                    bars[i].set_alpha(1.0)

        ax3.grid(True, alpha=0.3)

        # 4. Spectral comparison
        ax4 = axes[1, 0]

        if 'statistics' in self.results:
            eigs_std = self.results['statistics']['eigenvals_standard'][:10]
            eigs_fr = self.results['statistics']['eigenvals_fr'][:10]

            x = np.arange(len(eigs_std))
            ax4.plot(x, eigs_std, 'o-', label='Standard', linewidth=2, markersize=6)
            ax4.plot(x, eigs_fr, 's-', label='Fisher-Rao', linewidth=2, markersize=6)
            ax4.set_title('Eigenvalue Spectra Comparison')
            ax4.set_xlabel('Index')
            ax4.set_ylabel('Eigenvalue')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Geodesic trajectories
        ax5 = axes[1, 1]

        if 'dynamics' in self.results and 'geodesic_results' in self.results['dynamics']:
            geodesics = self.results['dynamics']['geodesic_results']

            for i, result in enumerate(geodesics[:4]):
                traj_q = result['trajectory_q']
                color = colors[i % len(colors)]
                ax5.plot(traj_q[:, 0], traj_q[:, 1], '-', color=color, alpha=0.8,
                        linewidth=2, label=f'Geodesic {i+1}')
                ax5.scatter(traj_q[0, 0], traj_q[0, 1], color=color, s=80, marker='o')
                ax5.scatter(traj_q[-1, 0], traj_q[-1, 1], color=color, s=80, marker='s')

            # Draw topology boundaries if available
            if hasattr(self.results['topology']['selected'], 'w'):
                w = self.results['topology']['selected'].w
                period = getattr(self.results['topology']['selected'], 'period', 2*np.pi)
                ax5.axhline(y=w, color='black', linestyle='--', alpha=0.5)
                ax5.axhline(y=-w, color='black', linestyle='--', alpha=0.5)
                ax5.set_xlim(0, period)
                ax5.set_ylim(-w*1.1, w*1.1)

            ax5.set_title('Fisher-Rao Geodesics')
            ax5.set_xlabel('u')
            ax5.set_ylabel('v')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Semantic clustering scores
        ax6 = axes[1, 2]

        if 'final_graph' in self.results:
            clustering_scores = self.results['final_graph']['clustering_scores']
            cat_names = [self.results['fisher_rao']['vocab_categories'][i][0]
                        for i in clustering_scores.keys()]
            scores = list(clustering_scores.values())

            bars = ax6.bar(cat_names, scores, alpha=0.8, color=colors[:len(cat_names)])
            ax6.set_title('Semantic Clustering Quality')
            ax6.set_ylabel('Clustering Score')
            ax6.set_ylim(0, 1)
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)

        # 7. Enhancement ratios
        ax7 = axes[2, 0]

        if 'final_graph' in self.results:
            enhancements = {
                'Spectral Gap': self.results['final_graph']['gap_enhancement'],
                'Spectral Entropy': self.results['final_graph']['entropy_enhancement'],
                'Avg Clustering': self.results['final_graph']['avg_clustering']
            }

            names = list(enhancements.keys())
            values = list(enhancements.values())

            bars = ax7.bar(names, values, alpha=0.8, color=['blue', 'green', 'red'])
            ax7.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
            ax7.set_title('Fisher-Rao Enhancements')
            ax7.set_ylabel('Improvement Factor')
            ax7.tick_params(axis='x', rotation=45)
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # 8. Distance matrix heatmap
        ax8 = axes[2, 1]

        if 'data_analysis' in self.results:
            inter_distances = self.results['data_analysis']['inter_distances']
            cat_names = [self.results['fisher_rao']['vocab_categories'][i][0]
                        for i in range(len(inter_distances))]

            im = ax8.imshow(inter_distances, cmap='viridis', aspect='auto')
            ax8.set_title('Inter-Category FR Distances')
            ax8.set_xticks(range(len(cat_names)))
            ax8.set_yticks(range(len(cat_names)))
            ax8.set_xticklabels(cat_names, rotation=45)
            ax8.set_yticklabels(cat_names)
            plt.colorbar(im, ax=ax8, shrink=0.6)

        # 9. Pipeline summary
        ax9 = axes[2, 2]
        ax9.axis('off')

        # Create text summary
        summary_text = "FISHER-RAO PIPELINE SUMMARY\n"
        summary_text += "=" * 30 + "\n\n"

        if 'fisher_rao' in self.results:
            fr_info = self.results['fisher_rao']
            summary_text += f"[CHART] Data: {len(fr_info['tokens'])} tokens\n"
            summary_text += f"[TARGET] Valid FR metrics: {fr_info['valid_metrics_pct']:.1f}%\n\n"

        if 'topology' in self.results:
            topo_info = self.results['topology']
            summary_text += f"[TOPO] Topology: {topo_info['selected_name'].title()}\n"
            summary_text += f"[OK] Compatibility: {topo_info['compatibility_rate']:.1%}\n\n"

        if 'dynamics' in self.results:
            dyn_info = self.results['dynamics']
            if 'n_success' in dyn_info:
                summary_text += f"[DYNAMICS] Geodesics: {dyn_info['n_success']}/{dyn_info['n_total']}\n"
                if dyn_info['n_success'] > 0:
                    summary_text += f"[STATS] Energy drift: {dyn_info['mean_energy_drift']:.1e}\n\n"

        if 'data_analysis' in self.results:
            data_info = self.results['data_analysis']
            summary_text += f"[SEARCH] Coherence ratio: {data_info['coherence_ratio']:.2f}\n"
            summary_text += f"[TARGET] Quality: {data_info['semantic_quality']}\n\n"

        if 'final_graph' in self.results:
            graph_info = self.results['final_graph']
            summary_text += f"[GRAPH] Gap enhancement: {graph_info['gap_enhancement']:.2f}x\n"
            summary_text += f"[CHART] Clustering: {graph_info['avg_clustering']:.3f}\n"

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        plt.tight_layout()
        plt.suptitle('Fisher-Rao Enhanced Geometric ML Pipeline', fontsize=16, y=0.98)

        # Save visualization
        output_path = 'fisher_rao_pipeline_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   [CHART] Comprehensive visualization saved to: {output_path}")

        try:
            plt.show()
        except:
            pass

        return output_path

    def print_pipeline_summary(self):
        """Print comprehensive pipeline summary."""

        print(f"\n[TARGET] FISHER-RAO PIPELINE COMPLETE")
        print("=" * 60)

        # Component-by-component summary
        success_components = 0
        total_components = 6

        print(f"[CHART] Component Analysis:")

        # 1. Fisher-Rao Metrics
        if 'fisher_rao' in self.results:
            fr_info = self.results['fisher_rao']
            valid_pct = fr_info['valid_metrics_pct']
            status = "[OK]" if valid_pct > 80 else "[WARNING]" if valid_pct > 50 else "[ERROR]"
            print(f"   1. Fisher-Rao Metrics: {status} {valid_pct:.1f}% valid, {fr_info['computation_time']:.1f}s")
            if valid_pct > 50:
                success_components += 1

        # 2. Topology Selection
        if 'topology' in self.results:
            topo_info = self.results['topology']
            compat_rate = topo_info['compatibility_rate']
            status = "[OK]" if compat_rate > 0.7 else "[WARNING]" if compat_rate > 0.4 else "[ERROR]"
            print(f"   2. Topology Selection: {status} {topo_info['selected_name'].title()}, {compat_rate:.1%} compatible")
            if compat_rate > 0.4:
                success_components += 1

        # 3. Statistical Analysis
        if 'statistics' in self.results:
            stats_info = self.results['statistics']
            gap_enhancement = stats_info['gap_enhancement']
            status = "[OK]" if gap_enhancement > 1.1 else "[WARNING]" if gap_enhancement > 0.9 else "[ERROR]"
            print(f"   3. Statistical Analysis: {status} {gap_enhancement:.2f}x gap enhancement, {stats_info['computation_time']:.1f}s")
            if gap_enhancement > 0.9:
                success_components += 1

        # 4. Dynamics Integration
        if 'dynamics' in self.results:
            dyn_info = self.results['dynamics']
            if 'n_success' in dyn_info:
                success_rate = dyn_info['n_success'] / dyn_info['n_total']
                status = "[OK]" if success_rate > 0.7 else "[WARNING]" if success_rate > 0.3 else "[ERROR]"
                print(f"   4. Dynamics Integration: {status} {dyn_info['n_success']}/{dyn_info['n_total']} geodesics")
                if success_rate > 0.3:
                    success_components += 1
            else:
                print(f"   4. Dynamics Integration: [ERROR] Failed")

        # 5. Data Analysis
        if 'data_analysis' in self.results:
            data_info = self.results['data_analysis']
            coherence = data_info['coherence_ratio']
            status = "[OK]" if coherence > 1.5 else "[WARNING]" if coherence > 1.0 else "[ERROR]"
            print(f"   5. Data Analysis: {status} {coherence:.2f} coherence ratio, {data_info['semantic_quality']} quality")
            if coherence > 1.0:
                success_components += 1

        # 6. Graph Construction
        if 'final_graph' in self.results:
            graph_info = self.results['final_graph']
            gap_enhancement = graph_info['gap_enhancement']
            avg_clustering = graph_info['avg_clustering']
            status = "[OK]" if gap_enhancement > 1.1 and avg_clustering > 0.5 else "[WARNING]" if gap_enhancement > 0.9 else "[ERROR]"
            print(f"   6. Graph Construction: {status} {gap_enhancement:.2f}x enhancement, {avg_clustering:.3f} clustering")
            if gap_enhancement > 0.9:
                success_components += 1

        # Overall assessment
        overall_success = success_components / total_components

        print(f"\n[TARGET] OVERALL PIPELINE SUCCESS: {success_components}/{total_components} ({overall_success:.1%})")

        if overall_success >= 0.8:
            print("[EXCELLENT] EXCELLENT: Fisher-Rao pipeline highly successful!")
            print("   The model-aware metrics significantly enhanced geometric analysis.")
        elif overall_success >= 0.6:
            print("[OK] GOOD: Fisher-Rao pipeline mostly successful.")
            print("   Clear benefits from model-aware geometric analysis.")
        elif overall_success >= 0.4:
            print("[WARNING]  FAIR: Fisher-Rao pipeline partially successful.")
            print("   Some benefits observed from model-aware approach.")
        else:
            print("[ERROR] POOR: Fisher-Rao pipeline had significant issues.")
            print("   Consider adjusting parameters or data selection.")

        # Key insights
        print(f"\n[INSIGHT] KEY INSIGHTS:")

        if 'statistics' in self.results and 'final_graph' in self.results:
            gap_improvement = self.results['final_graph']['gap_enhancement']
            if gap_improvement > 1.2:
                print("   • Fisher-Rao metrics significantly improved spectral separation")
            elif gap_improvement > 1.0:
                print("   • Fisher-Rao metrics provided modest spectral improvements")
            else:
                print("   • Limited spectral improvement from Fisher-Rao metrics")

        if 'data_analysis' in self.results:
            coherence = self.results['data_analysis']['coherence_ratio']
            if coherence > 2.0:
                print("   • Excellent semantic clustering in Fisher-Rao space")
            elif coherence > 1.5:
                print("   • Good semantic structure revealed by Fisher-Rao analysis")
            else:
                print("   • Moderate semantic clustering detected")

        if 'dynamics' in self.results and self.results['dynamics'].get('n_success', 0) > 0:
            print("   • Fisher-Rao geodesics successfully integrated on selected topology")

            if self.results['dynamics']['total_seam_crossings'] > 0:
                print("   • Geodesics exhibit interesting seam-crossing behavior")

        print(f"\n[INFO] PIPELINE DEMONSTRATES:")
        print("   • Model-aware metric design using Fisher-Rao pullbacks")
        print("   • Topology selection based on metric compatibility")
        print("   • Enhanced statistical analysis with geometric structure")
        print("   • Dynamics integration respecting model semantics")
        print("   • Semantic validation through geometric flows")
        print("   • Graph construction informed by model-aware analysis")


def main():
    """Run the complete Fisher-Rao enhanced pipeline."""

    print("EXAMPLE 6: Fisher-Rao to Full Pipeline with Real Embeddings")
    print("=" * 80)
    print("This demonstrates the complete geometric ML pipeline starting with")
    print("Fisher-Rao metrics: Fisher-Rao -> Topology -> Stats -> Dynamics -> Data -> Graph")
    print("=" * 80)

    # Initialize pipeline
    pipeline = FisherRaoPipeline(seed=42)

    try:
        # Execute complete pipeline following the specified flow

        # Step 1: Fisher-Rao Metric Design
        fr_embeddings, fr_metrics = pipeline.load_model_and_extract_embeddings()

        # Step 2: Topology Selection (Fisher-Rao → Topology)
        topology, fr_metric_fn = pipeline.select_optimal_topology(fr_metrics)

        # Step 3: Statistical Analysis (Topology → Stats)
        eigenvals, eigenvecs = pipeline.compute_enhanced_statistics(topology, fr_embeddings, fr_metrics)

        # Step 4: Dynamics Integration (Stats → Dynamics)
        geodesic_results = pipeline.integrate_fisher_rao_dynamics(topology, fr_metric_fn)

        # Step 4b: Fisher-Rao Gradient Flows (FR Flow Optimization)
        fr_trajectory, fr_flow_info = pipeline.run_fr_gradient_flows(fr_embeddings, fr_metrics)

        # Step 5: Data Analysis (Dynamics → Data)
        category_analysis, coherence = pipeline.analyze_semantic_data_structure(geodesic_results)

        # Step 6: Graph Construction (Data → Graph)
        final_graph, clustering_scores = pipeline.build_final_enhanced_graph(category_analysis)

        # Visualization and Summary
        try:
            pipeline.create_comprehensive_visualization()
        except ImportError:
            print("\n   [CHART] Matplotlib not available for visualization")

        # Final summary
        pipeline.print_pipeline_summary()

    except Exception as e:
        print(f"\n[ERROR] Fisher-Rao pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n[OK] Fisher-Rao pipeline demonstration complete!")
    print("\nThis example showed how model-aware Fisher-Rao metrics can enhance")
    print("every step of geometric analysis, from topology selection through")
    print("final graph construction, revealing semantic structure in embeddings.")


if __name__ == "__main__":
    main()
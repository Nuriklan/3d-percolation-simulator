"""
Unit and integration tests for Phase 8: Tunneling Contact Conductance.

Covers:
- TunnelingConfig validation, exponential conductance/resistance formulas, d_min clamp,
  effective_cutoff with cutoff_distance & min_conductance, Simmons barrier factory,
  and dict serialization.
- Contact backward compatibility and vectorized tunneling enrichment & statistics.
- find_contacts with tunneling across all spatial backends (brute_force, grid, hash, kdtree, auto).
- find_electrode_contacts and build_weighted_graph.
- Weighted percolation algorithms:
    * find_min_resistance_path (Dijkstra vs BFS hop-count path)
    * find_bottleneck_conductance (Ambegaokar-Halperin-Langer critical path)
    * check_conductance_percolation & filter_graph_by_conductance
- Simulation pipeline integration: create_sample, run_sample, run_concentration,
  run_monte_carlo, run_from_config (YAML) with tunneling enabled & disabled,
  JSON/CSV serialization, and reproducibility.
- Analysis module: analyze_sample_tunneling, run_tunneling_alpha_study,
  run_contact_cutoff_study, and serialization.
- Visualization: plot_tunneling_study, plot_contact_conductance_distribution,
  and plot_sample_3d with conductance coloring & min-resistance path highlighting.
- CLI smoke tests.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from percolation.analysis.tunneling import (
    TunnelingStudyResult,
    analyze_sample_tunneling,
    run_contact_cutoff_study,
    run_tunneling_alpha_study,
)
from percolation.domain.cnt import Contact, MaterialBox, Nanotube
from percolation.domain.tunneling import (
    LANDAUER_CONDUCTANCE_S,
    TunnelingConfig,
    apply_tunneling_to_contacts,
    compute_contact_conductance,
    simmons_decay_constant_nm,
    summarize_contact_conductance,
)
from percolation.generation.cnt_generator import generate_nanotubes
from percolation.graph.contact_graph import (
    LEFT,
    RIGHT,
    build_weighted_graph,
    find_contacts,
    find_electrode_contacts,
)
from percolation.graph.percolation import (
    check_conductance_percolation,
    filter_graph_by_conductance,
    find_bottleneck_conductance,
    find_min_resistance_path,
    find_percolating_path,
    path_series_resistance,
)
from percolation.simulation.experiment import create_sample, run_sample
from percolation.simulation.monte_carlo import run_from_config
from percolation.visualization.renderer import plot_sample_3d
from percolation.visualization.tunneling_plot import (
    plot_contact_conductance_distribution,
    plot_tunneling_study,
)

matplotlib.use("Agg")

TEST_BOX = MaterialBox(size_x=2.0, size_y=1.0, size_z=1.0)


def _tube_x(x0: float, x1: float, y: float = 0.5, z: float = 0.5) -> Nanotube:
    return Nanotube(
        center=np.array([(x0 + x1) / 2.0, y, z]),
        direction=np.array([1.0, 0.0, 0.0]),
        length=abs(x1 - x0),
        diameter=0.05,
    )


# ---------------------------------------------------------------------------
# 1. TunnelingConfig physics & validation
# ---------------------------------------------------------------------------

class TestTunnelingConfig:
    def test_exponential_law_and_inverse_resistance(self):
        cfg = TunnelingConfig(g0=2.5, alpha=40.0, d_min=0.0)
        assert cfg.r0 == pytest.approx(0.4)
        for d in [0.0, 0.01, 0.05, 0.1]:
            g = cfg.conductance(d)
            r = cfg.resistance(d)
            assert g == pytest.approx(2.5 * math.exp(-40.0 * d))
            assert r == pytest.approx(0.4 * math.exp(40.0 * d))
            assert g * r == pytest.approx(1.0, rel=1e-12)

    def test_d_min_clamping(self):
        cfg = TunnelingConfig(g0=1.0, alpha=100.0, d_min=0.02)
        assert cfg.conductance(0.0) == pytest.approx(1.0)
        assert cfg.conductance(0.01) == pytest.approx(1.0)
        assert cfg.conductance(0.02) == pytest.approx(1.0)
        assert cfg.conductance(0.05) == pytest.approx(math.exp(-100.0 * 0.03))
        assert cfg.resistance(0.01) == pytest.approx(1.0)

    def test_vectorized_matches_scalar(self):
        cfg = TunnelingConfig(g0=3.2, alpha=75.0, d_min=0.005)
        dists = np.array([0.0, 0.003, 0.005, 0.02, 0.05, 0.1])
        g_vec = cfg.conductance_array(dists)
        r_vec = cfg.resistance_array(dists)
        for idx, d in enumerate(dists):
            assert g_vec[idx] == pytest.approx(cfg.conductance(float(d)))
            assert r_vec[idx] == pytest.approx(cfg.resistance(float(d)))

    def test_extreme_exponent_safe_no_overflow(self):
        cfg = TunnelingConfig(g0=1.0, alpha=1e6)
        assert cfg.conductance(1.0) == 0.0
        assert math.isinf(cfg.resistance(1.0))
        g_arr = cfg.conductance_array([0.0, 1.0])
        r_arr = cfg.resistance_array([0.0, 1.0])
        assert g_arr[0] == pytest.approx(1.0)
        assert g_arr[1] == 0.0
        assert math.isinf(r_arr[1])

    def test_effective_cutoff_with_min_conductance(self):
        # G(d) = exp(-50 * d) >= exp(-2) => d <= 0.04
        cfg = TunnelingConfig(g0=1.0, alpha=50.0, min_conductance=math.exp(-2.0))
        assert cfg.effective_cutoff(0.1) == pytest.approx(0.04, rel=1e-12)
        assert cfg.effective_cutoff(0.02) == pytest.approx(0.02, rel=1e-12)

        # min_conductance > g0 => impossible => negative cutoff
        cfg_impossible = TunnelingConfig(g0=1.0, alpha=50.0, min_conductance=2.0)
        assert cfg_impossible.effective_cutoff(0.1) < 0.0
        assert cfg_impossible.is_contact_active(0.0, 0.1) is False

    def test_from_resistance_and_simmons_barrier(self):
        cfg_r = TunnelingConfig.from_resistance(r0=500.0, alpha=25.0)
        assert cfg_r.g0 == pytest.approx(1.0 / 500.0)
        assert cfg_r.r0 == pytest.approx(500.0)

        alpha_1ev = simmons_decay_constant_nm(1.0)
        assert alpha_1ev == pytest.approx(10.2463, rel=1e-3)

        cfg_sim = TunnelingConfig.from_simmons_barrier(
            barrier_ev=1.0, length_unit_nm=1000.0, d_min_nm=0.34, cutoff_nm=1.4
        )
        assert cfg_sim.g0 == pytest.approx(LANDAUER_CONDUCTANCE_S)
        assert cfg_sim.alpha == pytest.approx(alpha_1ev * 1000.0)
        assert cfg_sim.d_min == pytest.approx(0.00034)
        assert cfg_sim.cutoff_distance == pytest.approx(0.0014)

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            TunnelingConfig(g0=0.0)
        with pytest.raises(ValueError):
            TunnelingConfig(alpha=-1.0)
        with pytest.raises(ValueError):
            TunnelingConfig(d_min=-0.01)
        with pytest.raises(ValueError):
            TunnelingConfig(cutoff_distance=-0.1)
        with pytest.raises(ValueError):
            TunnelingConfig(min_conductance=0.0)
        with pytest.raises(ValueError):
            simmons_decay_constant_nm(-0.5)

    def test_serialization_roundtrip(self):
        cfg = TunnelingConfig(
            enabled=True,
            g0=2.0,
            alpha=65.0,
            d_min=0.002,
            cutoff_distance=0.08,
            min_conductance=1e-4,
        )
        restored = TunnelingConfig.from_dict(cfg.to_dict())
        assert restored.enabled is True
        assert restored.g0 == pytest.approx(2.0)
        assert restored.alpha == pytest.approx(65.0)
        assert restored.d_min == pytest.approx(0.002)
        assert restored.cutoff_distance == pytest.approx(0.08)
        assert restored.min_conductance == pytest.approx(1e-4)

        assert TunnelingConfig.from_dict(None).enabled is False


# ---------------------------------------------------------------------------
# 2. Contact enrichment, stats & spatial backend equivalence
# ---------------------------------------------------------------------------

class TestContactEnrichmentAndSpatialEquivalence:
    def test_contact_backward_compatibility(self):
        c = Contact(tube_a=0, tube_b=1, distance=0.02)
        assert c.conductance is None
        assert c.resistance is None
        assert c.has_conductance is False
        assert c.to_dict() == {"tube_a": 0, "tube_b": 1, "distance": 0.02}

    def test_apply_tunneling_and_stats(self):
        cfg = TunnelingConfig(g0=1.0, alpha=50.0, min_conductance=0.1)
        raw = [
            Contact(0, 1, 0.0),    # G = 1.0 (kept)
            Contact(1, 2, 0.02),   # G = e^-1 ~ 0.3679 (kept)
            Contact(2, 3, 0.10),   # G = e^-5 ~ 0.0067 < 0.1 (filtered)
        ]
        enriched = apply_tunneling_to_contacts(raw, cfg)
        assert len(enriched) == 2
        assert enriched[0].has_conductance is True
        g0, r0 = compute_contact_conductance(0.02, cfg)
        assert enriched[1].conductance == pytest.approx(g0)
        assert enriched[1].resistance == pytest.approx(r0)

        stats = summarize_contact_conductance(enriched)
        assert stats.count == 2
        assert stats.max_conductance == pytest.approx(1.0)
        assert stats.geom_mean_conductance == pytest.approx(math.sqrt(1.0 * math.exp(-1.0)))

    def test_all_spatial_methods_match_with_tunneling(self):
        box = MaterialBox(size_x=5.0, size_y=3.0, size_z=3.0)
        rng = np.random.default_rng(77)
        tubes = generate_nanotubes(120, box, cnt_length=1.0, cnt_diameter=0.1, rng=rng)
        cfg = TunnelingConfig(g0=2.0, alpha=60.0, d_min=0.005, min_conductance=0.05)

        ref = find_contacts(tubes, threshold=0.1, box=box, method="brute_force", tunneling=cfg)
        assert len(ref) > 0
        for c in ref:
            assert c.conductance is not None and c.conductance >= 0.05 * (1.0 - 1e-12)
            assert c.resistance is not None and c.conductance * c.resistance == pytest.approx(1.0)

        for method in ("grid", "hash", "kdtree", "auto"):
            cand = find_contacts(tubes, threshold=0.1, box=box, method=method, tunneling=cfg)
            assert len(cand) == len(ref), f"Count mismatch for method={method}"
            for c_ref, c_m in zip(ref, cand):
                assert (c_ref.tube_a, c_ref.tube_b) == (c_m.tube_a, c_m.tube_b)
                assert c_m.distance == pytest.approx(c_ref.distance, abs=1e-10)
                assert c_m.conductance == pytest.approx(c_ref.conductance, rel=1e-10)
                assert c_m.resistance == pytest.approx(c_ref.resistance, rel=1e-10)


# ---------------------------------------------------------------------------
# 3. Weighted Graph & Path Algorithms (Dijkstra vs BFS, Bottleneck G_c)
# ---------------------------------------------------------------------------

class TestWeightedGraphAndPercolation:
    def test_electrode_contacts_and_weighted_graph(self):
        box = MaterialBox(size_x=10.0, size_y=5.0, size_z=5.0)
        t0 = _tube_x(0.02, 5.0, y=2.5, z=2.5)   # 0.02 from LEFT
        t1 = _tube_x(5.03, 9.99, y=2.5, z=2.5)  # 0.03 from t0, 0.01 from RIGHT
        cfg = TunnelingConfig(g0=1.0, alpha=50.0)

        contacts = find_contacts([t0, t1], threshold=0.05, tunneling=cfg)
        left_c, right_c = find_electrode_contacts([t0, t1], box, electrode_threshold=0.05, tunneling=cfg)
        assert len(left_c) == 1 and left_c[0].tube_idx == 0
        assert left_c[0].distance == pytest.approx(0.02)
        assert left_c[0].conductance == pytest.approx(math.exp(-50.0 * 0.02))
        assert len(right_c) == 1 and right_c[0].tube_idx == 1
        assert right_c[0].distance == pytest.approx(0.01)

        wg = build_weighted_graph([t0, t1], contacts, box, electrode_threshold=0.05, tunneling=cfg)
        assert wg[LEFT][0] == pytest.approx(math.exp(-1.0))
        assert wg[0][1] == pytest.approx(math.exp(-1.5))
        assert wg[1][RIGHT] == pytest.approx(math.exp(-0.5))
        # Symmetric
        assert wg[0][LEFT] == wg[LEFT][0]
        assert wg[1][0] == wg[0][1]

    def test_dijkstra_beats_bfs_on_high_resistance_shortcut(self):
        """Diamond topology:
        Upper path: LEFT --(G=0.01)-- 0 --(G=0.01)-- RIGHT  (2 edges, 1 tube, R=200)
        Lower path: LEFT --(G=1.0)-- 1 --(G=1.0)-- 2 --(G=1.0)-- RIGHT (3 edges, 2 tubes, R=3)
        BFS must choose the 1-tube upper path; Dijkstra must choose the 2-tube lower path.
        """
        wgraph = {
            LEFT: {0: 0.01, 1: 1.0},
            0: {LEFT: 0.01, RIGHT: 0.01},
            1: {LEFT: 1.0, 2: 1.0},
            2: {1: 1.0, RIGHT: 1.0},
            RIGHT: {0: 0.01, 2: 1.0},
        }
        unweighted = {u: list(nbrs.keys()) for u, nbrs in wgraph.items()}

        bfs_path = find_percolating_path(unweighted)
        assert bfs_path == [LEFT, 0, RIGHT]
        assert path_series_resistance(wgraph, bfs_path) == pytest.approx(200.0)

        min_r_res = find_min_resistance_path(wgraph)
        assert min_r_res is not None
        min_path, total_r = min_r_res
        assert min_path == [LEFT, 1, 2, RIGHT]
        assert total_r == pytest.approx(3.0)

    def test_bottleneck_conductance_and_conductance_percolation(self):
        """Two parallel paths:
        Path A: LEFT --(0.5)-- 0 --(0.05)-- RIGHT  (bottleneck = 0.05)
        Path B: LEFT --(0.8)-- 1 --(0.25)-- 2 --(0.9)-- RIGHT  (bottleneck = 0.25)
        """
        wgraph = {
            LEFT: {0: 0.5, 1: 0.8},
            0: {LEFT: 0.5, RIGHT: 0.05},
            1: {LEFT: 0.8, 2: 0.25},
            2: {1: 0.25, RIGHT: 0.9},
            RIGHT: {0: 0.05, 2: 0.9},
        }
        bn_res = find_bottleneck_conductance(wgraph)
        assert bn_res is not None
        g_c, bn_path = bn_res
        assert g_c == pytest.approx(0.25)
        assert bn_path == [LEFT, 1, 2, RIGHT]

        assert check_conductance_percolation(wgraph, min_conductance=0.25) is True
        assert check_conductance_percolation(wgraph, min_conductance=0.2501) is False

        filtered = filter_graph_by_conductance(wgraph, min_conductance=0.25)
        assert 0 not in filtered[RIGHT]
        assert 2 in filtered[RIGHT]

    def test_disconnected_weighted_graph_returns_none(self):
        wgraph = {LEFT: {0: 1.0}, 0: {LEFT: 1.0}, RIGHT: {}}
        assert find_min_resistance_path(wgraph) is None
        assert find_bottleneck_conductance(wgraph) is None
        assert check_conductance_percolation(wgraph, min_conductance=1e-9) is False


# ---------------------------------------------------------------------------
# 4. Simulation & Monte Carlo Integration
# ---------------------------------------------------------------------------

class TestSimulationIntegrationWithTunneling:
    def test_create_sample_and_run_sample_with_tunneling(self):
        cfg = TunnelingConfig(g0=1.0, alpha=40.0)
        rng1 = np.random.default_rng(42)
        sample = create_sample(
            concentration=0.18,
            box=TEST_BOX,
            cnt_length=0.5,
            cnt_diameter=0.05,
            contact_threshold=0.08,
            rng=rng1,
            tunneling=cfg,
        )
        rng2 = np.random.default_rng(42)
        sim_res = run_sample(
            concentration=0.18,
            box=TEST_BOX,
            cnt_length=0.5,
            cnt_diameter=0.05,
            contact_threshold=0.08,
            rng=rng2,
            tunneling=cfg,
        )

        assert sample.weighted_graph is not None
        assert sample.percolates == sim_res.percolates
        assert sample.contact_count == sim_res.contact_count
        if sample.contacts:
            assert all(c.has_conductance for c in sample.contacts)
            assert sim_res.avg_contact_conductance is not None
            assert sim_res.geom_mean_conductance is not None
        if sample.percolates:
            assert sample.min_resistance_path is not None
            assert sample.min_path_resistance == pytest.approx(sim_res.min_path_resistance)
            assert sample.bottleneck_conductance == pytest.approx(sim_res.bottleneck_conductance)

        diag = analyze_sample_tunneling(sample)
        assert diag["percolates"] == sample.percolates
        if sample.percolates and diag["resistance_improvement_ratio"] is not None:
            assert diag["resistance_improvement_ratio"] >= 1.0 - 1e-12

    def test_monte_carlo_and_yaml_config_with_tunneling(self, tmp_path: Path):
        cfg = {
            "box": {"x": 1.5, "y": 0.8, "z": 0.8},
            "cnt": {"length": 0.4, "diameter": 0.05},
            "contact": {
                "threshold": 0.08,
                "method": "grid",
                "tunneling": {
                    "enabled": True,
                    "g0": 2.0,
                    "alpha": 35.0,
                    "d_min": 0.005,
                    "min_conductance": 0.05,
                },
            },
            "simulation": {
                "concentrations": [0.02, 0.20],
                "samples_per_concentration": 5,
                "random_seed": 123,
            },
        }
        cfg_file = tmp_path / "tunn_config.yaml"
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        res1 = run_from_config(cfg_file, verbose=False)
        res2 = run_from_config(cfg_file, verbose=False)
        assert res1.tunneling is not None and res1.tunneling.enabled is True
        assert res1.probabilities == res2.probabilities
        assert res1.concentration_results[1].avg_contact_conductance == pytest.approx(
            res2.concentration_results[1].avg_contact_conductance
        )

        json_path = tmp_path / "mc_tunn.json"
        csv_path = tmp_path / "mc_tunn.csv"
        res1.save_json(json_path)
        res1.save_csv(csv_path)

        with open(json_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)
        assert jdata["parameters"]["tunneling"]["enabled"] is True
        assert jdata["parameters"]["tunneling"]["alpha"] == pytest.approx(35.0)

        with open(csv_path, "r", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert "avg_bottleneck_conductance" in header
        assert "avg_min_path_resistance" in header


# ---------------------------------------------------------------------------
# 5. Parametric Studies & Visualization
# ---------------------------------------------------------------------------

class TestTunnelingStudiesAndVisualization:
    def test_alpha_study_and_serialization_and_plot(self, tmp_path: Path):
        box = MaterialBox(size_x=1.5, size_y=0.8, size_z=0.8)
        study = run_tunneling_alpha_study(
            alpha_values=[25.0, 75.0],
            g0=1.0,
            min_conductance=0.05,
            box=box,
            cnt_length=0.45,
            cnt_diameter=0.05,
            contact_threshold=0.12,
            samples_per_concentration=5,
            n_concentrations=4,
            concentrations=[0.02, 0.08, 0.16, 0.25],
            random_seed=42,
        )
        assert study.study_mode == "alpha"
        assert len(study.points) == 2
        # Higher alpha => smaller effective cutoff
        assert study.points[1].effective_cutoff < study.points[0].effective_cutoff

        json_path = tmp_path / "study.json"
        csv_path = tmp_path / "study.csv"
        plot_path = tmp_path / "study.png"
        study.save_json(json_path)
        study.save_csv(csv_path)

        reloaded = TunnelingStudyResult.load_json(json_path)
        assert reloaded.study_mode == "alpha"
        assert len(reloaded.points) == 2
        assert reloaded.points[0].effective_cutoff == pytest.approx(study.points[0].effective_cutoff)
        assert csv_path.exists()

        fig = plot_tunneling_study(study, save_path=plot_path, dpi=100)
        assert isinstance(fig, plt.Figure)
        assert plot_path.exists() and plot_path.stat().st_size > 0
        plt.close(fig)

    def test_cutoff_study_and_plot(self, tmp_path: Path):
        box = MaterialBox(size_x=1.5, size_y=0.8, size_z=0.8)
        study = run_contact_cutoff_study(
            cutoff_values=[0.03, 0.09],
            g0=1.0,
            alpha=40.0,
            box=box,
            cnt_length=0.45,
            cnt_diameter=0.05,
            samples_per_concentration=5,
            concentrations=[0.02, 0.08, 0.16, 0.25],
            random_seed=7,
        )
        assert study.study_mode == "cutoff"
        assert len(study.points) == 2
        assert study.points[0].effective_cutoff == pytest.approx(0.03)
        assert study.points[1].effective_cutoff == pytest.approx(0.09)

        plot_path = tmp_path / "cutoff_study.png"
        fig = plot_tunneling_study(study, save_path=plot_path, dpi=100)
        assert plot_path.exists()
        plt.close(fig)

    def test_empty_sweep_raises(self):
        with pytest.raises(ValueError):
            run_tunneling_alpha_study(alpha_values=[])
        with pytest.raises(ValueError):
            run_contact_cutoff_study(cutoff_values=[])

    def test_single_sample_distribution_and_3d_conductance_render(self, tmp_path: Path):
        cfg = TunnelingConfig(g0=1.0, alpha=50.0)
        rng = np.random.default_rng(11)
        sample = create_sample(
            concentration=0.22,
            box=TEST_BOX,
            cnt_length=0.5,
            cnt_diameter=0.05,
            contact_threshold=0.08,
            rng=rng,
            tunneling=cfg,
        )
        dist_path = tmp_path / "dist.png"
        fig_dist = plot_contact_conductance_distribution(sample, save_path=dist_path, dpi=100)
        assert isinstance(fig_dist, plt.Figure)
        assert dist_path.exists()
        plt.close(fig_dist)

        render_path = tmp_path / "render_tunn.png"
        fig_3d = plot_sample_3d(
            sample,
            color_contacts_by_conductance=True,
            highlight_min_resistance_path=True,
            save_path=render_path,
            dpi=100,
        )
        assert isinstance(fig_3d, plt.Figure)
        assert render_path.exists()
        plt.close(fig_3d)


# ---------------------------------------------------------------------------
# 6. CLI Smoke Tests
# ---------------------------------------------------------------------------

class TestTunnelingCli:
    def test_analysis_cli_minimal(self, tmp_path: Path):
        cmd = [
            sys.executable,
            "-m",
            "percolation.analysis.tunneling",
            "--mode",
            "alpha",
            "--alpha-values",
            "30,70",
            "--box-size",
            "1.5,0.8,0.8",
            "--cnt-length",
            "0.45",
            "--cnt-diameter",
            "0.05",
            "--threshold",
            "0.1",
            "--samples",
            "4",
            "--n-concentrations",
            "4",
            "--output-dir",
            str(tmp_path),
            "--dpi",
            "80",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Phase 8: Tunneling Contact Conductance Study" in proc.stdout
        assert (tmp_path / "tunneling_study.json").exists()
        assert (tmp_path / "tunneling_study.csv").exists()
        assert (tmp_path / "tunneling_study.png").exists()

    def test_experiment_cli_with_tunneling(self):
        cmd = [
            sys.executable,
            "-m",
            "percolation.simulation.experiment",
            "--phi",
            "0.15",
            "--box-x",
            "1.5",
            "--box-y",
            "0.8",
            "--box-z",
            "0.8",
            "--length",
            "0.45",
            "--diameter",
            "0.05",
            "--threshold",
            "0.08",
            "--tunneling",
            "--alpha",
            "45",
            "--g0",
            "1.0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Tunneling Model" in proc.stdout

"""
Unit and integration tests for Phase 7: CNT Orientation Study.

Covers:
- OrientationConfig validation, normalization, Hermans order parameter theory, serialization
- Direction sampling: isotropic backwards-compatibility, aligned, cone (partially_aligned), Fisher
- Empirical Hermans order parameter S & acute angle calculations
- Box containment and seed reproducibility across all orientation modes
- Simulation pipeline & YAML config integration
- OrientationPoint / OrientationStudyResult JSON & CSV round-trip
- run_orientation_study (explicit & adaptive concentration grids)
- plot_orientation_study & plot_sample_3d with oriented samples
- CLI execution
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from percolation.analysis.orientation import (
    OrientationPoint,
    OrientationStudyResult,
    build_default_orientations,
    main,
    run_orientation_study,
)
from percolation.domain.cnt import MaterialBox, OrientationConfig
from percolation.generation.cnt_generator import (
    compute_order_parameter,
    compute_orientation_angles,
    generate_nanotubes,
    sample_direction,
)
from percolation.simulation.experiment import create_sample, run_sample
from percolation.simulation.monte_carlo import run_from_config, run_monte_carlo
from percolation.visualization.orientation_plot import plot_orientation_study
from percolation.visualization.renderer import plot_sample_3d

matplotlib.use("Agg")

TEST_BOX = MaterialBox(size_x=4.0, size_y=2.0, size_z=2.0)


# ---------------------------------------------------------------------------
# OrientationConfig tests
# ---------------------------------------------------------------------------

class TestOrientationConfig:
    def test_factory_methods_and_order_parameters(self):
        iso = OrientationConfig.isotropic()
        assert iso.mode == "isotropic"
        assert iso.theoretical_order_parameter("x") == pytest.approx(0.0, abs=1e-12)

        al_x = OrientationConfig.aligned("x")
        assert al_x.mode == "aligned"
        assert al_x.max_angle_deg == 0.0
        assert al_x.theoretical_order_parameter("x") == pytest.approx(1.0, abs=1e-12)
        assert al_x.theoretical_order_parameter("y") == pytest.approx(-0.5, abs=1e-12)

        al_y = OrientationConfig.aligned("y")
        assert al_y.theoretical_order_parameter("x") == pytest.approx(-0.5, abs=1e-12)
        assert al_y.theoretical_order_parameter("y") == pytest.approx(1.0, abs=1e-12)

        # Cone half-angle 60 deg: cos(60) = 0.5 -> S = 0.5 * 0.5 * 1.5 = 0.375
        cone_60 = OrientationConfig.partially_aligned(60.0, axis="x")
        assert cone_60.theoretical_order_parameter("x") == pytest.approx(0.375, rel=1e-9)
        assert cone_60.theoretical_order_parameter("y") == pytest.approx(-0.1875, rel=1e-9)

        # Cone alias
        cone_alias = OrientationConfig(mode="cone", axis="x", max_angle_deg=0.0)
        assert cone_alias.mode == "partially_aligned"
        assert cone_alias.theoretical_order_parameter("x") == pytest.approx(1.0, abs=1e-12)

        # Cone 90 deg == isotropic S = 0
        cone_90 = OrientationConfig.partially_aligned(90.0, axis="x")
        assert cone_90.theoretical_order_parameter("x") == pytest.approx(0.0, abs=1e-12)

        # Fisher limits
        f0 = OrientationConfig.fisher(0.0, axis="x")
        assert f0.theoretical_order_parameter("x") == pytest.approx(0.0, abs=1e-12)
        f_small = OrientationConfig.fisher(1e-8, axis="x")
        assert f_small.theoretical_order_parameter("x") == pytest.approx(0.0, abs=1e-12)
        f_large = OrientationConfig.fisher(1000.0, axis="x")
        assert f_large.theoretical_order_parameter("x") > 0.99

    def test_custom_axis_normalization(self):
        cfg = OrientationConfig.aligned(axis=(3.0, 4.0, 0.0))
        assert np.linalg.norm(cfg.axis_vector) == pytest.approx(1.0, rel=1e-12)
        assert cfg.axis == pytest.approx((0.6, 0.8, 0.0))

    def test_invalid_configurations_raise(self):
        with pytest.raises(ValueError, match="Invalid orientation mode"):
            OrientationConfig(mode="unknown_mode")
        with pytest.raises(ValueError, match="Unknown axis alias"):
            OrientationConfig(mode="aligned", axis="w")
        with pytest.raises(ValueError, match="non-zero"):
            OrientationConfig(mode="aligned", axis=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="max_angle_deg"):
            OrientationConfig(mode="partially_aligned", max_angle_deg=120.0)
        with pytest.raises(ValueError, match="max_angle_deg"):
            OrientationConfig(mode="partially_aligned", max_angle_deg=-5.0)
        with pytest.raises(ValueError, match="kappa"):
            OrientationConfig(mode="fisher", kappa=-1.0)

    def test_serialization_round_trip(self):
        cfg = OrientationConfig.partially_aligned(max_angle_deg=25.0, axis="z")
        d = cfg.to_dict()
        restored = OrientationConfig.from_dict(d)
        assert restored.mode == cfg.mode
        assert restored.max_angle_deg == cfg.max_angle_deg
        assert restored.axis == pytest.approx(cfg.axis)
        assert OrientationConfig.from_dict(None).mode == "isotropic"


# ---------------------------------------------------------------------------
# Direction sampling & generator tests
# ---------------------------------------------------------------------------

class TestDirectionSamplingAndGeneration:
    def test_isotropic_backwards_compatibility(self):
        """Passing orientation=None or isotropic gives bit-identical tubes."""
        tubes_none = generate_nanotubes(30, TEST_BOX, 1.0, 0.05, np.random.default_rng(77))
        tubes_iso = generate_nanotubes(
            30, TEST_BOX, 1.0, 0.05, np.random.default_rng(77),
            orientation=OrientationConfig.isotropic(),
        )
        for t1, t2 in zip(tubes_none, tubes_iso):
            np.testing.assert_array_equal(t1.center, t2.center)
            np.testing.assert_array_equal(t1.direction, t2.direction)

    def test_aligned_directions_exact(self):
        rng = np.random.default_rng(10)
        cfg = OrientationConfig.aligned("x")
        tubes = generate_nanotubes(40, TEST_BOX, 0.8, 0.05, rng, orientation=cfg)
        for t in tubes:
            assert TEST_BOX.contains_segment(t.p0, t.p1)
            np.testing.assert_allclose(t.direction, [1.0, 0.0, 0.0], atol=1e-12)
        assert compute_order_parameter(tubes, "x") == pytest.approx(1.0, abs=1e-12)
        angles = compute_orientation_angles(tubes, "x")
        np.testing.assert_allclose(angles, 0.0, atol=1e-9)

    def test_transverse_aligned_y(self):
        rng = np.random.default_rng(11)
        cfg = OrientationConfig.aligned("y")
        tubes = generate_nanotubes(40, TEST_BOX, 0.8, 0.05, rng, orientation=cfg)
        assert compute_order_parameter(tubes, "x") == pytest.approx(-0.5, abs=1e-12)
        assert compute_order_parameter(tubes, "y") == pytest.approx(1.0, abs=1e-12)
        angles_x = compute_orientation_angles(tubes, "x")
        np.testing.assert_allclose(angles_x, 90.0, atol=1e-9)

    @pytest.mark.parametrize("max_angle", [15.0, 30.0, 45.0, 60.0])
    def test_partially_aligned_cone_bounds_and_statistics(self, max_angle: float):
        rng = np.random.default_rng(2026)
        cfg = OrientationConfig.partially_aligned(max_angle_deg=max_angle, axis="x")

        # Draw 5000 raw directions directly to verify exact distribution without wall bias
        dirs = np.vstack([sample_direction(rng, cfg) for _ in range(5000)])
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

        # Hard cone boundary check
        angles_deg = np.rad2deg(np.arccos(np.clip(np.abs(dirs[:, 0]), 0.0, 1.0)))
        assert float(np.max(angles_deg)) <= max_angle + 1e-7

        # Empirical Hermans order parameter matches analytical formula
        s_emp = 0.5 * (3.0 * float(np.mean(dirs[:, 0] ** 2)) - 1.0)
        s_theo = cfg.theoretical_order_parameter("x")
        assert abs(s_emp - s_theo) < 0.02

    @pytest.mark.parametrize("kappa", [2.0, 10.0, 50.0, 400.0])
    def test_fisher_distribution_statistics(self, kappa: float):
        rng = np.random.default_rng(3030)
        cfg = OrientationConfig.fisher(kappa=kappa, axis="z")
        dirs = np.vstack([sample_direction(rng, cfg) for _ in range(5000)])
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

        s_emp = 0.5 * (3.0 * float(np.mean(dirs[:, 2] ** 2)) - 1.0)
        s_theo = cfg.theoretical_order_parameter("z")
        assert abs(s_emp - s_theo) < 0.03

    def test_all_modes_box_containment_and_reproducibility(self):
        configs = [
            OrientationConfig.isotropic(),
            OrientationConfig.aligned("x"),
            OrientationConfig.partially_aligned(30.0, "x"),
            OrientationConfig.fisher(8.0, "y"),
        ]
        for cfg in configs:
            t1 = generate_nanotubes(25, TEST_BOX, 0.8, 0.05, np.random.default_rng(55), orientation=cfg)
            t2 = generate_nanotubes(25, TEST_BOX, 0.8, 0.05, np.random.default_rng(55), orientation=cfg)
            for a, b in zip(t1, t2):
                assert TEST_BOX.contains_segment(a.p0, a.p1)
                np.testing.assert_array_equal(a.center, b.center)
                np.testing.assert_array_equal(a.direction, b.direction)

    def test_empty_tubes_helpers(self):
        assert compute_order_parameter([]) == 0.0
        assert compute_orientation_angles([]).size == 0


# ---------------------------------------------------------------------------
# Pipeline & config integration tests
# ---------------------------------------------------------------------------

class TestSimulationPipelineIntegration:
    def test_create_sample_and_run_sample_with_orientation(self):
        orient = OrientationConfig.partially_aligned(20.0, axis="x")
        sample = create_sample(
            concentration=0.05,
            box=TEST_BOX,
            cnt_length=0.8,
            cnt_diameter=0.1,
            contact_threshold=0.05,
            rng=np.random.default_rng(12),
            orientation=orient,
        )
        assert sample.orientation == orient
        assert compute_order_parameter(sample.tubes, "x") > 0.7

        res = run_sample(
            concentration=0.05,
            box=TEST_BOX,
            cnt_length=0.8,
            cnt_diameter=0.1,
            contact_threshold=0.05,
            rng=np.random.default_rng(12),
            orientation=orient,
        )
        assert res.percolates == sample.percolates
        assert res.contact_count == sample.contact_count

    def test_run_monte_carlo_and_config_with_orientation(self, tmp_path: Path):
        orient = OrientationConfig.partially_aligned(30.0, axis="x")
        mc = run_monte_carlo(
            concentrations=[0.02, 0.06],
            samples_per_concentration=3,
            box=MaterialBox(1.5, 1.0, 1.0),
            cnt_length=0.5,
            cnt_diameter=0.08,
            contact_threshold=0.06,
            random_seed=7,
            orientation=orient,
        )
        assert mc.orientation == orient
        d = mc.to_dict()
        assert d["parameters"]["cnt"]["orientation"]["mode"] == "partially_aligned"
        assert d["parameters"]["cnt"]["orientation"]["max_angle_deg"] == 30.0

        cfg = {
            "box": {"x": 1.5, "y": 1.0, "z": 1.0},
            "cnt": {
                "length": 0.5,
                "diameter": 0.08,
                "orientation": {"mode": "partially_aligned", "axis": "x", "max_angle_deg": 25.0},
            },
            "contact": {"threshold": 0.06},
            "simulation": {
                "concentrations": [0.02, 0.06],
                "samples_per_concentration": 3,
                "random_seed": 7,
            },
        }
        cfg_file = tmp_path / "orient_config.yaml"
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        res = run_from_config(cfg_file, verbose=False)
        assert res.orientation.mode == "partially_aligned"
        assert res.orientation.max_angle_deg == 25.0


# ---------------------------------------------------------------------------
# Orientation study & serialization tests
# ---------------------------------------------------------------------------

class TestOrientationStudy:
    def test_build_default_orientations(self):
        orients = build_default_orientations((0.0, 30.0, 90.0), axis="x", include_transverse=True)
        assert [o.mode for o in orients] == ["aligned", "partially_aligned", "isotropic", "aligned"]
        assert orients[-1].axis == pytest.approx((0.0, 1.0, 0.0))

    def test_empty_orientations_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            run_orientation_study(orientations=[])

    def test_run_orientation_study_fast_and_reproducible(self):
        box = MaterialBox(size_x=1.5, size_y=1.0, size_z=1.0)
        orients = [
            OrientationConfig.aligned("x"),
            OrientationConfig.partially_aligned(45.0, "x"),
            OrientationConfig.isotropic(),
        ]
        progress = []

        def cb(done, total, pt):
            progress.append((done, total, pt.label))

        study1 = run_orientation_study(
            orientations=orients,
            box=box,
            cnt_length=0.5,
            cnt_diameter=0.06,
            contact_threshold=0.05,
            samples_per_concentration=4,
            n_concentrations=4,
            random_seed=42,
            progress_callback=cb,
        )
        study2 = run_orientation_study(
            orientations=orients,
            box=box,
            cnt_length=0.5,
            cnt_diameter=0.06,
            contact_threshold=0.05,
            samples_per_concentration=4,
            n_concentrations=4,
            random_seed=42,
        )

        assert len(study1.points) == 3
        assert len(progress) == 3
        assert study1.isotropic_phi_c is not None
        assert study1.optimal_point is not None

        for p1, p2 in zip(study1.points, study2.points):
            assert p1.phi_c == pytest.approx(p2.phi_c)
            assert p1.measured_order_parameter == pytest.approx(p2.measured_order_parameter)
            assert p1.phi_c > 0.0

    def test_serialization_json_and_csv(self, tmp_path: Path):
        p_iso = OrientationPoint(
            orientation=OrientationConfig.isotropic(),
            phi_c=0.045,
            phi_c_stderr=0.002,
            r_squared=0.98,
            theoretical_order_parameter=0.0,
            measured_order_parameter=0.01,
            concentrations=[0.02, 0.045, 0.07],
            probabilities=[0.0, 0.5, 1.0],
        )
        p_cone = OrientationPoint(
            orientation=OrientationConfig.partially_aligned(30.0, "x"),
            phi_c=0.039,
            phi_c_stderr=0.0015,
            r_squared=0.99,
            theoretical_order_parameter=0.808,
            measured_order_parameter=0.80,
            concentrations=[0.02, 0.04, 0.06],
            probabilities=[0.0, 0.55, 1.0],
        )
        study = OrientationStudyResult(
            points=[p_iso, p_cone],
            box_size=(4.0, 2.0, 2.0),
            cnt_length=1.0,
            cnt_diameter=0.1,
            contact_threshold=0.05,
        )

        assert study.isotropic_phi_c == pytest.approx(0.045)
        assert study.optimal_point is p_cone
        assert study.relative_threshold(p_cone) == pytest.approx(0.039 / 0.045)

        json_file = tmp_path / "orientation.json"
        csv_file = tmp_path / "orientation.csv"
        study.save_json(json_file)
        study.save_csv(csv_file)

        loaded = OrientationStudyResult.load_json(json_file)
        assert len(loaded.points) == 2
        assert loaded.points[1].orientation.mode == "partially_aligned"
        assert loaded.points[1].phi_c == pytest.approx(0.039)
        assert loaded.box_size == (4.0, 2.0, 2.0)

        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3
        assert rows[0][0] == "label"
        assert "phi_c" in rows[0]


# ---------------------------------------------------------------------------
# Visualization & CLI tests
# ---------------------------------------------------------------------------

class TestVisualizationAndCli:
    def test_plot_orientation_study_and_3d_renderer(self, tmp_path: Path):
        points = [
            OrientationPoint(
                orientation=OrientationConfig.aligned("y"),
                phi_c=0.095,
                phi_c_stderr=0.004,
                theoretical_order_parameter=-0.5,
                measured_order_parameter=-0.5,
                concentrations=[0.05, 0.095, 0.14],
                probabilities=[0.0, 0.5, 1.0],
            ),
            OrientationPoint(
                orientation=OrientationConfig.isotropic(),
                phi_c=0.050,
                phi_c_stderr=0.002,
                theoretical_order_parameter=0.0,
                measured_order_parameter=0.01,
                concentrations=[0.02, 0.05, 0.08],
                probabilities=[0.0, 0.5, 1.0],
            ),
            OrientationPoint(
                orientation=OrientationConfig.partially_aligned(45.0, "x"),
                phi_c=0.044,
                phi_c_stderr=0.002,
                theoretical_order_parameter=0.6036,
                measured_order_parameter=0.60,
                concentrations=[0.02, 0.045, 0.07],
                probabilities=[0.0, 0.52, 1.0],
            ),
            OrientationPoint(
                orientation=OrientationConfig.aligned("x"),
                phi_c=0.110,
                phi_c_stderr=0.005,
                theoretical_order_parameter=1.0,
                measured_order_parameter=1.0,
                concentrations=[0.06, 0.11, 0.16],
                probabilities=[0.0, 0.5, 1.0],
            ),
        ]
        study = OrientationStudyResult(points=points)
        out_png = tmp_path / "orientation_plot.png"
        fig = plot_orientation_study(study, save_path=out_png, dpi=100)
        assert len(fig.axes) == 2
        assert out_png.exists()
        assert out_png.stat().st_size > 1000
        plt.close(fig)

        # 3D render of an oriented sample
        sample = create_sample(
            concentration=0.04,
            box=MaterialBox(2.0, 1.0, 1.0),
            cnt_length=0.5,
            cnt_diameter=0.08,
            contact_threshold=0.05,
            rng=np.random.default_rng(4),
            orientation=OrientationConfig.partially_aligned(20.0, "x"),
        )
        fig3d = plot_sample_3d(sample, save_path=tmp_path / "oriented_3d.png", dpi=100)
        assert (tmp_path / "oriented_3d.png").exists()
        plt.close(fig3d)

    def test_cli_execution_minimal(self, tmp_path: Path):
        ret = main([
            "--angles", "30,90",
            "--samples", "2",
            "--n-concentrations", "3",
            "--box-size", "1.5", "1.0", "1.0",
            "--cnt-length", "0.5",
            "--cnt-diameter", "0.06",
            "--threshold", "0.05",
            "--output-dir", str(tmp_path),
            "--dpi", "100",
            "--seed", "99",
        ])
        assert ret == 0
        assert (tmp_path / "orientation_study.json").exists()
        assert (tmp_path / "orientation_study.csv").exists()
        assert (tmp_path / "orientation_study.png").exists()

        with open(tmp_path / "orientation_study.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["points"]) == 2
        assert data["isotropic_phi_c"] is not None

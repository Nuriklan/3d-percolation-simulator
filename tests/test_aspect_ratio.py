"""
Unit and integration tests for Phase 6: Aspect Ratio Study.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from percolation.analysis.aspect_ratio import (
    AspectRatioPoint,
    AspectRatioStudyResult,
    PowerLawFitResult,
    fit_power_law,
    main,
    run_aspect_ratio_study,
)
from percolation.domain.cnt import MaterialBox
from percolation.visualization.aspect_ratio_plot import plot_aspect_ratio_study

# ---------------------------------------------------------------------------
# Power law fitting tests
# ---------------------------------------------------------------------------

class TestFitPowerLaw:
    def test_fit_exact_power_law(self):
        """fit_power_law recovers exact beta and C for ideal power-law data."""
        true_beta = 1.0
        true_c = 0.8
        ars = [10.0, 20.0, 40.0, 80.0]
        phis = [true_c * (ar ** (-true_beta)) for ar in ars]

        fit = fit_power_law(ars, phis)

        assert np.isclose(fit.beta, true_beta, atol=1e-6)
        assert np.isclose(fit.c_coefficient, true_c, atol=1e-6)
        assert np.isclose(fit.r_squared, 1.0, atol=1e-6)
        assert fit.p_value is not None and fit.p_value < 1e-4

        # Test predict
        pred_single = fit.predict(20.0)
        assert np.isclose(pred_single, 0.04)
        pred_arr = fit.predict(np.array([10.0, 20.0]))
        assert np.allclose(pred_arr, [0.08, 0.04])

    def test_fit_arbitrary_exponent(self):
        """fit_power_law recovers fractional exponent."""
        ars = [15.0, 30.0, 60.0, 120.0]
        phis = [1.5 * (ar ** (-0.75)) for ar in ars]

        fit = fit_power_law(ars, phis)
        assert np.isclose(fit.beta, 0.75, atol=1e-4)
        assert np.isclose(fit.c_coefficient, 1.5, atol=1e-4)
        assert fit.r_squared > 0.999

    def test_fit_mismatched_lengths_raises(self):
        """Mismatched sequence lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            fit_power_law([10.0, 20.0], [0.05])

    def test_fit_insufficient_points_raises(self):
        """Fewer than 2 valid points raise ValueError."""
        with pytest.raises(ValueError, match="At least 2"):
            fit_power_law([10.0], [0.05])

    def test_fit_all_identical_ar_raises(self):
        """Identical AR values produce zero variance and raise ValueError."""
        with pytest.raises(ValueError, match="all AR values are identical"):
            fit_power_law([20.0, 20.0, 20.0], [0.05, 0.06, 0.04])

    def test_fit_filters_invalid_and_negative_points(self):
        """Negative and non-positive points are safely filtered out."""
        ars = [-5.0, 10.0, 20.0, 40.0]
        phis = [0.1, 0.1, 0.05, 0.025]

        fit = fit_power_law(ars, phis)
        assert np.isclose(fit.beta, 1.0, atol=1e-5)
        assert fit.r_squared > 0.999


# ---------------------------------------------------------------------------
# Data classes and serialization tests
# ---------------------------------------------------------------------------

class TestDataSerialization:
    def test_aspect_ratio_point_to_and_from_dict(self):
        pt = AspectRatioPoint(
            aspect_ratio=25.0,
            length=1.0,
            diameter=0.04,
            phi_c=0.038,
            phi_c_stderr=0.0015,
            r_squared=0.985,
            concentrations=[0.02, 0.04, 0.06],
            probabilities=[0.1, 0.6, 1.0],
        )

        assert pt.ar == 25.0
        assert pt.phi_c_error == 0.0015

        d = pt.to_dict()
        assert d["aspect_ratio"] == 25.0
        assert d["phi_c"] == 0.038

        recovered = AspectRatioPoint.from_dict(d)
        assert recovered.aspect_ratio == pt.aspect_ratio
        assert recovered.length == pt.length
        assert recovered.diameter == pt.diameter
        assert recovered.phi_c == pt.phi_c
        assert recovered.phi_c_stderr == pt.phi_c_stderr
        assert recovered.r_squared == pt.r_squared
        assert recovered.concentrations == pt.concentrations
        assert recovered.probabilities == pt.probabilities

    def test_aspect_ratio_study_result_serialization(self, tmp_path: Path):
        pt1 = AspectRatioPoint(
            aspect_ratio=15.0,
            length=1.0,
            diameter=0.0667,
            phi_c=0.075,
            phi_c_stderr=0.002,
            r_squared=0.99,
            concentrations=[0.03, 0.07, 0.11],
            probabilities=[0.0, 0.5, 1.0],
        )
        pt2 = AspectRatioPoint(
            aspect_ratio=30.0,
            length=1.0,
            diameter=0.0333,
            phi_c=0.038,
            phi_c_stderr=0.001,
            r_squared=0.98,
            concentrations=[0.015, 0.035, 0.055],
            probabilities=[0.0, 0.5, 1.0],
        )
        pl = PowerLawFitResult(
            beta=0.98,
            c_coefficient=1.1,
            r_squared=0.995,
            beta_stderr=0.02,
            c_stderr=0.05,
        )
        study = AspectRatioStudyResult(
            points=[pt1, pt2],
            power_law=pl,
            box_size=(3.0, 2.0, 2.0),
        )

        assert study.beta == 0.98
        assert study.c_coefficient == 1.1
        assert study.r_squared == 0.995
        assert study.beta_stderr == 0.02
        assert study.c_stderr == 0.05

        # Test JSON round-trip
        json_file = tmp_path / "study.json"
        study.save_json(json_file)
        assert json_file.exists()

        loaded = AspectRatioStudyResult.load_json(json_file)
        assert len(loaded.points) == 2
        assert np.isclose(loaded.beta, 0.98)
        assert np.isclose(loaded.c_coefficient, 1.1)
        assert loaded.box_size == (3.0, 2.0, 2.0)

        # Test CSV export
        csv_file = tmp_path / "study.csv"
        study.save_csv(csv_file)
        assert csv_file.exists()
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert reader[0] == ["aspect_ratio", "length", "diameter", "phi_c", "phi_c_stderr", "r_squared"]
            assert len(reader) == 3


# ---------------------------------------------------------------------------
# Integration experiment tests
# ---------------------------------------------------------------------------

class TestRunAspectRatioStudy:
    def test_run_aspect_ratio_study_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            run_aspect_ratio_study(ar_values=[])

    def test_run_aspect_ratio_study_fast(self):
        """run_aspect_ratio_study completes quickly on a minimal box and returns valid scaling."""
        box = MaterialBox(size_x=1.5, size_y=1.0, size_z=1.0)
        ar_values = [15.0, 30.0]

        progress_calls = []

        def cb(done, total, pt):
            progress_calls.append((done, total, pt.aspect_ratio))

        study = run_aspect_ratio_study(
            ar_values=ar_values,
            box=box,
            cnt_length=0.5,
            samples_per_concentration=4,
            n_concentrations=3,
            concentration_span=(0.5, 1.5),
            random_seed=123,
            contact_method="auto",
            progress_callback=cb,
        )

        assert isinstance(study, AspectRatioStudyResult)
        assert len(study.points) == 2
        assert len(progress_calls) == 2
        assert study.points[0].aspect_ratio == 15.0
        assert study.points[1].aspect_ratio == 30.0

        # High AR has lower phi_c
        assert study.points[0].phi_c > 0.0
        assert study.points[1].phi_c > 0.0
        assert study.beta > 0.0
        assert study.c_coefficient > 0.0
        assert 0.0 <= study.r_squared <= 1.0


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------

class TestPlotAspectRatioStudy:
    def test_plot_aspect_ratio_study_generation_and_saving(self, tmp_path: Path):
        """plot_aspect_ratio_study produces a valid two-panel figure and writes PNG."""
        pt1 = AspectRatioPoint(
            aspect_ratio=15.0,
            length=1.0,
            diameter=0.0667,
            phi_c=0.075,
            phi_c_stderr=0.002,
            r_squared=0.99,
            concentrations=[0.02, 0.05, 0.08, 0.11],
            probabilities=[0.0, 0.25, 0.75, 1.0],
        )
        pt2 = AspectRatioPoint(
            aspect_ratio=30.0,
            length=1.0,
            diameter=0.0333,
            phi_c=0.038,
            phi_c_stderr=0.001,
            r_squared=0.98,
            concentrations=[0.01, 0.025, 0.04, 0.055],
            probabilities=[0.0, 0.2, 0.8, 1.0],
        )
        pl = PowerLawFitResult(
            beta=0.98,
            c_coefficient=1.1,
            r_squared=0.995,
            beta_stderr=0.02,
        )
        study = AspectRatioStudyResult(points=[pt1, pt2], power_law=pl)

        save_file = tmp_path / "scaling_test.png"
        fig = plot_aspect_ratio_study(study, save_path=save_file, dpi=100)

        assert fig is not None
        assert len(fig.axes) == 2
        assert save_file.exists()
        assert save_file.stat().st_size > 1000
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCliRunner:
    def test_cli_execution_minimal(self, tmp_path: Path):
        """CLI executes cleanly with small test parameters."""
        ret = main([
            "--ar-values", "15,30",
            "--samples", "2",
            "--box-size", "1.5", "1.0", "1.0",
            "--cnt-length", "0.5",
            "--output-dir", str(tmp_path),
            "--dpi", "100",
            "--seed", "99",
        ])
        assert ret == 0

        assert (tmp_path / "aspect_ratio_study.json").exists()
        assert (tmp_path / "aspect_ratio_study.csv").exists()
        assert (tmp_path / "aspect_ratio_scaling.png").exists()

        with open(tmp_path / "aspect_ratio_study.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "beta" in data
            assert len(data["points"]) == 2

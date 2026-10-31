"""
Unit tests for Phase 3: percolation threshold estimation, confidence intervals,
logistic curve fitting, and visualization.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless tests
import matplotlib.pyplot as plt
import numpy as np
import pytest

from percolation.analysis.threshold import (
    ThresholdResult,
    estimate_threshold,
    estimate_threshold_linear,
    fit_logistic,
    logistic_function,
    wilson_score_interval,
)
from percolation.domain.cnt import (
    ConcentrationResult,
    MaterialBox,
    MonteCarloResult,
)
from percolation.visualization.threshold_plot import plot_percolation_probability

# ---------------------------------------------------------------------------
# Wilson score interval tests
# ---------------------------------------------------------------------------

class TestWilsonScoreInterval:
    def test_zero_successes(self):
        low, high = wilson_score_interval(successes=0, trials=100, confidence=0.95)
        assert low == 0.0
        assert 0.0 < high < 0.05

    def test_all_successes(self):
        low, high = wilson_score_interval(successes=100, trials=100, confidence=0.95)
        assert high == 1.0
        assert 0.95 < low < 1.0

    def test_symmetry(self):
        # Interval for 30/100 should mirror interval for 70/100 around 0.5
        low_30, high_30 = wilson_score_interval(30, 100, 0.95)
        low_70, high_70 = wilson_score_interval(70, 100, 0.95)
        assert np.isclose(low_30, 1.0 - high_70, atol=1e-7)
        assert np.isclose(high_30, 1.0 - low_70, atol=1e-7)

    def test_half_successes(self):
        low, high = wilson_score_interval(50, 100, 0.95)
        assert low < 0.5 < high
        assert np.isclose(0.5 - low, high - 0.5, atol=1e-7)

    def test_higher_confidence_widens_interval(self):
        low_90, high_90 = wilson_score_interval(50, 100, 0.90)
        low_99, high_99 = wilson_score_interval(50, 100, 0.99)
        assert (high_99 - low_99) > (high_90 - low_90)

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError, match="trials must be positive"):
            wilson_score_interval(0, 0)

        with pytest.raises(ValueError, match="Successes must be between"):
            wilson_score_interval(15, 10)

        with pytest.raises(ValueError, match="Successes must be between"):
            wilson_score_interval(-1, 10)

        with pytest.raises(ValueError, match="Confidence must be strictly between"):
            wilson_score_interval(5, 10, confidence=1.0)

        with pytest.raises(ValueError, match="Confidence must be strictly between"):
            wilson_score_interval(5, 10, confidence=0.0)


# ---------------------------------------------------------------------------
# Logistic function tests
# ---------------------------------------------------------------------------

class TestLogisticFunction:
    def test_center_value_is_half(self):
        phi_c = 0.005
        assert np.isclose(logistic_function(phi_c, phi_c=phi_c, k=100.0), 0.5)

    def test_symmetry_around_center(self):
        phi_c = 0.004
        k = 500.0
        delta = 0.001
        p_plus = logistic_function(phi_c + delta, phi_c=phi_c, k=k)
        p_minus = logistic_function(phi_c - delta, phi_c=phi_c, k=k)
        assert np.isclose(p_plus + p_minus, 1.0)

    def test_asymptotes(self):
        phi_c = 0.005
        k = 1000.0
        assert np.isclose(logistic_function(1.0, phi_c=phi_c, k=k), 1.0)
        assert np.isclose(logistic_function(-1.0, phi_c=phi_c, k=k), 0.0)
        assert logistic_function(0.0, phi_c=phi_c, k=k) < 0.01

    def test_array_evaluation(self):
        phis = np.array([0.001, 0.005, 0.010])
        res = logistic_function(phis, phi_c=0.005, k=500.0)
        assert isinstance(res, np.ndarray)
        assert len(res) == 3
        assert res[0] < res[1] < res[2]


# ---------------------------------------------------------------------------
# Linear interpolation tests
# ---------------------------------------------------------------------------

class TestLinearInterpolation:
    def test_exact_midpoint(self):
        phis = [0.002, 0.004]
        probs = [0.25, 0.75]
        # At P=0.5, midway between 0.002 and 0.004 -> 0.003
        phi_c = estimate_threshold_linear(phis, probs, target_probability=0.5)
        assert phi_c is not None
        assert np.isclose(phi_c, 0.003)

    def test_exact_data_point(self):
        phis = [0.001, 0.003, 0.005]
        probs = [0.0, 0.5, 1.0]
        phi_c = estimate_threshold_linear(phis, probs, target_probability=0.5)
        assert phi_c is not None
        assert np.isclose(phi_c, 0.003)

    def test_out_of_bounds_returns_none(self):
        # All probabilities below 0.5
        assert estimate_threshold_linear([0.001, 0.002], [0.0, 0.2]) is None
        # All probabilities above 0.5
        assert estimate_threshold_linear([0.001, 0.002], [0.8, 1.0]) is None

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            estimate_threshold_linear([0.001], [0.1, 0.2])

    def test_insufficient_points_returns_none(self):
        assert estimate_threshold_linear([0.001], [0.5]) is None


# ---------------------------------------------------------------------------
# Logistic fitting tests
# ---------------------------------------------------------------------------

class TestFitLogistic:
    def test_fit_synthetic_logistic(self):
        true_phi_c = 0.0045
        true_k = 1500.0
        phis = np.linspace(0.001, 0.008, 15)
        probs = logistic_function(phis, phi_c=true_phi_c, k=true_k)

        fit = fit_logistic(phis, probs)
        assert fit is not None
        assert fit.converged
        assert np.isclose(fit.phi_c, true_phi_c, atol=1e-5)
        assert np.isclose(fit.k, true_k, rtol=0.05)
        assert fit.r_squared > 0.999
        assert fit.phi_c_stderr is not None
        assert fit.phi_c_stderr >= 0.0

    def test_fit_flat_data_returns_none(self):
        phis = [0.001, 0.002, 0.003, 0.004]
        probs = [0.0, 0.0, 0.0, 0.0]
        assert fit_logistic(phis, probs) is None

    def test_fit_insufficient_points_returns_none(self):
        assert fit_logistic([0.001, 0.002], [0.1, 0.9]) is None

    def test_fit_noisy_transition(self):
        # Realistic noisy Monte Carlo transition
        phis = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
        probs = [0.00, 0.02, 0.15, 0.52, 0.88, 0.98, 1.00]

        fit = fit_logistic(phis, probs)
        assert fit is not None
        assert 0.0035 < fit.phi_c < 0.0045
        assert fit.k > 0
        assert fit.r_squared > 0.95


# ---------------------------------------------------------------------------
# High-level pipeline & ThresholdResult tests
# ---------------------------------------------------------------------------

class TestEstimateThreshold:
    @pytest.fixture
    def mock_mc_result(self) -> MonteCarloResult:
        box = MaterialBox(size_x=10.0, size_y=5.0, size_z=5.0)
        phis = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
        percolating = [0, 5, 25, 65, 92, 100]
        samples = 100
        conc_results = [
            ConcentrationResult(
                concentration=phi,
                samples=samples,
                percolating_samples=p,
                probability=p / samples,
            )
            for phi, p in zip(phis, percolating)
        ]
        return MonteCarloResult(
            box=box,
            cnt_length=1.0,
            cnt_diameter=0.1,
            contact_threshold=0.05,
            random_seed=42,
            concentration_results=conc_results,
        )

    def test_estimate_from_mc_result(self, mock_mc_result):
        res = estimate_threshold(mock_mc_result)
        assert isinstance(res, ThresholdResult)
        assert res.phi_c is not None
        assert 0.003 < res.phi_c < 0.005
        assert res.method in ("logistic", "linear_interpolation")
        assert res.phi_c_linear is not None
        assert len(res.confidence_intervals) == 6

    def test_monte_carlo_result_helper_method(self, mock_mc_result):
        res = mock_mc_result.estimate_threshold()
        assert isinstance(res, ThresholdResult)
        assert res.phi_c is not None

    def test_serialization_to_dict_and_json(self, mock_mc_result, tmp_path):
        res = estimate_threshold(mock_mc_result)
        d = res.to_dict()
        assert "phi_c" in d
        assert "confidence_intervals" in d
        assert len(d["confidence_intervals"]) == 6

        json_file = tmp_path / "threshold.json"
        res.save_json(json_file)
        assert json_file.exists()

        with open(json_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["phi_c"] == res.phi_c


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------

class TestPlotPercolation:
    def test_plot_generation_and_saving(self, tmp_path):
        box = MaterialBox(size_x=10.0, size_y=5.0, size_z=5.0)
        phis = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
        conc_results = [
            ConcentrationResult(
                concentration=phi,
                samples=50,
                percolating_samples=int(p * 50),
                probability=p,
            )
            for phi, p in zip(phis, [0.0, 0.04, 0.20, 0.60, 0.90, 1.0])
        ]
        mc_result = MonteCarloResult(
            box=box,
            cnt_length=1.0,
            cnt_diameter=0.1,
            contact_threshold=0.05,
            random_seed=42,
            concentration_results=conc_results,
        )

        plot_file = tmp_path / "plot.png"
        fig = plot_percolation_probability(mc_result, save_path=plot_file)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert plot_file.exists()
        assert plot_file.stat().st_size > 1000
        plt.close(fig)

    def test_plot_on_custom_axes(self):
        fig, ax = plt.subplots()
        box = MaterialBox(size_x=10.0, size_y=5.0, size_z=5.0)
        conc_results = [
            ConcentrationResult(
                concentration=0.001,
                samples=10,
                percolating_samples=0,
                probability=0.0,
            ),
            ConcentrationResult(
                concentration=0.005,
                samples=10,
                percolating_samples=10,
                probability=1.0,
            ),
        ]
        mc_result = MonteCarloResult(
            box=box,
            cnt_length=1.0,
            cnt_diameter=0.1,
            contact_threshold=0.05,
            random_seed=42,
            concentration_results=conc_results,
        )

        returned_fig = plot_percolation_probability(mc_result, ax=ax)
        assert returned_fig is fig
        assert ax.get_ylabel() == "Percolation Probability P(φ)"
        plt.close(fig)

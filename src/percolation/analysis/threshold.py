"""
Percolation threshold estimation — Phase 3 implementation.

Provides methods for analyzing Monte Carlo percolation simulation results:
1. Wilson score confidence intervals for percolation probability P(φ).
2. Approach A: Linear interpolation to estimate φ_c where P(φ_c) ≈ 0.5.
3. Approach B: Non-linear logistic sigmoid fitting:
      P(φ) = 1 / (1 + exp(-k * (φ - φ_c)))
   providing transition threshold φ_c, steepness k, parameter standard errors,
   and coefficient of determination R².
4. High-level analysis pipeline returning a structured ThresholdResult.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize, stats

if TYPE_CHECKING:
    from percolation.domain.cnt import ConcentrationResult, MonteCarloResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInterval:
    """Binomial confidence interval for percolation probability at a concentration."""

    concentration: float
    probability: float
    ci_lower: float
    ci_upper: float
    confidence: float = 0.95

    @property
    def error_minus(self) -> float:
        """Lower error bar magnitude: P - ci_lower."""
        return max(0.0, self.probability - self.ci_lower)

    @property
    def error_plus(self) -> float:
        """Upper error bar magnitude: ci_upper - P."""
        return max(0.0, self.ci_upper - self.probability)

    def to_dict(self) -> dict:
        return {
            "concentration": self.concentration,
            "probability": self.probability,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence": self.confidence,
        }


@dataclass
class LogisticFitResult:
    """Outcome of logistic sigmoid curve fitting."""

    phi_c: float
    k: float
    phi_c_stderr: float | None = None
    k_stderr: float | None = None
    r_squared: float = 0.0
    converged: bool = True

    def predict(self, phi: float | np.ndarray) -> float | np.ndarray:
        """Evaluate fitted logistic curve at given concentration(s)."""
        return logistic_function(phi, phi_c=self.phi_c, k=self.k)

    def to_dict(self) -> dict:
        return {
            "phi_c": self.phi_c,
            "k": self.k,
            "phi_c_stderr": self.phi_c_stderr,
            "k_stderr": self.k_stderr,
            "r_squared": self.r_squared,
            "converged": self.converged,
        }


@dataclass
class ThresholdResult:
    """Comprehensive percolation threshold estimation result.

    Attributes
    ----------
    phi_c                  : primary estimated percolation threshold φ_c
    method                 : estimation method used for phi_c ('logistic', 'linear_interpolation', etc.)
    phi_c_linear           : threshold from linear interpolation (Approach A)
    logistic_fit           : detailed result of logistic fitting (Approach B), if converged
    confidence_intervals   : Wilson confidence intervals for each sampled concentration
    confidence_level       : confidence level used (e.g. 0.95 for 95%)
    target_probability     : target threshold probability (default 0.5)
    """

    phi_c: float | None
    method: str
    phi_c_linear: float | None = None
    logistic_fit: LogisticFitResult | None = None
    confidence_intervals: list[ConfidenceInterval] = field(default_factory=list)
    confidence_level: float = 0.95
    target_probability: float = 0.5

    @property
    def phi_c_logistic(self) -> float | None:
        """Threshold from logistic fit, if available."""
        return self.logistic_fit.phi_c if self.logistic_fit else None

    @property
    def logistic_k(self) -> float | None:
        """Transition steepness k from logistic fit, if available."""
        return self.logistic_fit.k if self.logistic_fit else None

    @property
    def logistic_r2(self) -> float | None:
        """R² goodness of fit from logistic fit, if available."""
        return self.logistic_fit.r_squared if self.logistic_fit else None

    @property
    def logistic_phi_c_stderr(self) -> float | None:
        """Standard error of φ_c from logistic fit, if available."""
        return self.logistic_fit.phi_c_stderr if self.logistic_fit else None

    def to_dict(self) -> dict:
        return {
            "phi_c": self.phi_c,
            "method": self.method,
            "target_probability": self.target_probability,
            "confidence_level": self.confidence_level,
            "phi_c_linear": self.phi_c_linear,
            "logistic_fit": self.logistic_fit.to_dict() if self.logistic_fit else None,
            "confidence_intervals": [ci.to_dict() for ci in self.confidence_intervals],
        }

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save threshold analysis result to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)


# ---------------------------------------------------------------------------
# Core mathematical functions
# ---------------------------------------------------------------------------

def logistic_function(phi: float | np.ndarray, phi_c: float, k: float) -> float | np.ndarray:
    """Logistic sigmoid function for percolation probability:

        P(φ) = 1 / (1 + exp(-k * (φ - φ_c)))

    Parameters
    ----------
    phi   : concentration or array of concentrations
    phi_c : threshold concentration (where P = 0.5)
    k     : transition steepness (k > 0)
    """
    x = -k * (np.asarray(phi) - phi_c)
    # Clip to prevent overflow in exp
    x_clipped = np.clip(x, -500.0, 500.0)
    result = 1.0 / (1.0 + np.exp(x_clipped))
    if np.isscalar(phi):
        return float(result)
    return result


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes  : number of successes (percolating samples), 0 <= successes <= trials
    trials     : number of Bernoulli trials (total samples), > 0
    confidence : confidence level, 0 < confidence < 1 (default 0.95)

    Returns
    -------
    (ci_lower, ci_upper) bounded in [0.0, 1.0]
    """
    if trials <= 0:
        raise ValueError(f"Number of trials must be positive, got {trials}")
    if successes < 0 or successes > trials:
        raise ValueError(f"Successes must be between 0 and {trials}, got {successes}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"Confidence must be strictly between 0 and 1, got {confidence}")

    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))

    p_hat = successes / trials
    n = float(trials)
    z2 = z * z

    denominator = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = (z / denominator) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))

    if successes == 0:
        lower = 0.0
    else:
        lower = max(0.0, float(centre - half_width))

    if successes == trials:
        upper = 1.0
    else:
        upper = min(1.0, float(centre + half_width))

    return lower, upper


def compute_confidence_intervals(
    results: Sequence[ConcentrationResult],
    confidence: float = 0.95,
) -> list[ConfidenceInterval]:
    """Compute Wilson score confidence intervals for a sequence of ConcentrationResult."""
    intervals: list[ConfidenceInterval] = []
    for r in results:
        lower, upper = wilson_score_interval(
            successes=r.percolating_samples,
            trials=r.samples,
            confidence=confidence,
        )
        intervals.append(
            ConfidenceInterval(
                concentration=r.concentration,
                probability=r.probability,
                ci_lower=lower,
                ci_upper=upper,
                confidence=confidence,
            )
        )
    return intervals


# ---------------------------------------------------------------------------
# Approach A: Linear interpolation
# ---------------------------------------------------------------------------

def estimate_threshold_linear(
    concentrations: Sequence[float],
    probabilities: Sequence[float],
    target_probability: float = 0.5,
) -> float | None:
    """Estimate percolation threshold φ_c via linear interpolation (Approach A).

    Finds the interval [φ_i, φ_{i+1}] spanning target_probability and performs
    linear interpolation:
        φ_c = φ_i + (target - P_i) / (P_{i+1} - P_i) * (φ_{i+1} - φ_i)

    Parameters
    ----------
    concentrations     : sequence of tested concentrations
    probabilities      : corresponding empirical percolation probabilities P(φ)
    target_probability : target probability (default 0.5)

    Returns
    -------
    φ_c or None if the target probability is not bracketed by the data.
    """
    if len(concentrations) != len(probabilities):
        raise ValueError("concentrations and probabilities must have the same length.")
    if len(concentrations) < 2:
        return None

    # Sort strictly by concentration
    sorted_pairs = sorted(zip(concentrations, probabilities), key=lambda x: x[0])
    phis = np.array([p[0] for p in sorted_pairs], dtype=float)
    probs = np.array([p[1] for p in sorted_pairs], dtype=float)

    # Check whether target_probability is bracketed
    min_p, max_p = float(np.min(probs)), float(np.max(probs))
    if target_probability < min_p or target_probability > max_p:
        return None

    # Search for crossing interval
    # In case of exact match
    exact_matches = np.where(np.isclose(probs, target_probability, atol=1e-12))[0]
    if len(exact_matches) > 0:
        return float(np.mean(phis[exact_matches]))

    for i in range(len(phis) - 1):
        p1, p2 = probs[i], probs[i + 1]
        phi1, phi2 = phis[i], phis[i + 1]

        # Check if target is bracketed by [p1, p2]
        if (p1 <= target_probability <= p2) or (p2 <= target_probability <= p1):
            if np.isclose(p1, p2):
                return float(0.5 * (phi1 + phi2))
            fraction = (target_probability - p1) / (p2 - p1)
            return float(phi1 + fraction * (phi2 - phi1))

    return None


# ---------------------------------------------------------------------------
# Approach B: Logistic sigmoid fitting
# ---------------------------------------------------------------------------

def fit_logistic(
    concentrations: Sequence[float],
    probabilities: Sequence[float],
) -> LogisticFitResult | None:
    """Fit a logistic sigmoid curve P(φ) = 1 / (1 + exp(-k * (φ - φ_c))) (Approach B).

    Uses SciPy non-linear least squares optimization (curve_fit) with robust
    bounds and initial parameter heuristics.

    Parameters
    ----------
    concentrations : sequence of tested concentrations φ
    probabilities  : corresponding empirical percolation probabilities P(φ)

    Returns
    -------
    LogisticFitResult if fitting succeeds, or None if fitting fails or data is insufficient.
    """
    if len(concentrations) != len(probabilities):
        raise ValueError("concentrations and probabilities must have the same length.")
    if len(concentrations) < 3:
        return None

    # Sort by concentration
    sorted_pairs = sorted(zip(concentrations, probabilities), key=lambda x: x[0])
    phis = np.array([p[0] for p in sorted_pairs], dtype=float)
    probs = np.array([p[1] for p in sorted_pairs], dtype=float)

    # If all probabilities are identical (e.g. all 0 or all 1), cannot fit transition
    if np.allclose(probs, probs[0]):
        return None

    span = float(np.max(phis) - np.min(phis))
    if span <= 0.0:
        return None

    # Initial parameter estimates
    # Initial phi_c: linear threshold or midpoint
    linear_guess = estimate_threshold_linear(phis, probs, 0.5)
    phi_c_0 = linear_guess if linear_guess is not None else float(np.median(phis))

    # Initial k: scale so transition width roughly matches data span
    k_0 = max(1.0, 4.0 / span)

    p0 = [phi_c_0, k_0]

    # Parameter bounds: phi_c within expanded domain, k > 0
    bounds = (
        [float(np.min(phis) - 0.5 * span), 1e-3],
        [float(np.max(phis) + 0.5 * span), 1e6 / span],
    )

    try:
        popt, pcov = optimize.curve_fit(
            logistic_function,
            phis,
            probs,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
    except (RuntimeError, ValueError, optimize.OptimizeWarning):
        return None

    phi_c_opt, k_opt = float(popt[0]), float(popt[1])

    # Standard errors from covariance matrix diagonal
    if pcov is not None and np.all(np.isfinite(pcov)):
        diag = np.diagonal(pcov)
        phi_c_stderr = float(np.sqrt(max(0.0, diag[0])))
        k_stderr = float(np.sqrt(max(0.0, diag[1])))
    else:
        phi_c_stderr = None
        k_stderr = None

    # Compute R² coefficient of determination
    preds = logistic_function(phis, phi_c=phi_c_opt, k=k_opt)
    ss_res = float(np.sum((probs - preds) ** 2))
    ss_tot = float(np.sum((probs - np.mean(probs)) ** 2))

    if ss_tot > 1e-12:
        r_squared = float(max(0.0, 1.0 - (ss_res / ss_tot)))
    else:
        r_squared = 1.0 if ss_res < 1e-6 else 0.0

    return LogisticFitResult(
        phi_c=phi_c_opt,
        k=k_opt,
        phi_c_stderr=phi_c_stderr,
        k_stderr=k_stderr,
        r_squared=r_squared,
        converged=True,
    )


# ---------------------------------------------------------------------------
# High-level threshold estimation pipeline
# ---------------------------------------------------------------------------

def estimate_threshold(
    data: MonteCarloResult | Sequence[ConcentrationResult] | tuple[Sequence[float], Sequence[float]],
    confidence_level: float = 0.95,
    target_probability: float = 0.5,
    prefer_method: str = "logistic",
) -> ThresholdResult:
    """Analyze Monte Carlo simulation data to estimate percolation threshold.

    Parameters
    ----------
    data               : MonteCarloResult, sequence of ConcentrationResult,
                         or a (concentrations, probabilities) tuple
    confidence_level   : confidence level for Wilson score intervals (default 0.95)
    target_probability : target percolation probability (default 0.5)
    prefer_method      : preferred primary method ('logistic' or 'linear_interpolation')

    Returns
    -------
    ThresholdResult containing primary phi_c, linear & logistic estimates,
    parameter uncertainties, R², and confidence intervals.
    """
    confidence_intervals: list[ConfidenceInterval] = []

    # Parse input into concentrations and probabilities
    if hasattr(data, "concentration_results"):
        # MonteCarloResult
        conc_results = data.concentration_results
        concentrations = [r.concentration for r in conc_results]
        probabilities = [r.probability for r in conc_results]
        confidence_intervals = compute_confidence_intervals(conc_results, confidence_level)
    elif isinstance(data, (list, tuple)) and len(data) > 0 and hasattr(data[0], "percolating_samples"):
        # Sequence[ConcentrationResult]
        conc_results = data
        concentrations = [r.concentration for r in conc_results]
        probabilities = [r.probability for r in conc_results]
        confidence_intervals = compute_confidence_intervals(conc_results, confidence_level)
    elif isinstance(data, tuple) and len(data) == 2:
        # tuple[Sequence[float], Sequence[float]]
        concentrations, probabilities = list(data[0]), list(data[1])
        # Generate dummy confidence intervals assuming samples=100 if unknown
        for phi, p in zip(concentrations, probabilities):
            confidence_intervals.append(
                ConfidenceInterval(
                    concentration=phi,
                    probability=p,
                    ci_lower=p,
                    ci_upper=p,
                    confidence=confidence_level,
                )
            )
    else:
        raise TypeError(f"Unsupported data type for estimate_threshold: {type(data)}")

    # Approach A: Linear interpolation
    phi_c_linear = estimate_threshold_linear(
        concentrations=concentrations,
        probabilities=probabilities,
        target_probability=target_probability,
    )

    # Approach B: Logistic sigmoid fit
    logistic_fit = fit_logistic(
        concentrations=concentrations,
        probabilities=probabilities,
    )

    # Determine primary phi_c and primary method
    if prefer_method == "logistic" and logistic_fit is not None and logistic_fit.r_squared >= 0.5:
        primary_phi_c = logistic_fit.phi_c
        primary_method = "logistic"
    elif phi_c_linear is not None:
        primary_phi_c = phi_c_linear
        primary_method = "linear_interpolation"
    elif logistic_fit is not None:
        primary_phi_c = logistic_fit.phi_c
        primary_method = "logistic"
    else:
        primary_phi_c = None
        primary_method = "none"

    return ThresholdResult(
        phi_c=primary_phi_c,
        method=primary_method,
        phi_c_linear=phi_c_linear,
        logistic_fit=logistic_fit,
        confidence_intervals=confidence_intervals,
        confidence_level=confidence_level,
        target_probability=target_probability,
    )


def main() -> None:
    import argparse

    from percolation.domain.cnt import ConcentrationResult, MaterialBox, MonteCarloResult

    parser = argparse.ArgumentParser(description="Estimate CNT percolation threshold from simulation results")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input monte_carlo_results.json")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output threshold_results.json")
    parser.add_argument("--plot", "-p", type=str, default=None, help="Path to save percolation probability plot (.png)")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (default: 0.95)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = data.get("parameters", {})
    box_p = params.get("box", {})
    box = MaterialBox(
        size_x=float(box_p.get("size_x", 10.0)),
        size_y=float(box_p.get("size_y", 5.0)),
        size_z=float(box_p.get("size_z", 5.0)),
    )
    cnt_p = params.get("cnt", {})
    cnt_length = float(cnt_p.get("length", 1.0))
    cnt_diameter = float(cnt_p.get("diameter", 0.1))
    contact_threshold = float(params.get("contact_threshold", 0.05))
    random_seed = int(params.get("random_seed", 42))

    raw_results = data.get("results", [])
    conc_results = [
        ConcentrationResult(
            concentration=float(r["concentration"]),
            samples=int(r["samples"]),
            percolating_samples=int(r["percolating_samples"]),
            probability=float(r["probability"]),
            avg_contact_count=float(r.get("avg_contact_count", 0.0)),
            avg_nanotube_count=float(r.get("avg_nanotube_count", 0.0)),
        )
        for r in raw_results
    ]

    mc_result = MonteCarloResult(
        box=box,
        cnt_length=cnt_length,
        cnt_diameter=cnt_diameter,
        contact_threshold=contact_threshold,
        random_seed=random_seed,
        concentration_results=conc_results,
    )

    threshold_res = estimate_threshold(mc_result, confidence_level=args.confidence)

    print("=" * 60)
    print("  Percolation Threshold Estimation")
    print("=" * 60)
    print(f"Primary Method        : {threshold_res.method}")
    if threshold_res.phi_c is not None:
        if threshold_res.logistic_phi_c_stderr is not None:
            print(f"Estimated phi_c       : {threshold_res.phi_c:.6f} +/- {threshold_res.logistic_phi_c_stderr:.6f}")
        else:
            print(f"Estimated phi_c       : {threshold_res.phi_c:.6f}")
    else:
        print("Estimated phi_c       : None (threshold not reached)")

    if threshold_res.phi_c_linear is not None:
        print(f"Linear Interp phi_c   : {threshold_res.phi_c_linear:.6f}")

    if threshold_res.logistic_fit is not None and threshold_res.logistic_fit.converged:
        print(f"Logistic Steepness k  : {threshold_res.logistic_fit.k:.2f}")
        print(f"Logistic R^2          : {threshold_res.logistic_fit.r_squared:.4f}")
    print("=" * 60)

    if args.output:
        threshold_res.save_json(args.output)
        print(f"Saved threshold result to: {args.output}")

    if args.plot:
        from percolation.visualization.threshold_plot import plot_percolation_probability
        plot_percolation_probability(mc_result, threshold_result=threshold_res, save_path=args.plot)
        print(f"Saved percolation plot to: {args.plot}")


if __name__ == "__main__":
    main()

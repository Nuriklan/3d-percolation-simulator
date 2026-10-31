"""
Percolation probability curve and threshold plotting — Phase 3 implementation.

Provides publication-quality visualization for:
- Empirical percolation probability points P(φ) with Wilson confidence intervals.
- Fitted logistic sigmoid curve.
- Percolation threshold φ_c reference line and annotation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from percolation.analysis.threshold import ThresholdResult
    from percolation.domain.cnt import MonteCarloResult


def plot_percolation_probability(
    mc_result: MonteCarloResult,
    threshold_result: ThresholdResult | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show_fit: bool = True,
    show_confidence_intervals: bool = True,
    dpi: int = 300,
) -> matplotlib.figure.Figure:
    """Plot percolation probability P(φ) vs concentration φ with confidence intervals.

    Parameters
    ----------
    mc_result                 : MonteCarloResult containing concentration_results
    threshold_result          : optional precomputed ThresholdResult; if None, computed automatically
    title                     : plot title (default provides summary of parameters)
    save_path                 : optional path to save figure (PNG, PDF, SVG)
    ax                        : optional matplotlib Axes to plot onto
    show_fit                  : whether to draw fitted logistic curve
    show_confidence_intervals : whether to draw Wilson confidence errorbars
    dpi                       : resolution for saved figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    if threshold_result is None:
        from percolation.analysis.threshold import estimate_threshold
        threshold_result = estimate_threshold(mc_result)

    # Use existing Axes or create new Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)
    else:
        fig = ax.get_figure()

    phis = np.array(mc_result.concentrations, dtype=float)
    probs = np.array(mc_result.probabilities, dtype=float)

    # Plot empirical data with errorbars
    if show_confidence_intervals and threshold_result.confidence_intervals:
        y_err_lower = np.array([ci.error_minus for ci in threshold_result.confidence_intervals])
        y_err_upper = np.array([ci.error_plus for ci in threshold_result.confidence_intervals])
        yerr = [y_err_lower, y_err_upper]

        ax.errorbar(
            phis,
            probs,
            yerr=yerr,
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.5,
            capsize=3.5,
            capthick=1.2,
            label=f"Monte Carlo P(φ) ({int(threshold_result.confidence_level * 100)}% CI)",
            zorder=4,
        )
    else:
        ax.plot(
            phis,
            probs,
            "o",
            color="#1f77b4",
            markersize=6,
            label="Monte Carlo P(φ)",
            zorder=4,
        )

    # Plot logistic curve if available and requested
    if show_fit and threshold_result.logistic_fit is not None and threshold_result.logistic_fit.converged:
        fit = threshold_result.logistic_fit
        phi_min = min(float(np.min(phis)), fit.phi_c)
        phi_max = max(float(np.max(phis)), fit.phi_c)
        margin = 0.05 * (phi_max - phi_min) if (phi_max - phi_min) > 0 else 0.001
        smooth_phis = np.linspace(max(0.0, phi_min - margin), phi_max + margin, 300)
        smooth_probs = fit.predict(smooth_phis)

        fit_label = f"Logistic Fit (R² = {fit.r_squared:.3f})"
        ax.plot(
            smooth_phis,
            smooth_probs,
            "-",
            color="#d62728",
            linewidth=2.0,
            label=fit_label,
            zorder=3,
        )

    # Reference line at target probability P = 0.5
    ax.axhline(
        threshold_result.target_probability,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"P = {threshold_result.target_probability}",
    )

    # Vertical threshold line at phi_c
    if threshold_result.phi_c is not None:
        phi_c_val = threshold_result.phi_c
        stderr = threshold_result.logistic_phi_c_stderr
        if stderr is not None:
            phi_c_str = f"φ_c = {phi_c_val:.4f} ± {stderr:.4f}"
        else:
            phi_c_str = f"φ_c ≈ {phi_c_val:.4f}"

        ax.axvline(
            phi_c_val,
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.6,
            label=f"Threshold ({phi_c_str})",
            zorder=2,
        )

    # Formatting and aesthetics
    ax.set_xlabel("CNT Concentration φ (volumetric fraction)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Percolation Probability P(φ)", fontsize=11, fontweight="medium")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0.0)

    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    else:
        ar = mc_result.cnt_length / mc_result.cnt_diameter if mc_result.cnt_diameter > 0 else 0.0
        ax.set_title(
            f"CNT Percolation Transition (L={mc_result.cnt_length} μm, AR={ar:.1f}, "
            f"Box={mc_result.box.size_x}×{mc_result.box.size_y}×{mc_result.box.size_z} μm)",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig

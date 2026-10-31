"""
Aspect ratio scaling and percolation transition visualization — Phase 6 implementation.

Generates a two-panel publication-quality infographic (300 DPI):
- Panel A: Family of percolation probability S-curves P(φ) across various Aspect Ratios (AR).
- Panel B: Scaling law of critical threshold φ_c vs AR on a log-log scale with Balberg theory comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from percolation.analysis.aspect_ratio import AspectRatioStudyResult


def plot_aspect_ratio_study(
    study_result: AspectRatioStudyResult,
    save_path: str | Path | None = "results/aspect_ratio_scaling.png",
    dpi: int = 300,
    figsize: tuple[float, float] = (14.0, 6.0),
    palette: str = "viridis",
) -> matplotlib.figure.Figure:
    """Plot two-panel infographic of aspect ratio scaling and S-curve family.

    Parameters
    ----------
    study_result : AspectRatioStudyResult containing points and power law fit
    save_path    : optional path to save high-resolution figure (PNG, PDF, SVG)
    dpi          : figure resolution in dots per inch (default: 300)
    figsize      : figure width and height in inches (default: (14.0, 6.0))
    palette      : matplotlib colormap name for distinct AR curves

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_s, ax_log) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    points = sorted(study_result.points, key=lambda p: p.aspect_ratio)
    n_pts = len(points)

    # Colormap for distinct AR curves
    cmap = matplotlib.colormaps.get_cmap(palette)
    colors = [cmap(val) for val in np.linspace(0.15, 0.88, max(n_pts, 2))]

    # =======================================================================
    # Panel A: Family of S-curves P(φ)
    # =======================================================================
    ax_s.set_title(
        "(a) Percolation Probability Curves $P(\\phi)$ vs Concentration",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_s.set_xlabel("CNT Volumetric Concentration $\\phi$ (volume fraction)", fontsize=11, labelpad=8)
    ax_s.set_ylabel("Percolation Probability $P(\\phi)$", fontsize=11, labelpad=8)
    ax_s.set_ylim(-0.04, 1.05)
    ax_s.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Reference line at P = 0.5
    ax_s.axhline(
        0.5,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Threshold Criterion ($P = 0.5$)",
        zorder=2,
    )

    for idx, pt in enumerate(points):
        col = colors[idx]
        phis = np.array(pt.concentrations, dtype=float)
        probs = np.array(pt.probabilities, dtype=float)

        # Draw error bars if Wilson confidence intervals are present
        if pt.threshold_result and pt.threshold_result.confidence_intervals:
            cis = pt.threshold_result.confidence_intervals
            yerr_lower = np.array([ci.error_minus for ci in cis])
            yerr_upper = np.array([ci.error_plus for ci in cis])
            ax_s.errorbar(
                phis,
                probs,
                yerr=[yerr_lower, yerr_upper],
                fmt="o",
                color=col,
                ecolor=col,
                elinewidth=1.2,
                capsize=3.0,
                capthick=1.0,
                markersize=5,
                zorder=4,
            )
        else:
            ax_s.plot(phis, probs, "o", color=col, markersize=5, zorder=4)

        # Draw smooth fitted logistic curve if available
        fit = pt.threshold_result.logistic_fit if pt.threshold_result else None
        if fit is not None and fit.converged:
            span = np.max(phis) - np.min(phis)
            smooth_x = np.linspace(
                max(0.0, np.min(phis) - 0.1 * span),
                np.max(phis) + 0.1 * span,
                200,
            )
            smooth_y = fit.predict(smooth_x)
            lbl = f"$AR = {pt.aspect_ratio:g}$ ($\\phi_c \\approx {pt.phi_c:.4f}$)"
            ax_s.plot(smooth_x, smooth_y, "-", color=col, linewidth=2.0, label=lbl, zorder=3)
        else:
            lbl = f"$AR = {pt.aspect_ratio:g}$ ($\\phi_c \\approx {pt.phi_c:.4f}$)"
            ax_s.plot(phis, probs, "-", color=col, linewidth=1.8, label=lbl, zorder=3)

    ax_s.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.0)

    # Annotation arrow indicating the transition shift
    if n_pts >= 2:
        ax_s.annotate(
            "Increasing Aspect Ratio ($AR \\uparrow$)\n$\\rightarrow$ shift to lower $\\phi_c$",
            xy=(points[-1].phi_c, 0.45),
            xytext=(points[0].phi_c * 0.7, 0.2),
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2, "ls": "--"},
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "fc": "#ffffff", "ec": "#cccccc", "alpha": 0.9},
            zorder=5,
        )

    # =======================================================================
    # Panel B: Scaling law phi_c vs AR (Log-Log)
    # =======================================================================
    ax_log.set_title(
        "(b) Scaling of Critical Threshold: $\\phi_c \\propto AR^{-\\beta}$",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_log.set_xlabel("Aspect Ratio $AR = L / D$", fontsize=11, labelpad=8)
    ax_log.set_ylabel("Percolation Threshold $\\phi_c$", fontsize=11, labelpad=8)
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.grid(True, which="both", linestyle="--", alpha=0.3, zorder=0)

    ar_data = np.array([p.aspect_ratio for p in points], dtype=float)
    phi_data = np.array([p.phi_c for p in points], dtype=float)
    phi_errs = np.array(
        [p.phi_c_stderr if p.phi_c_stderr is not None else 0.0 for p in points],
        dtype=float,
    )

    # Smooth curve for fitted power law
    ar_fine = np.logspace(
        np.log10(np.min(ar_data) * 0.85),
        np.log10(np.max(ar_data) * 1.15),
        200,
    )
    phi_pred = study_result.predict(ar_fine)

    # Theoretical Balberg line: phi_c ~ AR^(-1.0), anchored at geometric mean
    geo_ar = np.exp(np.mean(np.log(ar_data)))
    geo_phi = np.exp(np.mean(np.log(phi_data)))
    c_theory = geo_phi * (geo_ar ** 1.0)
    phi_theory = c_theory * np.power(ar_fine, -1.0)

    # Draw theoretical line
    ax_log.plot(
        ar_fine,
        phi_theory,
        "--",
        color="#888888",
        linewidth=1.8,
        label="Balberg Theory: $\\phi_c \\propto AR^{-1.0}$",
        zorder=2,
    )

    # Draw fitted power law line
    fit_label = f"Fitted Power Law: $\\phi_c = {study_result.c_coefficient:.3f} \\cdot AR^{{-{study_result.beta:.3f}}}$"
    ax_log.plot(
        ar_fine,
        phi_pred,
        "-",
        color="#d62728",
        linewidth=2.2,
        label=fit_label,
        zorder=3,
    )

    # Draw empirical data points
    has_nonzero_errs = np.any(phi_errs > 0)
    if has_nonzero_errs:
        ax_log.errorbar(
            ar_data,
            phi_data,
            yerr=phi_errs,
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.5,
            capsize=4.0,
            capthick=1.2,
            markersize=7,
            label="Simulated $\\phi_c$ (Monte Carlo)",
            zorder=4,
        )
    else:
        ax_log.plot(
            ar_data,
            phi_data,
            "o",
            color="#1f77b4",
            markersize=7,
            label="Simulated $\\phi_c$ (Monte Carlo)",
            zorder=4,
        )

    # Add text box with fit metrics
    err_txt = f" \\pm {study_result.beta_stderr:.3f}" if study_result.beta_stderr is not None else ""
    metrics_box = (
        f"$\\mathbf{{Power\\ Law\\ Fit:}}$\n"
        f"$\\beta = {study_result.beta:.3f}{err_txt}$\n"
        f"$C = {study_result.c_coefficient:.4f}$\n"
        f"$R^2 = {study_result.r_squared:.4f}$\n"
        f"Theory: $\\beta \\approx 1.00$"
    )
    ax_log.text(
        0.05,
        0.06,
        metrics_box,
        transform=ax_log.transAxes,
        fontsize=9.5,
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f8f9fa", "edgecolor": "#333333", "alpha": 0.9},
        zorder=5,
    )

    ax_log.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.0)

    # Format ticks cleanly
    ax_log.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_log.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # =======================================================================
    # Overall Title and Layout
    # =======================================================================
    fig.suptitle(
        "CNT Percolation Simulator — Aspect Ratio Scaling Study (Balberg Law)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.01,
        "Simulated with 3D Monte Carlo + Spatial Partitioning Graph Engine",
        ha="center",
        fontsize=8.5,
        color="#555555",
        style="italic",
    )

    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.96, wspace=0.25)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig

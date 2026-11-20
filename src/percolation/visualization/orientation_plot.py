"""
Orientation study visualization - Phase 7 implementation.

Generates a two-panel publication-quality figure (300 DPI):
- Panel (a): Family of percolation probability S-curves P(phi) across CNT orientations.
- Panel (b): Critical percolation threshold phi_c vs 3D Hermans order parameter S_x,
             highlighting isotropic reference, optimal alignment, and alignment penalty.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from percolation.analysis.orientation import OrientationPoint, OrientationStudyResult


def _short_annotation(pt: OrientationPoint) -> str:
    """Concise point annotation for Panel (b)."""
    if pt.orientation.mode == "isotropic":
        return "Iso (90 deg)"
    if pt.orientation.mode == "aligned":
        return pt.label
    if pt.orientation.mode == "partially_aligned":
        return f"{pt.max_angle_deg:g} deg"
    if pt.orientation.mode == "fisher":
        return f"k={pt.orientation.kappa:g}"
    return pt.label


def plot_orientation_study(
    study_result: OrientationStudyResult,
    save_path: str | Path | None = "results/orientation_study.png",
    dpi: int = 300,
    figsize: tuple[float, float] = (14.5, 6.2),
    palette: str = "plasma",
) -> matplotlib.figure.Figure:
    """Plot two-panel infographic of CNT orientation effects on percolation.

    Parameters
    ----------
    study_result : OrientationStudyResult containing evaluated OrientationPoints
    save_path    : optional path to save high-resolution figure (PNG, PDF, SVG)
    dpi          : figure resolution in dots per inch (default: 300)
    figsize      : figure width and height in inches (default: (14.5, 6.2))
    palette      : matplotlib colormap name for distinct orientation curves

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_s, ax_ord) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Sort points by theoretical Hermans order parameter S_x (from transverse -> isotropic -> aligned)
    points = sorted(study_result.points, key=lambda p: p.theoretical_order_parameter)
    n_pts = len(points)

    cmap = matplotlib.colormaps.get_cmap(palette)
    colors = [cmap(val) for val in np.linspace(0.08, 0.88, max(n_pts, 2))]

    # =======================================================================
    # Panel A: Family of S-curves P(phi)
    # =======================================================================
    ax_s.set_title(
        "(a) Percolation Probability Curves $P(\\phi)$ by CNT Orientation",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_s.set_xlabel("CNT Volumetric Concentration $\\phi$ (volume fraction)", fontsize=11, labelpad=8)
    ax_s.set_ylabel("Percolation Probability $P(\\phi)$", fontsize=11, labelpad=8)
    ax_s.set_ylim(-0.04, 1.05)
    ax_s.grid(True, linestyle="--", alpha=0.3, zorder=0)

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
        if phis.size == 0:
            continue

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
                elinewidth=1.1,
                capsize=2.8,
                capthick=1.0,
                markersize=4.5,
                zorder=4,
            )
        else:
            ax_s.plot(phis, probs, "o", color=col, markersize=4.5, zorder=4)

        fit = pt.threshold_result.logistic_fit if pt.threshold_result else None
        lbl = f"{pt.label} ($S_x={pt.theoretical_order_parameter:+.2f}$, $\\phi_c={pt.phi_c:.4f}$)"
        if fit is not None and fit.converged:
            span = float(np.max(phis) - np.min(phis))
            smooth_x = np.linspace(
                max(0.0, float(np.min(phis)) - 0.08 * span),
                float(np.max(phis)) + 0.08 * span,
                200,
            )
            smooth_y = fit.predict(smooth_x)
            ax_s.plot(smooth_x, smooth_y, "-", color=col, linewidth=2.0, label=lbl, zorder=3)
        else:
            ax_s.plot(phis, probs, "-", color=col, linewidth=1.8, label=lbl, zorder=3)

    ax_s.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.3)

    # =======================================================================
    # Panel B: Critical Threshold phi_c vs Hermans Order Parameter S_x
    # =======================================================================
    ax_ord.set_title(
        "(b) Percolation Threshold $\\phi_c$ vs Hermans Order Parameter $S_x$",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_ord.set_xlabel(
        "Hermans Orientation Factor $S_x = \\frac{1}{2}(3\\langle\\cos^2\\theta_x\\rangle - 1)$",
        fontsize=11,
        labelpad=8,
    )
    ax_ord.set_ylabel("Percolation Threshold $\\phi_c$", fontsize=11, labelpad=8)
    ax_ord.grid(True, linestyle="--", alpha=0.3, zorder=0)

    s_vals = np.array([p.theoretical_order_parameter for p in points], dtype=float)
    phi_vals = np.array([p.phi_c for p in points], dtype=float)
    phi_errs = np.array(
        [p.phi_c_stderr if p.phi_c_stderr is not None else 0.0 for p in points],
        dtype=float,
    )

    # Reference vertical guides
    ax_ord.axvline(0.0, color="#64748b", linestyle=":", linewidth=1.3, alpha=0.8, zorder=1)
    ax_ord.axvline(1.0, color="#64748b", linestyle=":", linewidth=1.3, alpha=0.8, zorder=1)
    if np.any(s_vals < -0.1):
        ax_ord.axvline(-0.5, color="#64748b", linestyle=":", linewidth=1.3, alpha=0.8, zorder=1)

    # Isotropic reference horizontal line
    iso_phi = study_result.isotropic_phi_c
    if iso_phi is not None:
        ax_ord.axhline(
            iso_phi,
            color="#475569",
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label=f"Isotropic Baseline ($\\phi_c^{{\\mathrm{{iso}}}} = {iso_phi:.4f}$)",
            zorder=2,
        )

    # Connecting trend line + empirical markers
    ax_ord.plot(
        s_vals,
        phi_vals,
        "-",
        color="#1f77b4",
        linewidth=2.0,
        alpha=0.85,
        zorder=3,
    )

    has_errs = bool(np.any(phi_errs > 0))
    if has_errs:
        ax_ord.errorbar(
            s_vals,
            phi_vals,
            yerr=phi_errs,
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.4,
            capsize=4.0,
            capthick=1.1,
            markersize=7,
            label="Simulated $\\phi_c$ (Monte Carlo)",
            zorder=4,
        )
    else:
        ax_ord.plot(
            s_vals,
            phi_vals,
            "o",
            color="#1f77b4",
            markersize=7,
            label="Simulated $\\phi_c$ (Monte Carlo)",
            zorder=4,
        )

    # Highlight optimal (minimum phi_c) point
    opt = study_result.optimal_point
    if opt is not None:
        ax_ord.plot(
            [opt.theoretical_order_parameter],
            [opt.phi_c],
            marker="*",
            color="#dc2626",
            markersize=14,
            linestyle="None",
            label=f"Lowest $\\phi_c$: {opt.label} ($\\phi_c={opt.phi_c:.4f}$)",
            zorder=5,
        )

    # Annotate each point with its cone angle / mode
    for idx, pt in enumerate(points):
        s_x = pt.theoretical_order_parameter
        offset_y = 9 if idx % 2 == 0 else -14
        offset_x = 0
        ha = "center"
        if n_pts >= 2 and idx == 0 and s_x < -0.2:
            offset_x = 6
            offset_y = 6
            ha = "left"
        elif n_pts >= 2 and idx == n_pts - 1 and s_x > 0.85:
            offset_x = -6
            offset_y = 8
            ha = "right"
        ax_ord.annotate(
            _short_annotation(pt),
            xy=(s_x, pt.phi_c),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha=ha,
            fontsize=8.0,
            color="#1e293b",
            zorder=6,
        )

    # Axis limits with padding
    x_min = min(-0.08, float(np.min(s_vals)) - 0.14) if n_pts > 0 else -0.1
    x_max = max(1.08, float(np.max(s_vals)) + 0.14) if n_pts > 0 else 1.1
    ax_ord.set_xlim(x_min, x_max)
    if n_pts > 0:
        y_max = float(np.max(phi_vals + phi_errs))
        y_min = float(np.min(np.maximum(0.0, phi_vals - phi_errs)))
        y_span = max(y_max - y_min, y_max * 0.2, 1e-4)
        ax_ord.set_ylim(max(0.0, y_min - 0.18 * y_span), y_max + 0.28 * y_span)

    # Summary info box positioned in the open upper-center region of the U-curve
    summary_lines = [
        (
            f"Box: {study_result.box_size[0]:g}x{study_result.box_size[1]:g}x{study_result.box_size[2]:g} um | "
            f"L={study_result.cnt_length:g}, D={study_result.cnt_diameter:g} (AR={study_result.aspect_ratio:g})"
        ),
    ]
    if iso_phi is not None:
        summary_lines.append(f"Isotropic $\\phi_c^{{\\mathrm{{iso}}}} = {iso_phi:.5f}$")
    if opt is not None:
        rel = study_result.relative_threshold(opt)
        rel_str = f" ({rel:.2f}x iso)" if rel is not None else ""
        summary_lines.append(f"Lowest: {opt.label} = {opt.phi_c:.5f}{rel_str}")

    ax_ord.text(
        0.50,
        0.96,
        "\n".join(summary_lines),
        transform=ax_ord.transAxes,
        fontsize=8.6,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8f9fa", "edgecolor": "#333333", "alpha": 0.92},
        zorder=6,
    )

    ax_ord.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 0.77),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        fontsize=8.4,
    )

    # =======================================================================
    # Overall Title & Layout
    # =======================================================================
    fig.suptitle(
        "CNT Percolation Simulator - Orientation Dependence Study $\\phi_c(S_x)$",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Simulated with 3D Monte Carlo + Spatial Partitioning Graph Engine",
        ha="center",
        fontsize=8.5,
        color="#555555",
        style="italic",
    )

    plt.subplots_adjust(top=0.89, bottom=0.14, left=0.07, right=0.97, wspace=0.24)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig

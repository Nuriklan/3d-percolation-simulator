"""
Tunneling Contact Conductance Visualization — Phase 8 implementation.

Provides:
1. ``plot_tunneling_study``: Four-panel publication-quality figure (300 DPI):
   - Panel (a): Exponential tunneling conductance law G(d) / G_0 = exp(-alpha * d_eff)
                on semi-log scale with cutoff/threshold annotations.
   - Panel (b): Percolation probability S-curves P(phi) across tunneling parameters.
   - Panel (c): Critical percolation threshold phi_c vs alpha (or vs d_cutoff).
   - Panel (d): Critical-path (bottleneck) conductance G_c(phi) on semi-log scale.

2. ``plot_contact_conductance_distribution``: Two-panel single-sample diagnostic:
   - Panel (a): Distribution of contact distances d_ij (all vs percolating cluster).
   - Panel (b): Distribution of log10(G_ij) with bottleneck conductance G_c marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from percolation.analysis.tunneling import TunnelingStudyResult
    from percolation.domain.cnt import SampleState


def plot_tunneling_study(
    study_result: TunnelingStudyResult,
    save_path: str | Path | None = "results/tunneling_study.png",
    dpi: int = 300,
    figsize: tuple[float, float] = (14.5, 10.2),
    palette: str = "viridis",
) -> matplotlib.figure.Figure:
    """Plot 2x2 publication-quality summary of a Phase 8 tunneling conductance study.

    Parameters
    ----------
    study_result : TunnelingStudyResult from run_tunneling_alpha_study or run_contact_cutoff_study
    save_path    : optional path to save the figure (.png, .pdf, .svg)
    dpi          : figure resolution in DPI (default: 300)
    figsize      : figure dimensions (width, height) in inches
    palette      : matplotlib colormap name

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    ax_law, ax_prob = axes[0, 0], axes[0, 1]
    ax_thresh, ax_bn = axes[1, 0], axes[1, 1]

    points = sorted(study_result.points, key=lambda p: p.parameter_value)
    n_pts = len(points)
    cmap = matplotlib.colormaps.get_cmap(palette)
    colors = [cmap(val) for val in np.linspace(0.08, 0.88, max(n_pts, 2))]

    is_alpha_mode = study_result.study_mode == "alpha"

    # =======================================================================
    # Panel (a): Exponential Tunneling Law G(d) / G_0
    # =======================================================================
    ax_law.set_title(
        "(a) Tunneling Contact Law $G(d) / G_0 = e^{-\\alpha \\max(0, d - d_{\\min})}$",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    ax_law.set_xlabel("Contact Gap Distance $d$ ($\\mu$m)", fontsize=10.5)
    ax_law.set_ylabel("Normalized Conductance $G(d) / G_0$", fontsize=10.5)
    ax_law.set_yscale("log")
    ax_law.grid(True, which="both", linestyle="--", alpha=0.3)

    max_d = max(
        [study_result.contact_threshold] + [max(pt.effective_cutoff, 0.0) for pt in points]
    )
    max_d = max(max_d, 0.02)
    d_grid = np.linspace(0.0, max_d * 1.05, 300)

    min_g_ratios: list[float] = []
    for idx, pt in enumerate(points):
        col = colors[idx]
        g_norm = pt.tunneling.conductance_array(d_grid) / pt.tunneling.g0
        positive_mask = g_norm > 1e-12
        ax_law.plot(
            d_grid[positive_mask],
            g_norm[positive_mask],
            color=col,
            linewidth=2.0,
            label=f"{pt.label} ($d_{{eff}}={pt.effective_cutoff:.3f}\\,\\mu$m)",
        )
        if 0.0 <= pt.effective_cutoff <= max_d * 1.05:
            g_at_cut = pt.tunneling.conductance(pt.effective_cutoff) / pt.tunneling.g0
            if g_at_cut > 1e-12:
                ax_law.plot(
                    pt.effective_cutoff,
                    g_at_cut,
                    "o",
                    color=col,
                    markersize=6.5,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    zorder=5,
                )
        if pt.tunneling.min_conductance is not None:
            min_g_ratios.append(pt.tunneling.min_conductance / pt.tunneling.g0)

    if min_g_ratios:
        g_min_ratio = min_g_ratios[0]
        ax_law.axhline(
            g_min_ratio,
            color="#dc2626",
            linestyle=":",
            linewidth=1.5,
            label=f"Conductance Cutoff $G_{{\\min}}/G_0 = {g_min_ratio:.1e}$",
        )

    ax_law.legend(loc="lower left", fontsize=8.0, framealpha=0.9)

    # =======================================================================
    # Panel (b): Percolation Probability S-Curves P(phi)
    # =======================================================================
    ax_prob.set_title(
        "(b) Percolation Probability Curves $P(\\phi)$",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    ax_prob.set_xlabel("CNT Volumetric Concentration $\\phi$", fontsize=10.5)
    ax_prob.set_ylabel("Percolation Probability $P(\\phi)$", fontsize=10.5)
    ax_prob.set_ylim(-0.04, 1.05)
    ax_prob.grid(True, linestyle="--", alpha=0.3)

    ax_prob.axhline(
        0.5,
        color="#64748b",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label="$P = 0.5$ Threshold",
    )

    for idx, pt in enumerate(points):
        col = colors[idx]
        phis = np.array(pt.concentrations, dtype=float)
        probs = np.array(pt.probabilities, dtype=float)
        if phis.size == 0:
            continue

        if pt.threshold_result and pt.threshold_result.confidence_intervals:
            cis = pt.threshold_result.confidence_intervals
            yerr_lo = np.array([ci.error_minus for ci in cis])
            yerr_hi = np.array([ci.error_plus for ci in cis])
            ax_prob.errorbar(
                phis,
                probs,
                yerr=[yerr_lo, yerr_hi],
                fmt="o",
                color=col,
                ecolor=col,
                elinewidth=1.0,
                capsize=2.5,
                markersize=4.5,
                zorder=4,
            )
        else:
            ax_prob.plot(phis, probs, "o", color=col, markersize=4.5, zorder=4)

        phi_c_txt = f"{pt.phi_c:.4f}" if pt.phi_c is not None else "N/A"
        lbl = f"{pt.label} ($\\phi_c={phi_c_txt}$)"

        fit = pt.threshold_result.logistic_fit if pt.threshold_result else None
        if fit is not None and fit.converged:
            phi_smooth = np.linspace(float(phis.min()), float(phis.max()), 200)
            p_smooth = 1.0 / (1.0 + np.exp(-fit.k * (phi_smooth - fit.phi_c)))
            ax_prob.plot(phi_smooth, p_smooth, color=col, linewidth=2.0, label=lbl, zorder=3)
        else:
            ax_prob.plot(phis, probs, "-", color=col, linewidth=1.6, alpha=0.85, label=lbl, zorder=3)

    ax_prob.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    # =======================================================================
    # Panel (c): Percolation Threshold phi_c vs Parameter
    # =======================================================================
    if is_alpha_mode:
        ax_thresh.set_title(
            "(c) Percolation Threshold $\\phi_c$ vs Tunneling Decay $\\alpha$",
            fontsize=11.5,
            fontweight="bold",
            pad=8,
        )
        ax_thresh.set_xlabel("Tunneling Decay Constant $\\alpha$ ($\\mu\\mathrm{m}^{-1}$)", fontsize=10.5)
    else:
        ax_thresh.set_title(
            "(c) Percolation Threshold $\\phi_c$ vs Cutoff Distance $d_{\\mathrm{contact}}$",
            fontsize=11.5,
            fontweight="bold",
            pad=8,
        )
        ax_thresh.set_xlabel("Contact Cutoff Distance $d_{\\mathrm{contact}}$ ($\\mu$m)", fontsize=10.5)

    ax_thresh.set_ylabel("Percolation Threshold $\\phi_c$", fontsize=10.5)
    ax_thresh.grid(True, linestyle="--", alpha=0.3)

    valid_pts = [(idx, pt) for idx, pt in enumerate(points) if pt.phi_c is not None]
    if valid_pts:
        xs = np.array([pt.parameter_value for _, pt in valid_pts], dtype=float)
        ys = np.array([float(pt.phi_c) for _, pt in valid_pts], dtype=float)
        yerrs = np.array(
            [pt.phi_c_stderr if pt.phi_c_stderr is not None else 0.0 for _, pt in valid_pts],
            dtype=float,
        )

        ax_thresh.plot(xs, ys, "-", color="#1e293b", linewidth=1.8, alpha=0.7, zorder=2)
        for (idx, pt), x_v, y_v, e_v in zip(valid_pts, xs, ys, yerrs):
            ax_thresh.errorbar(
                x_v,
                y_v,
                yerr=e_v if e_v > 0 else None,
                fmt="o",
                color=colors[idx],
                ecolor=colors[idx],
                markersize=8.0,
                markeredgecolor="white",
                markeredgewidth=1.0,
                capsize=4.0,
                zorder=4,
            )

    # =======================================================================
    # Panel (d): Critical-Path (Bottleneck) Conductance G_c(phi)
    # =======================================================================
    ax_bn.set_title(
        "(d) Critical-Path (Bottleneck) Conductance $G_c(\\phi)$",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    ax_bn.set_xlabel("CNT Volumetric Concentration $\\phi$", fontsize=10.5)
    ax_bn.set_ylabel("Mean Bottleneck Conductance $G_c$ (S)", fontsize=10.5)
    ax_bn.set_yscale("log")
    ax_bn.grid(True, which="both", linestyle="--", alpha=0.3)

    has_bn_data = False
    for idx, pt in enumerate(points):
        col = colors[idx]
        x_bn = []
        y_bn = []
        for phi_v, g_v in zip(pt.concentrations, pt.avg_bottleneck_conductances):
            if g_v is not None and g_v > 0.0:
                x_bn.append(phi_v)
                y_bn.append(g_v)
        if x_bn:
            has_bn_data = True
            ax_bn.plot(
                x_bn,
                y_bn,
                "o-",
                color=col,
                linewidth=2.0,
                markersize=5.5,
                label=pt.label,
            )

    if has_bn_data:
        ax_bn.legend(loc="lower right", fontsize=8.0, framealpha=0.9)
    else:
        ax_bn.text(
            0.5,
            0.5,
            "No percolating samples recorded",
            transform=ax_bn.transAxes,
            ha="center",
            va="center",
            color="#64748b",
            fontsize=10,
        )

    box = study_result.box
    ar = study_result.cnt_length / study_result.cnt_diameter if study_result.cnt_diameter > 0 else 0.0
    fig.suptitle(
        f"Phase 8: Tunneling Contact Conductance Study (mode={study_result.study_mode})\n"
        f"Box={box.size_x:g}x{box.size_y:g}x{box.size_z:g} um | "
        f"L={study_result.cnt_length:g} um, D={study_result.cnt_diameter:g} um (AR={ar:g}) | "
        f"Samples/conc={study_result.samples_per_concentration}",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_contact_conductance_distribution(
    sample: SampleState,
    save_path: str | Path | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] = (13.0, 5.2),
) -> matplotlib.figure.Figure:
    """Plot contact gap distance and log10 tunneling conductance histograms for a sample.

    Distinguishes all network contacts from contacts inside the percolating cluster,
    and marks the critical bottleneck conductance G_c when the sample percolates.
    """
    fig, (ax_d, ax_g) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    all_dists = [c.distance for c in sample.contacts]
    cluster_set = sample.percolating_cluster if sample.percolates else set()
    cluster_contacts = [
        c for c in sample.contacts if c.tube_a in cluster_set and c.tube_b in cluster_set
    ]
    cluster_dists = [c.distance for c in cluster_contacts]

    # Panel (a): Gap distances
    ax_d.set_title("(a) Contact Distance Distribution", fontsize=11.5, fontweight="bold")
    ax_d.set_xlabel("Contact Gap Distance $d_{ij}$ ($\\mu$m)", fontsize=10.5)
    ax_d.set_ylabel("Count", fontsize=10.5)
    ax_d.grid(True, linestyle="--", alpha=0.3)

    if all_dists:
        bins_d = min(30, max(8, int(np.sqrt(len(all_dists)))))
        ax_d.hist(
            all_dists,
            bins=bins_d,
            color="#94a3b8",
            alpha=0.7,
            edgecolor="#475569",
            label=f"All Contacts (N={len(all_dists)})",
        )
        if cluster_dists:
            ax_d.hist(
                cluster_dists,
                bins=bins_d,
                color="#2563eb",
                alpha=0.7,
                edgecolor="#1d4ed8",
                label=f"Percolating Cluster (N={len(cluster_dists)})",
            )
        ax_d.legend(fontsize=8.5, framealpha=0.9)
    else:
        ax_d.text(0.5, 0.5, "No contacts found", transform=ax_d.transAxes, ha="center", va="center")

    # Panel (b): log10(Conductance)
    ax_g.set_title("(b) Tunneling Conductance Distribution", fontsize=11.5, fontweight="bold")
    ax_g.set_xlabel("$\\log_{10}(G_{ij}\\,[\\mathrm{S}])$", fontsize=10.5)
    ax_g.set_ylabel("Count", fontsize=10.5)
    ax_g.grid(True, linestyle="--", alpha=0.3)

    all_log_g = [
        float(np.log10(c.conductance))
        for c in sample.contacts
        if c.conductance is not None and c.conductance > 0.0
    ]
    cluster_log_g = [
        float(np.log10(c.conductance))
        for c in cluster_contacts
        if c.conductance is not None and c.conductance > 0.0
    ]

    if all_log_g:
        bins_g = min(30, max(8, int(np.sqrt(len(all_log_g)))))
        ax_g.hist(
            all_log_g,
            bins=bins_g,
            color="#94a3b8",
            alpha=0.7,
            edgecolor="#475569",
            label=f"All Contacts (N={len(all_log_g)})",
        )
        if cluster_log_g:
            ax_g.hist(
                cluster_log_g,
                bins=bins_g,
                color="#f59e0b",
                alpha=0.75,
                edgecolor="#b45309",
                label=f"Percolating Cluster (N={len(cluster_log_g)})",
            )
        if sample.bottleneck_conductance is not None and sample.bottleneck_conductance > 0.0:
            log_gc = float(np.log10(sample.bottleneck_conductance))
            ax_g.axvline(
                log_gc,
                color="#dc2626",
                linestyle="--",
                linewidth=2.0,
                label=f"Bottleneck $G_c = {sample.bottleneck_conductance:.2e}$ S",
            )
        ax_g.legend(fontsize=8.5, framealpha=0.9)
    else:
        ax_g.text(
            0.5,
            0.5,
            "No tunneling conductance data",
            transform=ax_g.transAxes,
            ha="center",
            va="center",
        )

    tunn_lbl = sample.tunneling.label if sample.tunneling is not None else "Binary"
    fig.suptitle(
        f"Single-Sample Contact Conductance Diagnostics | phi={sample.concentration:.4f} | {tunn_lbl}",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig

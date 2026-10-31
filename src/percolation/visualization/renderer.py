"""
3D Visualization of CNT composite samples — Phase 4 implementation.

Renders:
- 3D MaterialBox boundary wireframe.
- Planar semi-transparent LEFT (x=0) and RIGHT (x=Lx) electrodes.
- Carbon Nanotubes (CNTs) line segments categorized into:
    * Non-percolating / isolated tubes.
    * Percolating cluster tubes.
    * Shortest conductive backbone path (LEFT → RIGHT).
- Physical contact junctions between nanotubes.
- High-resolution image export (PNG, PDF, SVG).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from percolation.geometry.distance import nanotube_closest_points

if TYPE_CHECKING:
    from percolation.domain.cnt import SampleState


def plot_sample_3d(
    sample: SampleState,
    show_electrodes: bool = True,
    show_box: bool = True,
    show_tubes: bool = True,
    show_contacts: bool = True,
    highlight_cluster: bool = True,
    highlight_path: bool = True,
    ax: Axes3D | None = None,
    figsize: tuple[float, float] = (11, 8.5),
    view_elevation: float = 22.0,
    view_azimuth: float = -65.0,
    save_path: str | Path | None = None,
    dpi: int = 300,
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Render a 3D visualization of a single CNT percolation sample.

    Parameters
    ----------
    sample            : SampleState containing box, tubes, contacts, and graph topology
    show_electrodes   : whether to draw semi-transparent planar electrodes
    show_box          : whether to draw the bounding wireframe of the material box
    show_tubes        : whether to render CNT line segments
    show_contacts     : whether to render contact points at tube intersections
    highlight_cluster : whether to color tubes in the percolating cluster distinctly
    highlight_path    : whether to highlight the shortest conductive LEFT-to-RIGHT path
    ax                : optional pre-existing 3D Axes
    figsize           : figure width and height in inches
    view_elevation    : camera elevation angle in degrees
    view_azimuth      : camera azimuthal angle in degrees
    save_path         : optional path to save figure (.png, .pdf, .svg)
    dpi               : resolution for saved figure
    title             : custom title string (default provides parameter summary)

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    box = sample.box
    lx, ly, lz = box.size_x, box.size_y, box.size_z
    legend_handles: list[matplotlib.artist.Artist] = []

    # -----------------------------------------------------------------------
    # 1. Material Box Bounding Wireframe
    # -----------------------------------------------------------------------
    if show_box:
        # 12 edges of rectangular parallelepiped
        box_edges = [
            # Bottom face (z = 0)
            ([0, lx], [0, 0], [0, 0]),
            ([lx, lx], [0, ly], [0, 0]),
            ([lx, 0], [ly, ly], [0, 0]),
            ([0, 0], [ly, 0], [0, 0]),
            # Top face (z = lz)
            ([0, lx], [0, 0], [lz, lz]),
            ([lx, lx], [0, ly], [lz, lz]),
            ([lx, 0], [ly, ly], [lz, lz]),
            ([0, 0], [ly, 0], [lz, lz]),
            # Vertical pillars
            ([0, 0], [0, 0], [0, lz]),
            ([lx, lx], [0, 0], [0, lz]),
            ([lx, lx], [ly, ly], [0, lz]),
            ([0, 0], [ly, ly], [0, lz]),
        ]
        edge_segments = [
            [(xs[0], ys[0], zs[0]), (xs[1], ys[1], zs[1])]
            for xs, ys, zs in box_edges
        ]
        box_coll = Line3DCollection(
            edge_segments,
            colors="#64748b",
            linestyles="--",
            linewidths=1.0,
            alpha=0.6,
        )
        ax.add_collection3d(box_coll)

    # -----------------------------------------------------------------------
    # 2. Electrode Planes (LEFT at x=0, RIGHT at x=lx)
    # -----------------------------------------------------------------------
    if show_electrodes:
        left_verts = [
            [
                (0.0, 0.0, 0.0),
                (0.0, ly, 0.0),
                (0.0, ly, lz),
                (0.0, 0.0, lz),
            ]
        ]
        right_verts = [
            [
                (lx, 0.0, 0.0),
                (lx, ly, 0.0),
                (lx, ly, lz),
                (lx, 0.0, lz),
            ]
        ]

        left_poly = Poly3DCollection(
            left_verts,
            facecolors="#3b82f6",
            edgecolors="#1d4ed8",
            linewidths=1.5,
            alpha=0.20,
        )
        right_poly = Poly3DCollection(
            right_verts,
            facecolors="#f97316",
            edgecolors="#c2410c",
            linewidths=1.5,
            alpha=0.20,
        )

        ax.add_collection3d(left_poly)
        ax.add_collection3d(right_poly)

        legend_handles.append(
            Patch(facecolor="#3b82f6", edgecolor="#1d4ed8", alpha=0.5, label="LEFT electrode (x = 0)")
        )
        legend_handles.append(
            Patch(facecolor="#f97316", edgecolor="#c2410c", alpha=0.5, label=f"RIGHT electrode (x = {lx:g})")
        )

    # -----------------------------------------------------------------------
    # 3. CNT Line Segments (Categorized by Connectivity)
    # -----------------------------------------------------------------------
    if show_tubes and sample.tubes:
        path_tube_indices: set[int] = set()
        if highlight_path and sample.percolates and sample.percolating_path:
            path_tube_indices = {
                node for node in sample.percolating_path if isinstance(node, int)
            }

        cluster_tube_indices: set[int] = set()
        if highlight_cluster and sample.percolates and sample.percolating_cluster:
            cluster_tube_indices = sample.percolating_cluster - path_tube_indices

        other_tube_indices = (
            set(range(len(sample.tubes))) - path_tube_indices - cluster_tube_indices
        )

        # A: Non-percolating / isolated tubes
        if other_tube_indices:
            other_segs = [
                (sample.tubes[i].p0, sample.tubes[i].p1)
                for i in sorted(other_tube_indices)
            ]
            other_coll = Line3DCollection(
                other_segs,
                colors="#94a3b8",
                linewidths=1.1,
                alpha=0.45,
            )
            ax.add_collection3d(other_coll)
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color="#94a3b8",
                    lw=1.5,
                    alpha=0.8,
                    label=f"Isolated CNTs (N = {len(other_tube_indices)})",
                )
            )

        # B: Percolating cluster tubes
        if cluster_tube_indices:
            cluster_segs = [
                (sample.tubes[i].p0, sample.tubes[i].p1)
                for i in sorted(cluster_tube_indices)
            ]
            cluster_coll = Line3DCollection(
                cluster_segs,
                colors="#2563eb",
                linewidths=2.2,
                alpha=0.85,
            )
            ax.add_collection3d(cluster_coll)
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color="#2563eb",
                    lw=2.2,
                    label=f"Percolating Cluster (N = {len(sample.percolating_cluster)})",
                )
            )

        # C: Shortest Conductive Backbone Path
        if path_tube_indices:
            path_segs = [
                (sample.tubes[i].p0, sample.tubes[i].p1)
                for i in sorted(path_tube_indices)
            ]
            path_coll = Line3DCollection(
                path_segs,
                colors="#dc2626",
                linewidths=3.6,
                alpha=1.0,
            )
            ax.add_collection3d(path_coll)
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color="#dc2626",
                    lw=3.2,
                    label=f"Shortest Path ({len(path_tube_indices)} tubes)",
                )
            )

    # -----------------------------------------------------------------------
    # 4. Contact Junctions
    # -----------------------------------------------------------------------
    if show_contacts and sample.contacts:
        cluster_set = sample.percolating_cluster if sample.percolates else set()
        cluster_contact_pts: list[np.ndarray] = []
        other_contact_pts: list[np.ndarray] = []

        for c in sample.contacts:
            p_a, p_b, _ = nanotube_closest_points(
                sample.tubes[c.tube_a], sample.tubes[c.tube_b]
            )
            midpoint = 0.5 * (p_a + p_b)

            if c.tube_a in cluster_set and c.tube_b in cluster_set:
                cluster_contact_pts.append(midpoint)
            else:
                other_contact_pts.append(midpoint)

        # Draw other contacts
        if other_contact_pts:
            pts = np.array(other_contact_pts)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color="#64748b",
                s=16,
                alpha=0.6,
                marker="o",
                depthshade=True,
                label=f"Contacts (N = {len(sample.contacts)})",
            )
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#64748b",
                    markersize=6,
                    label=f"Contacts ({len(sample.contacts)} total)",
                )
            )

        # Draw percolating cluster contacts
        if cluster_contact_pts:
            c_pts = np.array(cluster_contact_pts)
            ax.scatter(
                c_pts[:, 0],
                c_pts[:, 1],
                c_pts[:, 2],
                color="#eab308",
                edgecolors="#ca8a04",
                s=32,
                alpha=0.95,
                marker="o",
                depthshade=False,
            )
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#eab308",
                    markeredgecolor="#ca8a04",
                    markersize=7,
                    label=f"Cluster Contacts ({len(cluster_contact_pts)})",
                )
            )

    # -----------------------------------------------------------------------
    # 5. Axes, Limits, Aspect Ratio & Camera
    # -----------------------------------------------------------------------
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)
    ax.set_zlim(0.0, lz)
    ax.set_box_aspect((lx, ly, lz))

    ax.set_xlabel("X (um) [LEFT -> RIGHT]", fontsize=9, labelpad=8)
    ax.set_ylabel("Y (um)", fontsize=9, labelpad=8)
    ax.set_zlabel("Z (um)", fontsize=9, labelpad=8)

    ax.view_init(elev=view_elevation, azim=view_azimuth)

    # Title
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    else:
        status = "PERCOLATING (Path exists!)" if sample.percolates else "NON-PERCOLATING"
        sample_title = (
            f"3D CNT Network Realization - {status}\n"
            f"phi={sample.concentration:.4f} | N={sample.nanotube_count} | "
            f"Contacts={sample.contact_count} | L={sample.cnt_length} um, D={sample.cnt_diameter} um"
        )
        ax.set_title(sample_title, fontsize=10, fontweight="bold", pad=12)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            framealpha=0.9,
            fontsize=8.5,
        )

    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D visualization of a CNT composite realization."
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.005,
        help="Volumetric CNT fraction phi (default: 0.005)",
    )
    parser.add_argument(
        "--box-x",
        type=float,
        default=10.0,
        help="Box size X (default: 10.0 um)",
    )
    parser.add_argument(
        "--box-y",
        type=float,
        default=5.0,
        help="Box size Y (default: 5.0 um)",
    )
    parser.add_argument(
        "--box-z",
        type=float,
        default=5.0,
        help="Box size Z (default: 5.0 um)",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=1.0,
        help="CNT length L (default: 1.0 um)",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=0.1,
        help="CNT diameter D (default: 0.1 um)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Contact threshold (default: 0.05 um)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results/sample_3d.png",
        help="Output image path (default: results/sample_3d.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI (default: 300)",
    )

    args = parser.parse_args()

    from percolation.domain.cnt import MaterialBox
    from percolation.simulation.experiment import create_sample

    box = MaterialBox(size_x=args.box_x, size_y=args.box_y, size_z=args.box_z)
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("  CNT 3D Visualization Generator")
    print("=" * 65)
    print(f"Generating sample at phi = {args.phi:.4f} (seed {args.seed}) ...")

    sample = create_sample(
        concentration=args.phi,
        box=box,
        cnt_length=args.length,
        cnt_diameter=args.diameter,
        contact_threshold=args.threshold,
        rng=rng,
    )

    print(f"Placed {sample.nanotube_count} CNTs, found {sample.contact_count} contacts.")
    status = "PERCOLATES" if sample.percolates else "DOES NOT PERCOLATE"
    print(f"Outcome: {status}")
    if sample.percolates:
        print(f"Percolating cluster size: {len(sample.percolating_cluster)} tubes")
        if sample.percolating_path:
            print(f"Shortest path length: {len(sample.percolating_path) - 2} tubes")

    print(f"Rendering 3D scene to '{args.output}' (dpi={args.dpi}) ...")
    plot_sample_3d(sample, save_path=args.output, dpi=args.dpi)
    print(f"Successfully saved 3D visualization to '{args.output}'!")
    print("=" * 65)


if __name__ == "__main__":
    main()

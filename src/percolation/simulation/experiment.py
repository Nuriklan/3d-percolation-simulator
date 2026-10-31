"""
Single-sample experiment: runs one complete simulation pipeline.

Pipeline:
    concentration phi
        → N = count_from_concentration(...)
        → generate_nanotubes(N, ...)
        → find_contacts(threshold, method=contact_method)
        → build_graph(...)
        → check_percolation(graph)
        → [find_percolating_cluster & find_percolating_path]
        → SampleState / SimulationResult
"""

from __future__ import annotations

import argparse

import numpy as np

from percolation.domain.cnt import MaterialBox, SampleState, SimulationResult
from percolation.generation.cnt_generator import count_from_concentration, generate_nanotubes
from percolation.graph.contact_graph import build_graph, find_contacts
from percolation.graph.percolation import (
    check_percolation,
    find_percolating_cluster,
    find_percolating_path,
)


def create_sample(
    concentration: float,
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    contact_threshold: float,
    rng: np.random.Generator,
    electrode_threshold: float | None = None,
    contact_method: str = "auto",
) -> SampleState:
    """Generate a single sample and return its complete geometrical and graph state.

    Parameters
    ----------
    concentration       : volumetric CNT fraction phi
    box                 : simulation domain
    cnt_length          : CNT length L, um
    cnt_diameter        : CNT diameter D, um
    contact_threshold   : maximum contact distance, um
    rng                 : reproducible random Generator
    electrode_threshold : maximum distance to electrode to form contact
                          (defaults to contact_threshold if None)
    contact_method      : contact search algorithm ('auto', 'grid', 'hash', 'kdtree', 'brute_force')

    Returns
    -------
    SampleState
        Contains tubes, contacts, adjacency graph, percolation status,
        percolating cluster set, and shortest percolating path.
    """
    if electrode_threshold is None:
        electrode_threshold = contact_threshold

    n = count_from_concentration(concentration, box, cnt_length, cnt_diameter)
    tubes = generate_nanotubes(n, box, cnt_length, cnt_diameter, rng)
    contacts = find_contacts(tubes, contact_threshold, box=box, method=contact_method)
    graph = build_graph(tubes, contacts, box, electrode_threshold=electrode_threshold)
    percolates = check_percolation(graph)
    cluster = find_percolating_cluster(graph)
    path = find_percolating_path(graph)

    return SampleState(
        box=box,
        tubes=tubes,
        contacts=contacts,
        graph=graph,
        percolates=percolates,
        percolating_cluster=cluster,
        percolating_path=path,
        concentration=concentration,
        cnt_length=cnt_length,
        cnt_diameter=cnt_diameter,
        contact_threshold=contact_threshold,
    )


def run_sample(
    concentration: float,
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    contact_threshold: float,
    rng: np.random.Generator,
    electrode_threshold: float | None = None,
    contact_method: str = "auto",
) -> SimulationResult:
    """Run a single Monte Carlo sample (fast evaluation path).

    Parameters
    ----------
    concentration       : volumetric CNT fraction phi
    box                 : simulation domain
    cnt_length          : CNT length L, um
    cnt_diameter        : CNT diameter D, um
    contact_threshold   : maximum contact distance, um
    rng                 : reproducible random Generator
    electrode_threshold : maximum distance to electrode to form contact
                          (defaults to contact_threshold if None)
    contact_method      : contact search algorithm ('auto', 'grid', 'hash', 'kdtree', 'brute_force')

    Returns
    -------
    SimulationResult
    """
    if electrode_threshold is None:
        electrode_threshold = contact_threshold

    n = count_from_concentration(concentration, box, cnt_length, cnt_diameter)
    tubes = generate_nanotubes(n, box, cnt_length, cnt_diameter, rng)
    contacts = find_contacts(tubes, contact_threshold, box=box, method=contact_method)
    graph = build_graph(tubes, contacts, box, electrode_threshold=electrode_threshold)
    percolates = check_percolation(graph)

    return SimulationResult(
        concentration=concentration,
        nanotube_count=n,
        percolates=percolates,
        contact_count=len(contacts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single-sample CNT percolation realization with optional 3D visualization."
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
        help="Box length in X (default: 10.0 um)",
    )
    parser.add_argument(
        "--box-y",
        type=float,
        default=5.0,
        help="Box length in Y (default: 5.0 um)",
    )
    parser.add_argument(
        "--box-z",
        type=float,
        default=5.0,
        help="Box length in Z (default: 5.0 um)",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=1.0,
        help="Nanotube length L (default: 1.0 um)",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=0.1,
        help="Nanotube diameter D (default: 0.1 um)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Contact distance threshold (default: 0.05 um)",
    )
    parser.add_argument(
        "--contact-method",
        type=str,
        default="auto",
        choices=["auto", "grid", "hash", "kdtree", "brute_force"],
        help="Contact detection algorithm (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-v", "--visualize",
        type=str,
        default=None,
        help="Path to save 3D visualization image (e.g., results/sample_3d.png)",
    )

    args = parser.parse_args()

    box = MaterialBox(size_x=args.box_x, size_y=args.box_y, size_z=args.box_z)
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("  CNT Percolation Simulator - Single Sample Realization")
    print("=" * 65)
    print(f"Domain Box         : {box.size_x} x {box.size_y} x {box.size_z} um")
    ar = args.length / args.diameter if args.diameter > 0 else 0.0
    print(f"CNT Geometry       : L={args.length} um, D={args.diameter} um (Aspect Ratio={ar:.1f})")
    print(f"Contact Threshold  : {args.threshold} um")
    print(f"Contact Method     : {args.contact_method}")
    print(f"Concentration (phi): {args.phi:.4f}")
    print(f"Random Seed        : {args.seed}")
    print("-" * 65)

    sample = create_sample(
        concentration=args.phi,
        box=box,
        cnt_length=args.length,
        cnt_diameter=args.diameter,
        contact_threshold=args.threshold,
        rng=rng,
        contact_method=args.contact_method,
    )

    print(f"Nanotubes Placed   : {sample.nanotube_count}")
    print(f"Contacts Detected  : {sample.contact_count}")
    print(f"Percolation Result : {'PERCOLATES (Path LEFT -> RIGHT exists!)' if sample.percolates else 'NO PERCOLATION'}")
    if sample.percolates:
        pct = len(sample.percolating_cluster) / sample.nanotube_count * 100
        print(f"Percolating Cluster: {len(sample.percolating_cluster)} tubes ({pct:.1f}%)")
        if sample.percolating_path:
            cnt_hops = len(sample.percolating_path) - 2
            print(f"Shortest Path Hops : {cnt_hops} tubes")
    print("=" * 65)

    if args.visualize:
        from percolation.visualization.renderer import plot_sample_3d
        print(f"Generating 3D visualization -> {args.visualize} ...")
        plot_sample_3d(sample, save_path=args.visualize)
        print("Done!")


if __name__ == "__main__":
    main()

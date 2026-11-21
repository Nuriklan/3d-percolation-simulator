"""
Single-sample experiment: runs one complete simulation pipeline.

Pipeline:
    concentration phi
        → N = count_from_concentration(...)
        → generate_nanotubes(N, ...)
        → find_contacts(threshold, method=contact_method, tunneling=tunneling)
        → build_graph(...) [+ build_weighted_graph(...) when tunneling enabled]
        → check_percolation(graph)
        → [find_percolating_cluster, find_percolating_path,
           find_min_resistance_path, find_bottleneck_conductance]
        → SampleState / SimulationResult
"""

from __future__ import annotations

import argparse

import numpy as np

from percolation.domain.cnt import (
    MaterialBox,
    OrientationConfig,
    SampleState,
    SimulationResult,
)
from percolation.domain.tunneling import (
    TunnelingConfig,
    summarize_contact_conductance,
)
from percolation.generation.cnt_generator import count_from_concentration, generate_nanotubes
from percolation.graph.contact_graph import (
    build_graph,
    build_weighted_graph,
    find_contacts,
)
from percolation.graph.percolation import (
    check_percolation,
    find_bottleneck_conductance,
    find_min_resistance_path,
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
    orientation: OrientationConfig | None = None,
    tunneling: TunnelingConfig | None = None,
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
    orientation         : optional OrientationConfig (defaults to isotropic)
    tunneling           : optional TunnelingConfig (Phase 8). When enabled, populates
                          contact conductances/resistances, weighted_graph,
                          min_resistance_path, and bottleneck_conductance.

    Returns
    -------
    SampleState
    """
    if electrode_threshold is None:
        electrode_threshold = contact_threshold
    orient_cfg = orientation if orientation is not None else OrientationConfig.isotropic()
    use_tunneling = tunneling is not None and tunneling.enabled

    n = count_from_concentration(concentration, box, cnt_length, cnt_diameter)
    tubes = generate_nanotubes(n, box, cnt_length, cnt_diameter, rng, orientation=orient_cfg)
    contacts = find_contacts(
        tubes,
        contact_threshold,
        box=box,
        method=contact_method,
        tunneling=tunneling if use_tunneling else None,
    )

    eff_elec_threshold = (
        tunneling.effective_cutoff(electrode_threshold)
        if use_tunneling and tunneling is not None
        else electrode_threshold
    )
    graph = build_graph(tubes, contacts, box, electrode_threshold=eff_elec_threshold)
    percolates = check_percolation(graph)
    cluster = find_percolating_cluster(graph)
    path = find_percolating_path(graph)

    weighted_graph = None
    min_r_path = None
    min_path_r = None
    bottleneck_g = None

    if use_tunneling and tunneling is not None:
        weighted_graph = build_weighted_graph(
            tubes,
            contacts,
            box,
            electrode_threshold=electrode_threshold,
            tunneling=tunneling,
        )
        if percolates:
            min_res = find_min_resistance_path(weighted_graph)
            if min_res is not None:
                min_r_path, min_path_r = min_res
            bn_res = find_bottleneck_conductance(weighted_graph)
            if bn_res is not None:
                bottleneck_g, _ = bn_res

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
        orientation=orient_cfg,
        tunneling=tunneling,
        weighted_graph=weighted_graph,
        min_resistance_path=min_r_path,
        min_path_resistance=min_path_r,
        bottleneck_conductance=bottleneck_g,
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
    orientation: OrientationConfig | None = None,
    tunneling: TunnelingConfig | None = None,
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
    orientation         : optional OrientationConfig (defaults to isotropic)
    tunneling           : optional TunnelingConfig (Phase 8)

    Returns
    -------
    SimulationResult
    """
    if electrode_threshold is None:
        electrode_threshold = contact_threshold
    use_tunneling = tunneling is not None and tunneling.enabled

    n = count_from_concentration(concentration, box, cnt_length, cnt_diameter)
    tubes = generate_nanotubes(n, box, cnt_length, cnt_diameter, rng, orientation=orientation)
    contacts = find_contacts(
        tubes,
        contact_threshold,
        box=box,
        method=contact_method,
        tunneling=tunneling if use_tunneling else None,
    )

    eff_elec_threshold = (
        tunneling.effective_cutoff(electrode_threshold)
        if use_tunneling and tunneling is not None
        else electrode_threshold
    )
    graph = build_graph(tubes, contacts, box, electrode_threshold=eff_elec_threshold)
    percolates = check_percolation(graph)

    avg_g: float | None = None
    geom_g: float | None = None
    min_path_r: float | None = None
    bottleneck_g: float | None = None

    if use_tunneling and tunneling is not None:
        if contacts:
            stats = summarize_contact_conductance(contacts)
            avg_g = stats.mean_conductance
            geom_g = stats.geom_mean_conductance
        if percolates:
            wgraph = build_weighted_graph(
                tubes,
                contacts,
                box,
                electrode_threshold=electrode_threshold,
                tunneling=tunneling,
            )
            min_res = find_min_resistance_path(wgraph)
            if min_res is not None:
                _, min_path_r = min_res
            bn_res = find_bottleneck_conductance(wgraph)
            if bn_res is not None:
                bottleneck_g, _ = bn_res

    return SimulationResult(
        concentration=concentration,
        nanotube_count=n,
        percolates=percolates,
        contact_count=len(contacts),
        avg_contact_conductance=avg_g,
        geom_mean_conductance=geom_g,
        min_path_resistance=min_path_r,
        bottleneck_conductance=bottleneck_g,
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
        "--orientation-mode",
        type=str,
        default="isotropic",
        choices=["isotropic", "aligned", "partially_aligned", "fisher"],
        help="CNT orientation distribution mode (default: isotropic)",
    )
    parser.add_argument(
        "--orientation-axis",
        type=str,
        default="x",
        help="Preferred alignment axis: 'x', 'y', or 'z' (default: x)",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=30.0,
        help="Cone half-angle in degrees for partially_aligned mode (default: 30.0)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=5.0,
        help="Concentration parameter kappa for fisher mode (default: 5.0)",
    )
    parser.add_argument(
        "--tunneling",
        action="store_true",
        help="Enable exponential tunneling conductance model (Phase 8)",
    )
    parser.add_argument(
        "--g0",
        type=float,
        default=1.0,
        help="Base direct-contact conductance G0 (default: 1.0 S)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Tunneling decay constant alpha in 1/um (default: 50.0)",
    )
    parser.add_argument(
        "--d-min",
        type=float,
        default=0.0,
        help="Direct-contact clamp distance d_min in um (default: 0.0)",
    )
    parser.add_argument(
        "--min-conductance",
        type=float,
        default=None,
        help="Optional minimum contact conductance cutoff G_min",
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
    orientation = OrientationConfig(
        mode=args.orientation_mode,
        axis=args.orientation_axis,
        max_angle_deg=args.max_angle,
        kappa=args.kappa,
    )
    tunneling = (
        TunnelingConfig(
            enabled=True,
            g0=args.g0,
            alpha=args.alpha,
            d_min=args.d_min,
            min_conductance=args.min_conductance,
        )
        if args.tunneling
        else None
    )

    print("=" * 65)
    print("  CNT Percolation Simulator - Single Sample Realization")
    print("=" * 65)
    print(f"Domain Box         : {box.size_x} x {box.size_y} x {box.size_z} um")
    ar = args.length / args.diameter if args.diameter > 0 else 0.0
    print(f"CNT Geometry       : L={args.length} um, D={args.diameter} um (Aspect Ratio={ar:.1f})")
    print(f"Orientation        : {orientation.label} (S_x={orientation.theoretical_order_parameter('x'):.3f})")
    print(f"Contact Threshold  : {args.threshold} um")
    print(f"Contact Method     : {args.contact_method}")
    if tunneling is not None:
        print(f"Tunneling Model    : {tunneling.label}")
        print(f"Effective Cutoff   : {tunneling.effective_cutoff(args.threshold):.5f} um")
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
        orientation=orientation,
        tunneling=tunneling,
    )

    print(f"Nanotubes Placed   : {sample.nanotube_count}")
    print(f"Contacts Detected  : {sample.contact_count}")
    if tunneling is not None and sample.contacts:
        stats = sample.contact_conductance_stats()
        print(f"Mean Conductance   : {stats.mean_conductance:.4e} S")
        print(f"Geom Mean G        : {stats.geom_mean_conductance:.4e} S")
        print(f"Median Resistance  : {stats.median_resistance:.4e} Ohm")
    print(f"Percolation Result : {'PERCOLATES (Path LEFT -> RIGHT exists!)' if sample.percolates else 'NO PERCOLATION'}")
    if sample.percolates:
        pct = len(sample.percolating_cluster) / sample.nanotube_count * 100
        print(f"Percolating Cluster: {len(sample.percolating_cluster)} tubes ({pct:.1f}%)")
        if sample.percolating_path:
            cnt_hops = len(sample.percolating_path) - 2
            print(f"Shortest Path Hops : {cnt_hops} tubes")
        if sample.min_resistance_path is not None and sample.min_path_resistance is not None:
            r_hops = len(sample.min_resistance_path) - 2
            print(f"Min-R Path         : {r_hops} tubes (R_series = {sample.min_path_resistance:.4e} Ohm)")
        if sample.bottleneck_conductance is not None:
            print(f"Bottleneck G_c     : {sample.bottleneck_conductance:.4e} S")
    print("=" * 65)

    if args.visualize:
        from percolation.visualization.renderer import plot_sample_3d
        print(f"Generating 3D visualization -> {args.visualize} ...")
        plot_sample_3d(
            sample,
            save_path=args.visualize,
            color_contacts_by_conductance=bool(args.tunneling),
            highlight_min_resistance_path=bool(args.tunneling),
        )
        print("Done!")


if __name__ == "__main__":
    main()

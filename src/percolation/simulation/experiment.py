"""
Single-sample experiment: runs one complete simulation pipeline.

Pipeline:
    concentration φ
        → N = count_from_concentration(...)
        → generate_nanotubes(N, ...)
        → find_contacts(threshold)
        → build_graph(...)
        → check_percolation(graph)
        → SimulationResult
"""

from __future__ import annotations

import numpy as np

from percolation.domain.cnt import MaterialBox, SimulationResult
from percolation.generation.cnt_generator import count_from_concentration, generate_nanotubes
from percolation.graph.contact_graph import build_graph, find_contacts
from percolation.graph.percolation import check_percolation


def run_sample(
    concentration: float,
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    contact_threshold: float,
    rng: np.random.Generator,
    electrode_threshold: float | None = None,
) -> SimulationResult:
    """Run a single Monte Carlo sample.

    Parameters
    ----------
    concentration       : volumetric CNT fraction φ
    box                 : simulation domain
    cnt_length          : CNT length L, μm
    cnt_diameter        : CNT diameter D, μm
    contact_threshold   : maximum contact distance, μm
    rng                 : reproducible random Generator
    electrode_threshold : maximum distance to electrode to form contact
                          (defaults to contact_threshold if None)

    Returns
    -------
    SimulationResult
    """
    if electrode_threshold is None:
        electrode_threshold = contact_threshold

    n = count_from_concentration(concentration, box, cnt_length, cnt_diameter)
    tubes = generate_nanotubes(n, box, cnt_length, cnt_diameter, rng)
    contacts = find_contacts(tubes, contact_threshold)
    graph = build_graph(tubes, contacts, box, electrode_threshold=electrode_threshold)
    percolates = check_percolation(graph)

    return SimulationResult(
        concentration=concentration,
        nanotube_count=n,
        percolates=percolates,
        contact_count=len(contacts),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  CNT Percolation Simulator - Phase 1 Single-Sample Demo")
    print("=" * 60)

    demo_box = MaterialBox(size_x=5.0, size_y=2.5, size_z=2.5)
    demo_phi = 0.04
    demo_L = 1.0
    demo_D = 0.1
    demo_threshold = 0.08
    demo_seed = 42

    print(f"Domain Box         : {demo_box.size_x} x {demo_box.size_y} x {demo_box.size_z} um")
    print(f"CNT Geometry       : Length={demo_L} um, Diameter={demo_D} um (AR={demo_L/demo_D:.0f})")
    print(f"Contact Threshold  : {demo_threshold} um")
    print(f"Concentration (phi): {demo_phi:.4f}")
    print(f"Random Seed        : {demo_seed}")
    print("-" * 60)

    demo_rng = np.random.default_rng(demo_seed)
    res = run_sample(
        concentration=demo_phi,
        box=demo_box,
        cnt_length=demo_L,
        cnt_diameter=demo_D,
        contact_threshold=demo_threshold,
        rng=demo_rng,
    )

    print(f"Nanotubes Placed   : {res.nanotube_count}")
    print(f"Contacts Detected  : {res.contact_count}")
    print(f"Percolation Result : {'PERCOLATES (Path LEFT -> RIGHT exists!)' if res.percolates else 'NO PERCOLATION'}")
    print("=" * 60)



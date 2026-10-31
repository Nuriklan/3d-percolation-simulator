"""
Contact graph: build and query the CNT connectivity network.

Graph structure
---------------
Nodes:
    0 … N-1  — indices of Nanotubes
    'LEFT'   — virtual left electrode  (x = 0)
    'RIGHT'  — virtual right electrode (x = box.size_x)

Edges:
    CNT i — CNT j   if minimum segment distance ≤ contact_threshold
    LEFT  — CNT i   if CNT i touches the left electrode plane
    RIGHT — CNT i   if CNT i touches the right electrode plane

Electrode contact criterion:
    touches LEFT  if min(p0.x, p1.x) ≤ electrode_threshold
    touches RIGHT if max(p0.x, p1.x) ≥ box.size_x - electrode_threshold
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from percolation.domain.cnt import Contact, MaterialBox, Nanotube
from percolation.geometry.spatial import (
    KDTreeIndex,
    SpatialHashGrid,
    UniformSpatialGrid,
    find_contacts_brute_force,
)

# Special electrode node labels
LEFT = "LEFT"
RIGHT = "RIGHT"
NodeId = int | str


def find_contacts(
    tubes: Sequence[Nanotube],
    threshold: float,
    box: MaterialBox | None = None,
    method: str = "auto",
    cell_size: float | None = None,
) -> list[Contact]:
    """Find all contact pairs between nanotubes within threshold distance.

    Parameters
    ----------
    tubes     : Sequence of Nanotubes
    threshold : Maximum segment-to-segment distance to count as contact, μm
    box       : Simulation domain box (optional, used for grid bounds)
    method    : Acceleration method:
                - 'auto'        : brute_force for N < 30, uniform grid for N >= 30
                - 'grid'        : UniformSpatialGrid (Phase 5 spatial partitioning)
                - 'hash'        : SpatialHashGrid (sparse 3D hash table)
                - 'kdtree'      : KDTreeIndex (SciPy cKDTree on bounding spheres)
                - 'brute_force' : pairwise O(N^2) comparison
    cell_size : Custom cell size for grid and hash methods (optional)

    Returns
    -------
    list[Contact]
        All pairs (i, j) with i < j and distance ≤ threshold, sorted by (tube_a, tube_b).
    """
    n = len(tubes)
    if n < 2:
        return []

    method = method.lower()
    if method == "auto":
        # For very small systems (N < 30), overhead of grid creation is not worth it
        if n < 30:
            method = "brute_force"
        else:
            method = "grid"

    if method == "brute_force":
        contacts = find_contacts_brute_force(tubes, threshold)
    elif method == "grid":
        grid = UniformSpatialGrid(tubes, threshold, box=box, cell_size=cell_size)
        contacts = grid.find_contacts()
    elif method == "hash":
        s_hash = SpatialHashGrid(tubes, threshold, cell_size=cell_size)
        contacts = s_hash.find_contacts()
    elif method == "kdtree":
        tree_idx = KDTreeIndex(tubes, threshold)
        contacts = tree_idx.find_contacts()
    else:
        raise ValueError(
            f"Unknown contact detection method: '{method}'. "
            f"Supported methods: 'auto', 'grid', 'hash', 'kdtree', 'brute_force'."
        )

    return sorted(contacts, key=lambda c: (c.tube_a, c.tube_b))


def build_graph(
    tubes: Sequence[Nanotube],
    contacts: Sequence[Contact],
    box: MaterialBox,
    electrode_threshold: float = 0.0,
) -> dict[NodeId, list[NodeId]]:
    """Build adjacency list including LEFT and RIGHT virtual electrode nodes.

    Parameters
    ----------
    tubes               : list of Nanotubes (indexed 0…N-1)
    contacts            : CNT–CNT contact pairs
    box                 : simulation domain (needed for electrode plane positions)
    electrode_threshold : maximum distance from tube to electrode plane (at x=0
                          and x=size_x) to form an electrical contact.

    Returns
    -------
    dict mapping each node → list of neighbours
    """
    graph: dict[NodeId, list[NodeId]] = defaultdict(list)

    # CNT–CNT edges
    for c in contacts:
        graph[c.tube_a].append(c.tube_b)
        graph[c.tube_b].append(c.tube_a)

    # Electrode edges
    for idx, tube in enumerate(tubes):
        x_min = min(tube.p0[0], tube.p1[0])
        x_max = max(tube.p0[0], tube.p1[0])

        if x_min <= electrode_threshold:
            graph[LEFT].append(idx)
            graph[idx].append(LEFT)

        if x_max >= box.size_x - electrode_threshold:
            graph[RIGHT].append(idx)
            graph[idx].append(RIGHT)

    # Ensure LEFT and RIGHT always exist as keys
    if LEFT not in graph:
        graph[LEFT] = []
    if RIGHT not in graph:
        graph[RIGHT] = []

    return dict(graph)

"""
Contact graph: build and query the CNT connectivity network.

Graph structure
---------------
Nodes:
    0 … N-1  — indices of Nanotubes
    'LEFT'   — virtual left electrode  (x = 0)
    'RIGHT'  — virtual right electrode (x = box.size_x)

Edges:
    CNT i — CNT j   if minimum segment distance ≤ effective contact threshold
    LEFT  — CNT i   if CNT i touches/tunnels to the left electrode plane
    RIGHT — CNT i   if CNT i touches/tunnels to the right electrode plane

Phase 8 (Tunneling Conductance):
    When a ``TunnelingConfig`` (with ``enabled=True``) is provided:
    - Each ``Contact`` and ``ElectrodeContact`` is enriched with exponential
      tunneling conductance ``G_ij = G_0 * exp(-alpha * max(0, d - d_min))``
      and resistance ``R_ij = 1 / G_ij``.
    - ``build_weighted_graph`` constructs a symmetric conductance adjacency
      mapping ``node -> {neighbour: conductance}``, ready for weighted path
      analysis (Phase 8) and Kirchhoff resistor network solvers (Phase 9).

Electrode contact criterion:
    distance to LEFT  = max(0, min(p0.x, p1.x))
    distance to RIGHT = max(0, box.size_x - max(p0.x, p1.x))
    touches LEFT  if d_LEFT  ≤ effective_electrode_threshold
    touches RIGHT if d_RIGHT ≤ effective_electrode_threshold
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from percolation.domain.cnt import Contact, MaterialBox, Nanotube
from percolation.domain.tunneling import (
    ElectrodeContact,
    TunnelingConfig,
    apply_tunneling_to_contacts,
)
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
WeightedGraph = dict[NodeId, dict[NodeId, float]]


def find_contacts(
    tubes: Sequence[Nanotube],
    threshold: float,
    box: MaterialBox | None = None,
    method: str = "auto",
    cell_size: float | None = None,
    tunneling: TunnelingConfig | None = None,
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
    tunneling : Optional TunnelingConfig (Phase 8). When enabled, populates
                ``conductance`` and ``resistance`` on each Contact and applies
                ``effective_cutoff`` (cutoff_distance / min_conductance).

    Returns
    -------
    list[Contact]
        All pairs (i, j) with i < j and distance ≤ effective threshold, sorted by (tube_a, tube_b).
    """
    n = len(tubes)
    if n < 2:
        return []

    use_tunneling = tunneling is not None and tunneling.enabled
    search_threshold = (
        tunneling.effective_cutoff(threshold) if use_tunneling and tunneling is not None else float(threshold)
    )
    if search_threshold < 0.0:
        return []

    method = method.lower()
    if method == "auto":
        # For very small systems (N < 30), overhead of grid creation is not worth it
        if n < 30:
            method = "brute_force"
        else:
            method = "grid"

    if method == "brute_force":
        contacts = find_contacts_brute_force(tubes, search_threshold)
    elif method == "grid":
        grid = UniformSpatialGrid(tubes, search_threshold, box=box, cell_size=cell_size)
        contacts = grid.find_contacts()
    elif method == "hash":
        s_hash = SpatialHashGrid(tubes, search_threshold, cell_size=cell_size)
        contacts = s_hash.find_contacts()
    elif method == "kdtree":
        tree_idx = KDTreeIndex(tubes, search_threshold)
        contacts = tree_idx.find_contacts()
    else:
        raise ValueError(
            f"Unknown contact detection method: '{method}'. "
            f"Supported methods: 'auto', 'grid', 'hash', 'kdtree', 'brute_force'."
        )

    contacts = sorted(contacts, key=lambda c: (c.tube_a, c.tube_b))
    if use_tunneling and tunneling is not None:
        contacts = apply_tunneling_to_contacts(
            contacts,
            tunneling,
            contact_threshold=threshold,
        )
    return contacts


def find_electrode_contacts(
    tubes: Sequence[Nanotube],
    box: MaterialBox,
    electrode_threshold: float = 0.0,
    tunneling: TunnelingConfig | None = None,
) -> tuple[list[ElectrodeContact], list[ElectrodeContact]]:
    """Identify tubes in electrical contact with LEFT (x=0) and RIGHT (x=size_x) electrodes.

    Parameters
    ----------
    tubes               : Sequence of Nanotubes (indexed 0..N-1)
    box                 : simulation domain
    electrode_threshold : maximum distance to electrode plane for contact
    tunneling           : optional TunnelingConfig to compute electrode junction
                          conductance and resistance

    Returns
    -------
    tuple[list[ElectrodeContact], list[ElectrodeContact]]
        ``(left_contacts, right_contacts)``
    """
    use_tunneling = tunneling is not None and tunneling.enabled
    eff_threshold = (
        tunneling.effective_cutoff(electrode_threshold)
        if use_tunneling and tunneling is not None
        else float(electrode_threshold)
    )
    if eff_threshold < 0.0:
        return [], []

    left_contacts: list[ElectrodeContact] = []
    right_contacts: list[ElectrodeContact] = []

    for idx, tube in enumerate(tubes):
        x_min = min(float(tube.p0[0]), float(tube.p1[0]))
        x_max = max(float(tube.p0[0]), float(tube.p1[0]))

        d_left = max(0.0, x_min)
        if d_left <= eff_threshold + 1e-15:
            if use_tunneling and tunneling is not None:
                g_l = tunneling.conductance(d_left)
                r_l = tunneling.resistance(d_left)
                if tunneling.min_conductance is None or g_l >= tunneling.min_conductance * (1.0 - 1e-12):
                    left_contacts.append(
                        ElectrodeContact(
                            electrode=LEFT,
                            tube_idx=idx,
                            distance=d_left,
                            conductance=g_l,
                            resistance=r_l,
                        )
                    )
            else:
                left_contacts.append(
                    ElectrodeContact(electrode=LEFT, tube_idx=idx, distance=d_left)
                )

        d_right = max(0.0, float(box.size_x) - x_max)
        if d_right <= eff_threshold + 1e-15:
            if use_tunneling and tunneling is not None:
                g_r = tunneling.conductance(d_right)
                r_r = tunneling.resistance(d_right)
                if tunneling.min_conductance is None or g_r >= tunneling.min_conductance * (1.0 - 1e-12):
                    right_contacts.append(
                        ElectrodeContact(
                            electrode=RIGHT,
                            tube_idx=idx,
                            distance=d_right,
                            conductance=g_r,
                            resistance=r_r,
                        )
                    )
            else:
                right_contacts.append(
                    ElectrodeContact(electrode=RIGHT, tube_idx=idx, distance=d_right)
                )

    return left_contacts, right_contacts


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
    if electrode_threshold >= 0.0:
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


def build_weighted_graph(
    tubes: Sequence[Nanotube],
    contacts: Sequence[Contact],
    box: MaterialBox,
    electrode_threshold: float = 0.0,
    tunneling: TunnelingConfig | None = None,
) -> WeightedGraph:
    """Build a conductance-weighted adjacency dictionary for the CNT network.

    Each edge ``(u, v)`` stores its electrical conductance ``G_uv`` (in S or
    normalized units). Both CNT–CNT contacts and electrode contacts are weighted:
    - If a ``Contact`` already carries ``conductance``, that value is used.
    - Otherwise, if ``tunneling`` is enabled, conductance is computed via
      ``tunneling.conductance(distance)``.
    - If neither is provided (binary fallback), unit conductance ``1.0`` is assigned.

    Parameters
    ----------
    tubes               : list of Nanotubes (indexed 0..N-1)
    contacts            : CNT-CNT Contact list
    box                 : simulation domain
    electrode_threshold : maximum distance to LEFT/RIGHT electrodes
    tunneling           : optional TunnelingConfig

    Returns
    -------
    dict[NodeId, dict[NodeId, float]]
        Symmetric conductance adjacency mapping ``node -> {neighbour: conductance}``.
    """
    use_tunneling = tunneling is not None and tunneling.enabled
    wgraph: dict[NodeId, dict[NodeId, float]] = defaultdict(dict)

    # CNT-CNT weighted edges
    for c in contacts:
        if c.conductance is not None:
            g = float(c.conductance)
        elif use_tunneling and tunneling is not None:
            g = tunneling.conductance(c.distance)
        else:
            g = 1.0

        if (
            use_tunneling
            and tunneling is not None
            and tunneling.min_conductance is not None
            and g < tunneling.min_conductance * (1.0 - 1e-12)
        ):
            continue

        if g > 0.0:
            wgraph[c.tube_a][c.tube_b] = g
            wgraph[c.tube_b][c.tube_a] = g

    # Electrode weighted edges
    left_contacts, right_contacts = find_electrode_contacts(
        tubes,
        box,
        electrode_threshold=electrode_threshold,
        tunneling=tunneling if use_tunneling else None,
    )

    for ec in left_contacts:
        g = float(ec.conductance) if ec.conductance is not None else 1.0
        if g > 0.0:
            wgraph[LEFT][ec.tube_idx] = g
            wgraph[ec.tube_idx][LEFT] = g

    for ec in right_contacts:
        g = float(ec.conductance) if ec.conductance is not None else 1.0
        if g > 0.0:
            wgraph[RIGHT][ec.tube_idx] = g
            wgraph[ec.tube_idx][RIGHT] = g

    if LEFT not in wgraph:
        wgraph[LEFT] = {}
    if RIGHT not in wgraph:
        wgraph[RIGHT] = {}

    return dict(wgraph)

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

Electrode contact criterion (Phase 1):
    A CNT touches an electrode if either endpoint is within
    `electrode_reach` of the electrode plane.
    For simplicity, electrode_reach = cnt_length / 2
    (i.e., the tube centre is within L/2 of the plane).
    Equivalently: min(p0.x, p1.x) ≤ 0  or max(p0.x, p1.x) ≥ box.size_x
    But since endpoints are clipped to [0, size_x], we check
    whether p0.x == 0 or p1.x == 0 etc.  In practice we just check
    whether x_min(tube) ≤ 0 and x_max(tube) ≥ size_x.

    The robust condition used here:
        touches LEFT  if min(p0.x, p1.x) ≤ 0.0
        touches RIGHT if max(p0.x, p1.x) ≥ size_x
"""

from __future__ import annotations

from collections import defaultdict
from typing import Union

from percolation.domain.cnt import Contact, MaterialBox, Nanotube
from percolation.geometry.distance import nanotube_distance

# Special electrode node labels
LEFT = "LEFT"
RIGHT = "RIGHT"
NodeId = Union[int, str]


def find_contacts(
    tubes: list[Nanotube],
    threshold: float,
) -> list[Contact]:
    """Pairwise contact detection with exact bounding rejection filter.

    Parameters
    ----------
    tubes     : list of Nanotubes
    threshold : maximum segment-to-segment distance to count as contact, μm

    Returns
    -------
    list[Contact]
        All pairs (i, j) with i < j and distance ≤ threshold.
    """
    contacts: list[Contact] = []
    n = len(tubes)
    if n < 2:
        return contacts

    centers = [t.center for t in tubes]
    half_lengths = [0.5 * t.length for t in tubes]

    for i in range(n):
        ci = centers[i]
        hl_i = half_lengths[i]
        tube_i = tubes[i]
        for j in range(i + 1, n):
            reach = hl_i + half_lengths[j] + threshold
            cj = centers[j]
            dx = abs(ci[0] - cj[0])
            if dx > reach:
                continue
            dy = abs(ci[1] - cj[1])
            if dy > reach:
                continue
            dz = abs(ci[2] - cj[2])
            if dz > reach:
                continue

            if (dx * dx + dy * dy + dz * dz) > reach * reach:
                continue

            d = nanotube_distance(tube_i, tubes[j])
            if d <= threshold:
                contacts.append(Contact(tube_a=i, tube_b=j, distance=d))
    return contacts


def build_graph(
    tubes: list[Nanotube],
    contacts: list[Contact],
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


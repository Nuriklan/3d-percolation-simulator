"""
Percolation check: BFS to detect LEFT → RIGHT connectivity, cluster analysis,
and weighted tunneling network path analysis (Phase 8).
"""

from __future__ import annotations

import heapq
from collections import deque
from itertools import pairwise

NodeId = int | str
WeightedGraph = dict[NodeId, dict[NodeId, float]]
LEFT = "LEFT"
RIGHT = "RIGHT"


def check_percolation(graph: dict[NodeId, list[NodeId]]) -> bool:
    """Return True if a path exists from LEFT to RIGHT in *graph*.

    Uses Breadth-First Search (BFS) starting from the LEFT electrode node.

    Parameters
    ----------
    graph : adjacency list (node → list of neighbours)
            Must contain 'LEFT' and 'RIGHT' as special nodes.

    Returns
    -------
    bool
        True  — percolating path found
        False — no path (system does not percolate)
    """
    if LEFT not in graph or not graph[LEFT]:
        return False

    visited: set[NodeId] = {LEFT}
    queue: deque[NodeId] = deque([LEFT])

    while queue:
        node = queue.popleft()

        if node == RIGHT:
            return True

        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return False


def find_percolating_path(graph: dict[NodeId, list[NodeId]]) -> list[NodeId] | None:
    """Return one shortest percolating path LEFT → RIGHT, or None.

    Uses BFS and backtracks through parent pointers.
    Useful for visualisation (Phase 4).

    Returns
    -------
    list[NodeId] | None
        Ordered sequence of nodes from LEFT to RIGHT, or None if no path.
    """
    if LEFT not in graph or not graph[LEFT]:
        return None

    parent: dict[NodeId, NodeId | None] = {LEFT: None}
    queue: deque[NodeId] = deque([LEFT])

    while queue:
        node = queue.popleft()

        if node == RIGHT:
            # Reconstruct path
            path: list[NodeId] = []
            cur: NodeId | None = RIGHT
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        for neighbour in graph.get(node, []):
            if neighbour not in parent:
                parent[neighbour] = node
                queue.append(neighbour)

    return None


def find_percolating_cluster(graph: dict[NodeId, list[NodeId]]) -> set[int]:
    """Return the set of nanotube indices belonging to the percolating network.

    If no continuous path connects LEFT to RIGHT, returns an empty set.
    Otherwise, returns all integer nanotube indices in the connected component
    bridging LEFT and RIGHT.

    Parameters
    ----------
    graph : adjacency list including LEFT and RIGHT nodes

    Returns
    -------
    set[int]
        Indices of all nanotubes in the percolating cluster.
    """
    if not check_percolation(graph):
        return set()

    visited: set[NodeId] = {LEFT}
    queue: deque[NodeId] = deque([LEFT])

    while queue:
        node = queue.popleft()
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return {node for node in visited if isinstance(node, int)}


def find_all_clusters(
    graph: dict[NodeId, list[NodeId]],
    num_tubes: int | None = None,
) -> list[set[int]]:
    """Partition nanotubes into connected clusters based on CNT-CNT contacts.

    Virtual electrode nodes (LEFT and RIGHT) are ignored so that clusters
    represent internal physical tube aggregates.

    Parameters
    ----------
    graph     : adjacency list
    num_tubes : total number of tubes; if None, inferred from graph keys

    Returns
    -------
    list[set[int]]
        List of tube index sets, sorted by cluster size descending.
    """
    if num_tubes is not None:
        all_tubes = set(range(num_tubes))
    else:
        all_tubes = {n for n in graph if isinstance(n, int)}
        for neighbors in graph.values():
            all_tubes.update(n for n in neighbors if isinstance(n, int))

    visited: set[int] = set()
    clusters: list[set[int]] = []

    for tube_id in sorted(all_tubes):
        if tube_id not in visited:
            cluster: set[int] = set()
            queue: deque[int] = deque([tube_id])
            visited.add(tube_id)
            while queue:
                curr = queue.popleft()
                cluster.add(curr)
                for nbr in graph.get(curr, []):
                    if isinstance(nbr, int) and nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            clusters.append(cluster)

    # Sort descending by cluster size
    clusters.sort(key=len, reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Phase 8: Weighted Tunneling Graph Percolation & Path Algorithms
# ---------------------------------------------------------------------------

def _node_sort_key(node: NodeId) -> tuple[int, str | int]:
    """Deterministic tie-breaking key for mixed int/str NodeId heap entries."""
    if isinstance(node, int):
        return (1, node)
    return (0, node)


def path_series_resistance(
    weighted_graph: WeightedGraph,
    path: list[NodeId],
) -> float:
    """Compute total series resistance sum(1 / G_uv) along *path*.

    Raises ValueError if any consecutive edge in *path* is missing or has G <= 0.
    """
    if len(path) < 2:
        return 0.0
    total_r = 0.0
    for u, v in pairwise(path):
        g = weighted_graph.get(u, {}).get(v, 0.0)
        if g <= 0.0:
            raise ValueError(f"Edge ({u!r}, {v!r}) not found or non-conductive in weighted_graph.")
        total_r += 1.0 / g
    return total_r


def find_min_resistance_path(
    weighted_graph: WeightedGraph,
) -> tuple[list[NodeId], float] | None:
    """Find the lowest total series-resistance path LEFT -> RIGHT via Dijkstra.

    Each edge ``(u, v)`` with conductance ``G_uv > 0`` has resistance
    ``R_uv = 1 / G_uv``. Unlike BFS (which minimizes hop count), Dijkstra finds
    the path minimizing ``sum(R_uv)``, which can differ dramatically when contact
    conductances vary exponentially with gap distance.

    Returns
    -------
    tuple[list[NodeId], float] | None
        ``(path, total_resistance)`` from LEFT to RIGHT, or None if disconnected.
    """
    if LEFT not in weighted_graph or not weighted_graph[LEFT]:
        return None

    dist: dict[NodeId, float] = {LEFT: 0.0}
    parent: dict[NodeId, NodeId | None] = {LEFT: None}
    # Heap items: (cumulative_resistance, tie_breaker, node)
    heap: list[tuple[float, tuple[int, str | int], NodeId]] = [
        (0.0, _node_sort_key(LEFT), LEFT)
    ]

    while heap:
        cur_r, _, node = heapq.heappop(heap)
        if cur_r > dist.get(node, float("inf")):
            continue

        if node == RIGHT:
            path: list[NodeId] = []
            curr: NodeId | None = RIGHT
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            return path[::-1], cur_r

        for neighbour, g in weighted_graph.get(node, {}).items():
            if g <= 0.0:
                continue
            step_r = 1.0 / g
            next_r = cur_r + step_r
            if next_r < dist.get(neighbour, float("inf")):
                dist[neighbour] = next_r
                parent[neighbour] = node
                heapq.heappush(heap, (next_r, _node_sort_key(neighbour), neighbour))

    return None


def find_bottleneck_conductance(
    weighted_graph: WeightedGraph,
) -> tuple[float, list[NodeId]] | None:
    """Find the critical-path (widest-path / minimax) conductance LEFT -> RIGHT.

    Computes:
        G_bottleneck = max_{path P: LEFT -> RIGHT}  min_{edge e in P} G_e

    In Ambegaokar-Halperin-Langer (AHL) critical-path theory of disordered
    tunneling networks, ``G_bottleneck`` is the exact critical conductance G_c
    above which the spanning conductive cluster breaks apart.

    Returns
    -------
    tuple[float, list[NodeId]] | None
        ``(bottleneck_conductance, path)`` or None if LEFT and RIGHT are disconnected.
    """
    if LEFT not in weighted_graph or not weighted_graph[LEFT]:
        return None

    best_g: dict[NodeId, float] = {LEFT: float("inf")}
    parent: dict[NodeId, NodeId | None] = {LEFT: None}
    # Max-heap via negated conductance: (-bottleneck_g, tie_breaker, node)
    heap: list[tuple[float, tuple[int, str | int], NodeId]] = [
        (-float("inf"), _node_sort_key(LEFT), LEFT)
    ]

    while heap:
        neg_g, _, node = heapq.heappop(heap)
        cur_g = -neg_g
        if cur_g < best_g.get(node, 0.0):
            continue

        if node == RIGHT:
            path: list[NodeId] = []
            curr: NodeId | None = RIGHT
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            return cur_g, path[::-1]

        for neighbour, edge_g in weighted_graph.get(node, {}).items():
            if edge_g <= 0.0:
                continue
            cand_g = min(cur_g, edge_g)
            if cand_g > best_g.get(neighbour, 0.0):
                best_g[neighbour] = cand_g
                parent[neighbour] = node
                heapq.heappush(heap, (-cand_g, _node_sort_key(neighbour), neighbour))

    return None


def filter_graph_by_conductance(
    weighted_graph: WeightedGraph,
    min_conductance: float,
) -> WeightedGraph:
    """Return a subgraph retaining only edges with conductance >= *min_conductance*."""
    min_g = float(min_conductance)
    filtered: WeightedGraph = {}
    for node, neighbours in weighted_graph.items():
        filtered[node] = {nbr: g for nbr, g in neighbours.items() if g >= min_g}
    if LEFT not in filtered:
        filtered[LEFT] = {}
    if RIGHT not in filtered:
        filtered[RIGHT] = {}
    return filtered


def check_conductance_percolation(
    weighted_graph: WeightedGraph,
    min_conductance: float,
) -> bool:
    """Return True if LEFT connects to RIGHT using only edges with G >= *min_conductance*."""
    min_g = float(min_conductance)
    if LEFT not in weighted_graph or not weighted_graph[LEFT]:
        return False

    visited: set[NodeId] = {LEFT}
    queue: deque[NodeId] = deque([LEFT])

    while queue:
        node = queue.popleft()
        if node == RIGHT:
            return True
        for neighbour, g in weighted_graph.get(node, {}).items():
            if g >= min_g and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return False

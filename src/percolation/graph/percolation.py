"""
Percolation check: BFS to detect LEFT → RIGHT connectivity and cluster analysis.
"""

from __future__ import annotations

from collections import deque

NodeId = int | str
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

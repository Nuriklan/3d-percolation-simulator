"""
Percolation check: BFS to detect LEFT → RIGHT connectivity.
"""

from __future__ import annotations

from collections import deque
from typing import Union

NodeId = Union[int, str]
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

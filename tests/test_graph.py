"""
Unit tests for contact graph construction and BFS percolation check.

Graph tests cover:
- Connected path LEFT–A–B–RIGHT → percolates
- Disconnected LEFT–A and B–RIGHT → does not percolate
- No tubes → does not percolate
- Direct bridge: single tube touching both electrodes

Percolation checker tests:
- Empty graph
- Single node loop (should not percolate without RIGHT)
"""

import numpy as np
import pytest

from percolation.domain.cnt import Contact, MaterialBox, Nanotube
from percolation.graph.contact_graph import LEFT, RIGHT, build_graph, find_contacts
from percolation.graph.percolation import check_percolation, find_percolating_path


BOX = MaterialBox(size_x=10.0, size_y=5.0, size_z=5.0)


def make_tube(x0: float, x1: float, y: float = 2.5, z: float = 2.5) -> Nanotube:
    """Convenience: create a tube along X between x0 and x1."""
    center = np.array([(x0 + x1) / 2, y, z])
    direction = np.array([1.0, 0.0, 0.0])
    length = abs(x1 - x0)
    return Nanotube(center=center, direction=direction, length=length, diameter=0.01)


# ---------------------------------------------------------------------------
# Contact detection
# ---------------------------------------------------------------------------

class TestFindContacts:
    def test_touching_tubes(self):
        """Two tubes whose endpoints meet should have distance 0."""
        t0 = make_tube(0.5, 3.0)
        t1 = make_tube(3.0, 6.0)   # shares endpoint with t0
        contacts = find_contacts([t0, t1], threshold=0.001)
        assert len(contacts) == 1
        assert contacts[0].distance == pytest.approx(0.0, abs=1e-10)

    def test_separated_tubes_no_contact(self):
        """Tubes with gap > threshold should not form contact."""
        t0 = make_tube(0.5, 2.0)
        t1 = make_tube(3.0, 5.0)   # gap = 1.0
        contacts = find_contacts([t0, t1], threshold=0.5)
        assert len(contacts) == 0

    def test_contact_within_threshold(self):
        """Tubes with gap <= threshold should form contact."""
        t0 = make_tube(0.5, 2.0)
        t1 = make_tube(2.05, 4.0)  # gap = 0.05
        contacts = find_contacts([t0, t1], threshold=0.1)
        assert len(contacts) == 1

    def test_three_tubes_two_contacts(self):
        t0 = make_tube(0.5, 3.0)
        t1 = make_tube(3.0, 6.0)
        t2 = make_tube(6.0, 9.0)
        contacts = find_contacts([t0, t1, t2], threshold=0.001)
        assert len(contacts) == 2


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_electrode_connection_left(self):
        """Tube starting at x=0 must connect to LEFT."""
        t = make_tube(0.0, 3.0)
        contacts = []
        graph = build_graph([t], contacts, BOX)
        assert 0 in graph[LEFT]

    def test_electrode_connection_right(self):
        """Tube ending at x=box.size_x must connect to RIGHT."""
        t = make_tube(7.0, 10.0)
        contacts = []
        graph = build_graph([t], contacts, BOX)
        assert 0 in graph[RIGHT]

    def test_tube_spans_full_width(self):
        """Tube touching both electrodes connects both LEFT and RIGHT."""
        t = make_tube(0.0, 10.0)
        graph = build_graph([t], [], BOX)
        assert 0 in graph[LEFT]
        assert 0 in graph[RIGHT]

    def test_internal_tube_no_electrode(self):
        """Tube entirely inside box should not touch electrodes."""
        t = make_tube(2.0, 5.0)
        graph = build_graph([t], [], BOX)
        assert 0 not in graph[LEFT]
        assert 0 not in graph[RIGHT]


# ---------------------------------------------------------------------------
# Percolation check
# ---------------------------------------------------------------------------

class TestCheckPercolation:
    def test_direct_bridge(self):
        """Single tube spanning full box: should percolate."""
        t = make_tube(0.0, 10.0)
        graph = build_graph([t], [], BOX)
        assert check_percolation(graph) is True

    def test_chain_percolates(self):
        """Chain: LEFT–t0–t1–t2–RIGHT."""
        t0 = make_tube(0.0, 3.5)          # touches LEFT (x_min=0)
        t1 = make_tube(3.5, 6.5)          # internal bridge
        t2 = make_tube(6.5, 10.0)         # touches RIGHT (x_max=10)
        contacts = find_contacts([t0, t1, t2], threshold=0.001)
        graph = build_graph([t0, t1, t2], contacts, BOX)
        assert check_percolation(graph) is True

    def test_disconnected_does_not_percolate(self):
        """Two separate groups: LEFT–t0 and t1–RIGHT, no link between them."""
        t0 = make_tube(0.0, 3.0)          # touches LEFT
        t1 = make_tube(7.0, 10.0)         # touches RIGHT
        # gap of 4 between them
        contacts = find_contacts([t0, t1], threshold=0.001)
        graph = build_graph([t0, t1], contacts, BOX)
        assert check_percolation(graph) is False

    def test_empty_system_does_not_percolate(self):
        graph = build_graph([], [], BOX)
        assert check_percolation(graph) is False

    def test_no_right_electrode_connection(self):
        t = make_tube(0.0, 5.0)           # touches LEFT but not RIGHT
        graph = build_graph([t], [], BOX)
        assert check_percolation(graph) is False


# ---------------------------------------------------------------------------
# Path finding
# ---------------------------------------------------------------------------

class TestFindPercolatingPath:
    def test_path_found(self):
        t0 = make_tube(0.0, 3.5)
        t1 = make_tube(3.5, 10.0)
        contacts = find_contacts([t0, t1], threshold=0.001)
        graph = build_graph([t0, t1], contacts, BOX)
        path = find_percolating_path(graph)
        assert path is not None
        assert path[0] == LEFT
        assert path[-1] == RIGHT

    def test_no_path_returns_none(self):
        t0 = make_tube(0.0, 3.0)
        graph = build_graph([t0], [], BOX)
        path = find_percolating_path(graph)
        assert path is None

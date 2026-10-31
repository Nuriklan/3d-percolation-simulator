"""
Unit tests for 3D segment-to-segment distance computation.

Covers:
- Parallel segments (both close and far)
- Perpendicular (crossing) segments
- Coincident / overlapping segments
- Point-to-point degenerate case
- Skew segments known distance
"""

import math

import numpy as np
import pytest

from percolation.geometry.distance import segment_to_segment_distance


def test_same_segment_distance_is_zero():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    assert segment_to_segment_distance(p0, p1, p0.copy(), p1.copy()) == pytest.approx(0.0, abs=1e-12)


def test_parallel_close_segments():
    """Parallel segments separated by distance 1 in Y."""
    p0, p1 = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
    q0, q1 = np.array([0.0, 1.0, 0.0]), np.array([2.0, 1.0, 0.0])
    assert segment_to_segment_distance(p0, p1, q0, q1) == pytest.approx(1.0, rel=1e-9)


def test_parallel_far_segments():
    """Parallel segments far apart."""
    p0, p1 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    q0, q1 = np.array([0.0, 5.0, 0.0]), np.array([1.0, 5.0, 0.0])
    assert segment_to_segment_distance(p0, p1, q0, q1) == pytest.approx(5.0, rel=1e-9)


def test_perpendicular_intersecting_segments():
    """Two segments that actually cross at origin — distance should be 0."""
    p0, p1 = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    q0, q1 = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert segment_to_segment_distance(p0, p1, q0, q1) == pytest.approx(0.0, abs=1e-12)


def test_skew_segments_known_distance():
    """
    P along X-axis, Q along Y-axis displaced by 3 in Z.
    Minimum distance between the two skew lines is 3.
    """
    p0, p1 = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    q0, q1 = np.array([0.0, -1.0, 3.0]), np.array([0.0, 1.0, 3.0])
    assert segment_to_segment_distance(p0, p1, q0, q1) == pytest.approx(3.0, rel=1e-9)


def test_endpoint_to_endpoint_non_overlapping():
    """Segments that don't overlap in projection; closest points are endpoints."""
    # P: [0,0,0] → [1,0,0]
    # Q: [3,0,0] → [4,0,0]   (gap of 2 between endpoints)
    p0, p1 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    q0, q1 = np.array([3.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])
    assert segment_to_segment_distance(p0, p1, q0, q1) == pytest.approx(2.0, rel=1e-9)


def test_point_degenerate_both():
    """Both segments degenerate to points."""
    p = np.array([1.0, 2.0, 3.0])
    q = np.array([4.0, 6.0, 3.0])
    expected = math.sqrt(25)  # sqrt((4-1)^2 + (6-2)^2)
    assert segment_to_segment_distance(p, p.copy(), q, q.copy()) == pytest.approx(expected, rel=1e-9)


def test_point_degenerate_one():
    """One segment is a point."""
    p0, p1 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    q = np.array([0.5, 2.0, 0.0])  # point directly above midpoint
    assert segment_to_segment_distance(p0, p1, q, q.copy()) == pytest.approx(2.0, rel=1e-9)

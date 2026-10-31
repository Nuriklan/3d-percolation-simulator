"""Geometry sub-package for 3D segment distance, closest point calculations, and spatial acceleration."""

from __future__ import annotations

from percolation.geometry.distance import (
    batch_segment_distance,
    nanotube_closest_points,
    nanotube_distance,
    segment_closest_points,
    segment_to_segment_distance,
)
from percolation.geometry.spatial import (
    KDTreeIndex,
    SpatialBenchmarkResult,
    SpatialHashGrid,
    SpatialIndex,
    UniformSpatialGrid,
    compare_contact_methods,
    find_contacts_brute_force,
)

__all__ = [
    "KDTreeIndex",
    "SpatialBenchmarkResult",
    "SpatialHashGrid",
    "SpatialIndex",
    "UniformSpatialGrid",
    "batch_segment_distance",
    "compare_contact_methods",
    "find_contacts_brute_force",
    "nanotube_closest_points",
    "nanotube_distance",
    "segment_closest_points",
    "segment_to_segment_distance",
]

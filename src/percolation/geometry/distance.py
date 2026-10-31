"""
Geometry: minimum distance between two 3D line segments.

Reference:
    Dan Sunday, "Distance between 3D Lines & Segments", 2001.
    http://geomalgorithms.com/a07-_distance.html

Physical context:
    Two CNTs are modelled as line segments A and B.
    The minimum distance between the segments determines whether
    an electrical contact exists (distance ≤ contact_threshold).

    For the Phase-1 binary model the actual tunnel conductance is ignored;
    only the boolean contact condition is used.
"""

from __future__ import annotations

import numpy as np

from percolation.domain.cnt import Nanotube, Vector3D

# Numerical tolerance for near-parallel segments
_SMALL = 1e-10


def segment_closest_points(
    p0: Vector3D, p1: Vector3D, q0: Vector3D, q1: Vector3D
) -> tuple[Vector3D, Vector3D, float]:
    """Compute the closest points between two 3D line segments P and Q.

    Parameters
    ----------
    p0, p1 : endpoints of segment P
    q0, q1 : endpoints of segment Q

    Returns
    -------
    tuple[Vector3D, Vector3D, float]
        (closest_point_on_P, closest_point_on_Q, min_distance)
    """
    d1 = p1 - p0   # direction of P
    d2 = q1 - q0   # direction of Q
    r = p0 - q0

    a = np.dot(d1, d1)  # squared length of P
    e = np.dot(d2, d2)  # squared length of Q
    f = np.dot(d2, r)

    # Degenerate cases: one or both segments are points
    if a < _SMALL and e < _SMALL:
        dist = float(np.linalg.norm(r))
        return p0.copy(), q0.copy(), dist

    if a < _SMALL:
        # P is a point
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = np.dot(d1, r)
        if e < _SMALL:
            # Q is a point
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            # General case
            b = np.dot(d1, d2)
            denom = a * e - b * b  # positive unless segments are parallel

            if denom > _SMALL:
                # Segments are not parallel
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                # Parallel segments — choose s = 0 arbitrarily
                s = 0.0

            # Compute t using the chosen s
            t = (b * s + f) / e

            # Clamp t and recompute s if needed
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    closest_on_p = p0 + s * d1
    closest_on_q = q0 + t * d2
    dist = float(np.linalg.norm(closest_on_p - closest_on_q))
    return closest_on_p, closest_on_q, dist


def segment_to_segment_distance(
    p0: Vector3D, p1: Vector3D, q0: Vector3D, q1: Vector3D
) -> float:
    """Return the minimum distance between segment P (p0→p1) and Q (q0→q1).

    Parameters
    ----------
    p0, p1 : endpoints of segment P
    q0, q1 : endpoints of segment Q

    Returns
    -------
    float
        Minimum Euclidean distance between any point on P and any point on Q.
    """
    _, _, dist = segment_closest_points(p0, p1, q0, q1)
    return dist


def nanotube_distance(a: Nanotube, b: Nanotube) -> float:
    """Return the minimum distance between the axes of two Nanotubes."""
    return segment_to_segment_distance(a.p0, a.p1, b.p0, b.p1)


def nanotube_closest_points(a: Nanotube, b: Nanotube) -> tuple[Vector3D, Vector3D, float]:
    """Return closest points on tube A and tube B and minimum distance."""
    return segment_closest_points(a.p0, a.p1, b.p0, b.p1)

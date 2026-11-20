"""
CNT generator: random placement of Nanotubes inside a MaterialBox.

Physical assumptions:
- CNT centres are uniformly distributed inside the box.
- CNT orientations support isotropic (uniform on S^2 via Marsaglia normal-vector
  normalisation), strictly aligned, partially aligned (uniform solid-angle cone),
  and von Mises-Fisher distributions (Phase 7).
- Both endpoints of the CNT must lie inside the box; tubes that
  would cross a boundary are resampled (rejection sampling).
- The number of CNTs N is computed from the volumetric fraction phi:

      phi ≈ N * pi * D^2 * L / 4 / V_box
  ->  N = round( phi * V_box / (pi * D^2 * L / 4) )
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from percolation.domain.cnt import MaterialBox, Nanotube, OrientationConfig, _normalize_axis


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly distributed random unit vector on S^2 (Marsaglia method)."""
    v = rng.standard_normal(3)
    norm = np.linalg.norm(v)
    while norm < 1e-12:          # extremely unlikely; retry
        v = rng.standard_normal(3)
        norm = np.linalg.norm(v)
    return v / norm


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a right-handed orthonormal basis (e1, e2, a) with a = axis / ||axis||."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    # Pick a helper vector least aligned with a
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, helper)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2, a


def sample_direction(
    rng: np.random.Generator,
    orientation: OrientationConfig | None = None,
) -> np.ndarray:
    """Draw a single unit direction vector according to ``orientation``.

    Parameters
    ----------
    rng         : NumPy random Generator
    orientation : OrientationConfig (defaults to isotropic if None)

    Returns
    -------
    np.ndarray
        Unit 3D direction vector of shape (3,).
    """
    if orientation is None or orientation.mode == "isotropic":
        # Preserve exact RNG stream for default isotropic mode
        return _random_unit_vector(rng)

    if orientation.mode == "aligned":
        return orientation.axis_vector.copy()

    if orientation.mode == "partially_aligned":
        if orientation.max_angle_deg <= 0.0:
            return orientation.axis_vector.copy()
        cos_min = float(np.cos(np.deg2rad(orientation.max_angle_deg)))
        cos_theta = float(rng.uniform(cos_min, 1.0))
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
        azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
        e1, e2, a = _orthonormal_basis(orientation.axis_vector)
        vec = cos_theta * a + sin_theta * (np.cos(azimuth) * e1 + np.sin(azimuth) * e2)
        return vec / np.linalg.norm(vec)

    if orientation.mode == "fisher":
        kappa = orientation.kappa
        if kappa <= 0.0:
            return _random_unit_vector(rng)
        u = float(rng.uniform(0.0, 1.0))
        # Exact Wood/Ulrich inverse-CDF sampling on S^2:
        #   cos(theta) = 1 + ln(u + (1 - u) * exp(-2 * kappa)) / kappa
        if kappa > 350.0:
            # exp(-2*kappa) underflows to 0; clamp u away from 0
            cos_theta = 1.0 + float(np.log(max(u, 1e-300))) / kappa
        else:
            cos_theta = 1.0 + float(np.log(u + (1.0 - u) * np.exp(-2.0 * kappa))) / kappa
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
        azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
        e1, e2, a = _orthonormal_basis(orientation.axis_vector)
        vec = cos_theta * a + sin_theta * (np.cos(azimuth) * e1 + np.sin(azimuth) * e2)
        return vec / np.linalg.norm(vec)

    raise ValueError(f"Unsupported orientation mode: {orientation.mode!r}")


def compute_orientation_angles(
    tubes: Sequence[Nanotube],
    reference_axis: tuple[float, float, float] | str = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """Compute acute angle theta in [0, 90] degrees between each CNT and ``reference_axis``.

    Because CNTs are unoriented line segments (d and -d represent the same rod),
    the acute angle arccos(|d . r|) is used.
    """
    if not tubes:
        return np.empty(0, dtype=float)
    ref = np.asarray(_normalize_axis(reference_axis), dtype=float)
    dirs = np.vstack([t.direction for t in tubes])
    abs_cos = np.clip(np.abs(dirs @ ref), 0.0, 1.0)
    return np.rad2deg(np.arccos(abs_cos))


def compute_order_parameter(
    tubes: Sequence[Nanotube],
    reference_axis: tuple[float, float, float] | str = (1.0, 0.0, 0.0),
) -> float:
    """Compute empirical 3D Hermans orientation factor S along ``reference_axis``.

    S = (3 * <cos^2(theta)> - 1) / 2

    Interpretation:
    - S =  1.0 : perfectly aligned parallel to reference_axis
    - S =  0.0 : isotropic (random 3D)
    - S = -0.5 : perfectly aligned perpendicular to reference_axis
    """
    if not tubes:
        return 0.0
    ref = np.asarray(_normalize_axis(reference_axis), dtype=float)
    dirs = np.vstack([t.direction for t in tubes])
    cos_vals = dirs @ ref
    mean_cos2 = float(np.mean(cos_vals * cos_vals))
    return 0.5 * (3.0 * mean_cos2 - 1.0)


def count_from_concentration(phi: float, box: MaterialBox, cnt_length: float, cnt_diameter: float) -> int:
    """Compute N (number of CNTs) from volumetric fraction phi.

    V_cnt_single ≈ pi * D^2 * L / 4   (cylinder without hemispherical caps)
    N = phi * V_box / V_cnt_single
    """
    v_box = box.volume()
    v_cnt = np.pi * cnt_diameter**2 * cnt_length / 4.0
    return max(1, round(phi * v_box / v_cnt))


def generate_nanotubes(
    n: int,
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    rng: np.random.Generator,
    max_attempts_per_tube: int = 1_000,
    orientation: OrientationConfig | None = None,
) -> list[Nanotube]:
    """Generate *n* Nanotubes randomly placed inside *box*.

    Tubes whose endpoints fall outside the box are rejected and resampled.

    Parameters
    ----------
    n                     : number of CNTs to place
    box                   : simulation domain
    cnt_length            : CNT length L, um
    cnt_diameter          : CNT diameter D, um
    rng                   : NumPy random Generator (for reproducibility)
    max_attempts_per_tube : safety limit for rejection sampling
    orientation           : optional OrientationConfig controlling 3D orientation
                            distribution (defaults to isotropic)

    Returns
    -------
    list[Nanotube]
        Exactly n Nanotubes with valid (inside-box) placement.

    Raises
    ------
    RuntimeError
        If a tube cannot be placed after max_attempts_per_tube retries.
        (Happens only for pathologically large CNTs relative to the box.)
    """
    tubes: list[Nanotube] = []

    for i in range(n):
        placed = False
        for _ in range(max_attempts_per_tube):
            center = np.array([
                rng.uniform(0.0, box.size_x),
                rng.uniform(0.0, box.size_y),
                rng.uniform(0.0, box.size_z),
            ])
            direction = sample_direction(rng, orientation)
            tube = Nanotube(center=center, direction=direction, length=cnt_length, diameter=cnt_diameter)
            if box.contains_segment(tube.p0, tube.p1):
                tubes.append(tube)
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place CNT {i} inside box after {max_attempts_per_tube} attempts. "
                "CNT length may be too large relative to the box dimensions."
            )

    return tubes

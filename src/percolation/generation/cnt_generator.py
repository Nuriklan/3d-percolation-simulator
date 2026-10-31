"""
CNT generator: random placement of Nanotubes inside a MaterialBox.

Physical assumptions:
- CNT centres are uniformly distributed inside the box.
- CNT orientations are uniformly distributed on the unit sphere
  using the Marsaglia (normal-vector normalisation) method — NOT
  independent uniform (θ, φ), which would over-sample the poles.
- Both endpoints of the CNT must lie inside the box; tubes that
  would cross a boundary are resampled (rejection sampling).
- The number of CNTs N is computed from the volumetric fraction φ:

      φ ≈ N · π · D² · L / 4 / V_box
  →   N = round( φ · V_box / (π · D² · L / 4) )

"""

from __future__ import annotations

import numpy as np

from percolation.domain.cnt import MaterialBox, Nanotube


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly distributed random unit vector on S² (Marsaglia method)."""
    v = rng.standard_normal(3)
    norm = np.linalg.norm(v)
    while norm < 1e-12:          # extremely unlikely; retry
        v = rng.standard_normal(3)
        norm = np.linalg.norm(v)
    return v / norm


def count_from_concentration(phi: float, box: MaterialBox, cnt_length: float, cnt_diameter: float) -> int:
    """Compute N (number of CNTs) from volumetric fraction φ.

    V_cnt_single ≈ π * D² * L / 4   (cylinder without hemispherical caps)
    N = φ * V_box / V_cnt_single
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
) -> list[Nanotube]:
    """Generate *n* Nanotubes randomly placed inside *box*.

    Tubes whose endpoints fall outside the box are rejected and resampled.

    Parameters
    ----------
    n                     : number of CNTs to place
    box                   : simulation domain
    cnt_length            : CNT length L, μm
    cnt_diameter          : CNT diameter D, μm
    rng                   : NumPy random Generator (for reproducibility)
    max_attempts_per_tube : safety limit for rejection sampling

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
            direction = _random_unit_vector(rng)
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

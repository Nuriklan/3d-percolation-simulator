"""
Unit tests for CNT generation.

Covers:
- All tubes fit inside the box
- Direction vectors are unit vectors
- count_from_concentration formula
- Reproducibility with fixed seed
"""

import numpy as np
import pytest

from percolation.domain.cnt import MaterialBox
from percolation.generation.cnt_generator import (
    count_from_concentration,
    generate_nanotubes,
)

BOX = MaterialBox(size_x=10.0, size_y=5.0, size_z=5.0)
L = 1.0
D = 0.01


def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_tubes_inside_box():
    rng = make_rng(0)
    tubes = generate_nanotubes(50, BOX, L, D, rng)
    for t in tubes:
        assert BOX.contains_segment(t.p0, t.p1), (
            f"Tube endpoints {t.p0} – {t.p1} are outside box"
        )


def test_direction_is_unit_vector():
    rng = make_rng(1)
    tubes = generate_nanotubes(20, BOX, L, D, rng)
    for t in tubes:
        norm = np.linalg.norm(t.direction)
        assert norm == pytest.approx(1.0, rel=1e-9), f"Direction norm = {norm}"


def test_endpoint_distance_equals_length():
    """p1 - p0 should have magnitude = length."""
    rng = make_rng(2)
    tubes = generate_nanotubes(20, BOX, L, D, rng)
    for t in tubes:
        dist = np.linalg.norm(t.p1 - t.p0)
        assert dist == pytest.approx(L, rel=1e-9)


def test_count_from_concentration_basic():
    phi = 0.004
    n = count_from_concentration(phi, BOX, L, D)
    # rough check: should be many tubes
    assert n > 0
    # verify inverse: recomputed phi should be close
    import math
    v_cnt = math.pi * D**2 * L / 4
    phi_back = n * v_cnt / BOX.volume()
    assert abs(phi_back - phi) / phi < 0.05  # within 5% due to rounding


def test_reproducibility():
    """Same seed must give identical tube positions."""
    rng1 = make_rng(99)
    rng2 = make_rng(99)
    tubes1 = generate_nanotubes(30, BOX, L, D, rng1)
    tubes2 = generate_nanotubes(30, BOX, L, D, rng2)
    for t1, t2 in zip(tubes1, tubes2):
        np.testing.assert_array_almost_equal(t1.center, t2.center)
        np.testing.assert_array_almost_equal(t1.direction, t2.direction)


def test_different_seeds_differ():
    """Different seeds should (almost certainly) produce different results."""
    rng1 = make_rng(1)
    rng2 = make_rng(2)
    tubes1 = generate_nanotubes(10, BOX, L, D, rng1)
    tubes2 = generate_nanotubes(10, BOX, L, D, rng2)
    any_diff = any(
        not np.allclose(t1.center, t2.center) for t1, t2 in zip(tubes1, tubes2)
    )
    assert any_diff, "Different seeds produced identical tubes — very unlikely"

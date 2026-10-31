"""
Tests for Phase 5: Spatial Optimization (Grid, Hash, KD-tree, and vectorized distance).
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from percolation.domain.cnt import MaterialBox, Nanotube
from percolation.generation.cnt_generator import generate_nanotubes
from percolation.geometry.distance import (
    batch_segment_distance,
    segment_to_segment_distance,
)
from percolation.geometry.spatial import (
    KDTreeIndex,
    SpatialHashGrid,
    UniformSpatialGrid,
    compare_contact_methods,
    find_contacts_brute_force,
)
from percolation.graph.contact_graph import find_contacts
from percolation.simulation.experiment import create_sample, run_sample
from percolation.simulation.monte_carlo import run_concentration


def make_tube(p0: list[float], p1: list[float], diameter: float = 0.1) -> Nanotube:
    p0_arr = np.array(p0, dtype=np.float64)
    p1_arr = np.array(p1, dtype=np.float64)
    length = float(np.linalg.norm(p1_arr - p0_arr))
    center = 0.5 * (p0_arr + p1_arr)
    direction = (p1_arr - p0_arr) / length if length > 0 else np.array([1.0, 0.0, 0.0])
    return Nanotube(center=center, direction=direction, length=length, diameter=diameter)


class TestBatchSegmentDistance:
    """Tests for vectorized batch_segment_distance function."""

    def test_empty_batch(self) -> None:
        p0 = np.empty((0, 3))
        p1 = np.empty((0, 3))
        q0 = np.empty((0, 3))
        q1 = np.empty((0, 3))
        res = batch_segment_distance(p0, p1, q0, q1)
        assert len(res) == 0

    def test_known_distances_and_equivalence(self) -> None:
        rng = np.random.default_rng(12345)
        n = 100
        p0 = rng.uniform(-5.0, 5.0, (n, 3))
        p1 = p0 + rng.uniform(-1.0, 1.0, (n, 3))
        q0 = rng.uniform(-5.0, 5.0, (n, 3))
        q1 = q0 + rng.uniform(-1.0, 1.0, (n, 3))

        # Include edge cases:
        # Parallel
        p0[0] = [0.0, 0.0, 0.0]; p1[0] = [1.0, 0.0, 0.0]
        q0[0] = [0.0, 2.0, 0.0]; q1[0] = [1.0, 2.0, 0.0]

        # Intersecting perpendicular
        p0[1] = [-1.0, 0.0, 0.0]; p1[1] = [1.0, 0.0, 0.0]
        q0[1] = [0.0, -1.0, 0.0]; q1[1] = [0.0, 1.0, 0.0]

        # Degenerate P (point)
        p0[2] = [1.0, 1.0, 1.0]; p1[2] = [1.0, 1.0, 1.0]
        q0[2] = [0.0, 0.0, 0.0]; q1[2] = [2.0, 0.0, 0.0]

        # Degenerate Q (point)
        p0[3] = [0.0, 0.0, 0.0]; p1[3] = [0.0, 2.0, 0.0]
        q0[3] = [2.0, 1.0, 0.0]; q1[3] = [2.0, 1.0, 0.0]

        # Both degenerate (two points)
        p0[4] = [1.0, 2.0, 3.0]; p1[4] = [1.0, 2.0, 3.0]
        q0[4] = [4.0, 2.0, 7.0]; q1[4] = [4.0, 2.0, 7.0]

        batch_d = batch_segment_distance(p0, p1, q0, q1)
        scalar_d = np.array([
            segment_to_segment_distance(p0[i], p1[i], q0[i], q1[i])
            for i in range(n)
        ])

        assert np.allclose(batch_d, scalar_d, atol=1e-12)
        assert abs(batch_d[0] - 2.0) < 1e-12  # parallel at distance 2.0
        assert abs(batch_d[1] - 0.0) < 1e-12  # intersecting
        assert abs(batch_d[4] - 5.0) < 1e-12  # sqrt((4-1)^2 + 0 + (7-3)^2) = 5.0


class TestUniformSpatialGrid:
    """Unit tests for UniformSpatialGrid."""

    def test_empty_system(self) -> None:
        grid = UniformSpatialGrid([], threshold=0.1)
        assert grid.query_candidate_pairs() == []
        assert grid.find_contacts() == []

    def test_single_tube(self) -> None:
        tube = make_tube([0, 0, 0], [1, 0, 0])
        grid = UniformSpatialGrid([tube], threshold=0.1)
        assert grid.query_candidate_pairs() == []
        assert grid.find_contacts() == []

    def test_touching_and_separated_pairs(self) -> None:
        t0 = make_tube([0, 0, 0], [1, 0, 0])
        t1 = make_tube([0.5, 0.05, 0], [0.5, 1.0, 0])  # touching at dist 0.05
        t2 = make_tube([10, 10, 10], [11, 10, 10])     # far away

        box = MaterialBox(12.0, 12.0, 12.0)
        grid = UniformSpatialGrid([t0, t1, t2], threshold=0.1, box=box)
        candidates = grid.query_candidate_pairs()

        # (0, 1) must be in candidates
        assert (0, 1) in candidates
        # (0, 2) and (1, 2) should NOT be candidates due to spatial separation
        assert (0, 2) not in candidates
        assert (1, 2) not in candidates

        contacts = grid.find_contacts()
        assert len(contacts) == 1
        assert contacts[0].tube_a == 0
        assert contacts[0].tube_b == 1
        assert abs(contacts[0].distance - 0.05) < 1e-6

    def test_matches_brute_force_on_random_network(self) -> None:
        box = MaterialBox(8.0, 4.0, 4.0)
        rng = np.random.default_rng(99)
        tubes = generate_nanotubes(80, box, cnt_length=1.0, cnt_diameter=0.1, rng=rng)
        threshold = 0.08

        c_bf = find_contacts_brute_force(tubes, threshold)
        grid = UniformSpatialGrid(tubes, threshold, box=box)
        c_grid = grid.find_contacts()

        set_bf = {(c.tube_a, c.tube_b) for c in c_bf}
        set_grid = {(c.tube_a, c.tube_b) for c in c_grid}

        assert set_bf == set_grid
        assert len(set_grid) > 0  # ensure non-trivial network


class TestSpatialHashGrid:
    """Unit tests for SpatialHashGrid."""

    def test_empty_and_single(self) -> None:
        h = SpatialHashGrid([], threshold=0.1)
        assert h.query_candidate_pairs() == []
        assert h.find_contacts() == []

        tube = make_tube([0, 0, 0], [1, 0, 0])
        h1 = SpatialHashGrid([tube], threshold=0.1)
        assert h1.query_candidate_pairs() == []
        assert h1.find_contacts() == []

    def test_matches_brute_force(self) -> None:
        box = MaterialBox(6.0, 6.0, 6.0)
        rng = np.random.default_rng(101)
        tubes = generate_nanotubes(100, box, cnt_length=1.0, cnt_diameter=0.1, rng=rng)
        threshold = 0.05

        c_bf = find_contacts_brute_force(tubes, threshold)
        s_hash = SpatialHashGrid(tubes, threshold)
        c_hash = s_hash.find_contacts()

        set_bf = {(c.tube_a, c.tube_b) for c in c_bf}
        set_hash = {(c.tube_a, c.tube_b) for c in c_hash}

        assert set_bf == set_hash


class TestKDTreeIndex:
    """Unit tests for KDTreeIndex."""

    def test_empty_and_single(self) -> None:
        tree = KDTreeIndex([], threshold=0.1)
        assert tree.query_candidate_pairs() == []
        assert tree.find_contacts() == []

        tube = make_tube([0, 0, 0], [1, 0, 0])
        t1 = KDTreeIndex([tube], threshold=0.1)
        assert t1.query_candidate_pairs() == []
        assert t1.find_contacts() == []

    def test_matches_brute_force(self) -> None:
        box = MaterialBox(10.0, 5.0, 5.0)
        rng = np.random.default_rng(202)
        tubes = generate_nanotubes(120, box, cnt_length=1.0, cnt_diameter=0.1, rng=rng)
        threshold = 0.06

        c_bf = find_contacts_brute_force(tubes, threshold)
        tree = KDTreeIndex(tubes, threshold)
        c_kd = tree.find_contacts()

        set_bf = {(c.tube_a, c.tube_b) for c in c_bf}
        set_kd = {(c.tube_a, c.tube_b) for c in c_kd}

        assert set_bf == set_kd


class TestCompareContactMethods:
    """Tests for compare_contact_methods benchmarking and validation suite."""

    def test_cross_verification(self) -> None:
        box = MaterialBox(8.0, 4.0, 4.0)
        rng = np.random.default_rng(303)
        tubes = generate_nanotubes(150, box, cnt_length=1.0, cnt_diameter=0.1, rng=rng)
        threshold = 0.05

        results = compare_contact_methods(tubes, threshold, box=box)

        assert "brute_force" in results
        assert "grid" in results
        assert "hash" in results
        assert "kdtree" in results

        # All must have identified the exact same count of contacts
        n_contacts = results["brute_force"].contacts_found
        assert n_contacts > 0
        assert results["grid"].contacts_found == n_contacts
        assert results["hash"].contacts_found == n_contacts
        assert results["kdtree"].contacts_found == n_contacts

        # Candidate pair reduction must be substantial (> 75%)
        assert results["grid"].candidate_reduction_pct > 75.0
        assert results["kdtree"].candidate_reduction_pct > 75.0


class TestFindContactsIntegration:
    """Tests for find_contacts supporting multiple acceleration backends."""

    def test_all_methods_support(self) -> None:
        box = MaterialBox(5.0, 5.0, 5.0)
        rng = np.random.default_rng(404)
        tubes = generate_nanotubes(60, box, 1.0, 0.1, rng)
        threshold = 0.05

        methods = ["auto", "grid", "hash", "kdtree", "brute_force"]
        results_by_method = {}

        for m in methods:
            contacts = find_contacts(tubes, threshold, box=box, method=m)
            results_by_method[m] = [(c.tube_a, c.tube_b, round(c.distance, 8)) for c in contacts]

        ref = results_by_method["brute_force"]
        for m in methods:
            assert results_by_method[m] == ref, f"Method {m} did not match brute_force"

    def test_invalid_method_raises(self) -> None:
        tube = make_tube([0, 0, 0], [1, 0, 0])
        with pytest.raises(ValueError, match="Unknown contact detection method"):
            find_contacts([tube, tube], threshold=0.1, method="quantum_search")


class TestSimulationIntegration:
    """Verify end-to-end simulation pipelines with spatial acceleration."""

    def test_create_sample_methods_match(self) -> None:
        box = MaterialBox(10.0, 5.0, 5.0)

        for m in ["grid", "kdtree", "hash", "brute_force"]:
            rng = np.random.default_rng(505)
            sample = create_sample(
                concentration=0.005,
                box=box,
                cnt_length=1.0,
                cnt_diameter=0.1,
                contact_threshold=0.05,
                rng=rng,
                contact_method=m,
            )
            assert sample.nanotube_count > 0
            assert sample.contact_count > 0

    def test_run_sample_methods_match(self) -> None:
        box = MaterialBox(10.0, 5.0, 5.0)

        results = {}
        for m in ["grid", "kdtree", "hash", "brute_force"]:
            rng = np.random.default_rng(606)
            res = run_sample(
                concentration=0.005,
                box=box,
                cnt_length=1.0,
                cnt_diameter=0.1,
                contact_threshold=0.05,
                rng=rng,
                contact_method=m,
            )
            results[m] = (res.percolates, res.contact_count, res.nanotube_count)

        ref = results["brute_force"]
        for m in ["grid", "kdtree", "hash"]:
            assert results[m] == ref, f"Method {m} produced different simulation result than brute_force"

    def test_run_concentration_with_grid(self) -> None:
        box = MaterialBox(6.0, 3.0, 3.0)
        c_res = run_concentration(
            concentration=0.005,
            samples=5,
            box=box,
            cnt_length=1.0,
            cnt_diameter=0.1,
            contact_threshold=0.05,
            seed_sequence=707,
            contact_method="grid",
        )
        assert c_res.samples == 5
        assert 0.0 <= c_res.probability <= 1.0


class TestCliRunner:
    """Test CLI entry point for spatial benchmark."""

    def test_cli_execution(self) -> None:
        cmd = [sys.executable, "-m", "percolation.geometry.spatial", "--count", "50"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 0
        assert "CNT Contact Detection Benchmark" in proc.stdout
        assert "All methods verified: 100% identical contacts identified!" in proc.stdout

"""
Spatial acceleration structures for fast 3D CNT contact detection.

Phase 5 implementation:
- Replaces brute-force O(N^2) pairwise distance checks with spatial partitioning.
- UniformSpatialGrid: uniform 3D grid with AABB-based cell insertion.
- SpatialHashGrid: hash-table based sparse 3D spatial partitioning.
- KDTreeIndex: KD-tree on tube bounding spheres using scipy.spatial.cKDTree.
- compare_contact_methods: benchmark and verification suite ensuring zero false
  negatives and 100% equivalence to brute force.

Theoretical guarantee:
    Two line segments P and Q can have minimum distance d(P, Q) <= threshold
    only if their axis-aligned bounding boxes (AABBs), each dilated by
    threshold / 2, overlap. Therefore, any spatial cell containing the midpoint
    of the closest pair of points will contain both tubes in its dilated cell
    coverage. This guarantees zero false negatives.
"""

from __future__ import annotations

import argparse
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from percolation.domain.cnt import Contact, MaterialBox, Nanotube
from percolation.geometry.distance import (
    batch_segment_distance,
    nanotube_distance,
)


@dataclass(frozen=True)
class SpatialBenchmarkResult:
    """Benchmark metrics for a specific contact detection method."""

    method: str
    execution_time_sec: float
    candidate_pairs_count: int
    contacts_found: int
    total_possible_pairs: int
    candidate_reduction_pct: float
    speedup_vs_brute_force: float


class SpatialIndex(ABC):
    """Abstract base class for spatial acceleration indices."""

    @abstractmethod
    def query_candidate_pairs(self) -> list[tuple[int, int]]:
        """Return unique candidate tube index pairs (i, j) with i < j."""
        ...

    @abstractmethod
    def find_contacts(self) -> list[Contact]:
        """Evaluate contacts among candidate pairs and return all contacts."""
        ...


class UniformSpatialGrid(SpatialIndex):
    """Uniform 3D spatial partitioning grid.

    Divides the simulation domain into a regular 3D grid of cubic or rectangular
    cells. Each nanotube's axis-aligned bounding box (dilated by threshold / 2)
    is inserted into all grid cells it overlaps.

    Candidate pairs are generated from cells containing two or more tubes.
    """

    def __init__(
        self,
        tubes: Sequence[Nanotube],
        threshold: float,
        box: MaterialBox | None = None,
        cell_size: float | None = None,
    ) -> None:
        self.tubes = list(tubes)
        self.threshold = float(threshold)
        self.n_tubes = len(self.tubes)

        if self.n_tubes == 0:
            self.cell_size = 1.0 if cell_size is None else float(cell_size)
            self.origin = np.zeros(3, dtype=np.float64)
            self.grid_shape = (1, 1, 1)
            self.cells: dict[int, list[int]] = {}
            return

        # Determine spatial bounding domain
        if box is not None:
            self.origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            box_extent = np.array([box.size_x, box.size_y, box.size_z], dtype=np.float64)
        else:
            p0_arr = np.array([t.p0 for t in self.tubes])
            p1_arr = np.array([t.p1 for t in self.tubes])
            all_pts = np.vstack([p0_arr, p1_arr])
            self.origin = np.min(all_pts, axis=0) - self.threshold
            box_extent = np.max(all_pts, axis=0) + self.threshold - self.origin

        # Automatic cell size: default to max tube length + threshold
        if cell_size is not None and cell_size > 0:
            self.cell_size = float(cell_size)
        else:
            max_len = max((t.length for t in self.tubes), default=1.0)
            self.cell_size = float(max_len + self.threshold)

        # Minimum cell size safeguard
        if self.cell_size <= 1e-9:
            self.cell_size = 1.0

        nx = max(1, math.ceil(box_extent[0] / self.cell_size))
        ny = max(1, math.ceil(box_extent[1] / self.cell_size))
        nz = max(1, math.ceil(box_extent[2] / self.cell_size))
        self.grid_shape = (nx, ny, nz)

        self._build_grid()

    def _build_grid(self) -> None:
        """Insert all nanotubes into their overlapping grid cells."""
        nx, ny, nz = self.grid_shape
        cs = self.cell_size
        ox, oy, oz = self.origin
        half_th = 0.5 * self.threshold

        self.cells: dict[int, list[int]] = {}

        for idx, tube in enumerate(self.tubes):
            p0 = tube.p0
            p1 = tube.p1

            x_min = min(p0[0], p1[0]) - half_th
            x_max = max(p0[0], p1[0]) + half_th
            y_min = min(p0[1], p1[1]) - half_th
            y_max = max(p0[1], p1[1]) + half_th
            z_min = min(p0[2], p1[2]) - half_th
            z_max = max(p0[2], p1[2]) + half_th

            ix0 = max(0, min(nx - 1, math.floor((x_min - ox) / cs)))
            ix1 = max(0, min(nx - 1, math.floor((x_max - ox) / cs)))
            iy0 = max(0, min(ny - 1, math.floor((y_min - oy) / cs)))
            iy1 = max(0, min(ny - 1, math.floor((y_max - oy) / cs)))
            iz0 = max(0, min(nz - 1, math.floor((z_min - oz) / cs)))
            iz1 = max(0, min(nz - 1, math.floor((z_max - oz) / cs)))

            for ix in range(ix0, ix1 + 1):
                base_x = ix * ny * nz
                for iy in range(iy0, iy1 + 1):
                    base_xy = base_x + iy * nz
                    for iz in range(iz0, iz1 + 1):
                        cell_key = base_xy + iz
                        if cell_key not in self.cells:
                            self.cells[cell_key] = [idx]
                        else:
                            self.cells[cell_key].append(idx)

    def query_candidate_pairs(self) -> list[tuple[int, int]]:
        """Return unique candidate pairs (i, j) with i < j."""
        if self.n_tubes < 2:
            return []

        candidate_set: set[tuple[int, int]] = set()
        for cell_tubes in self.cells.values():
            count = len(cell_tubes)
            if count < 2:
                continue
            for a in range(count):
                u = cell_tubes[a]
                for b in range(a + 1, count):
                    v = cell_tubes[b]
                    if u < v:
                        candidate_set.add((u, v))
                    elif u > v:
                        candidate_set.add((v, u))

        # Return sorted list for determinism
        return sorted(candidate_set)

    def find_contacts(self) -> list[Contact]:
        """Compute exact contacts among candidate pairs."""
        candidates = self.query_candidate_pairs()
        if not candidates:
            return []

        # Use batch segment distance for maximum performance
        pairs_arr = np.array(candidates, dtype=np.int32)
        u_idx = pairs_arr[:, 0]
        v_idx = pairs_arr[:, 1]

        p0_all = np.array([t.p0 for t in self.tubes], dtype=np.float64)
        p1_all = np.array([t.p1 for t in self.tubes], dtype=np.float64)

        p0_batch = p0_all[u_idx]
        p1_batch = p1_all[u_idx]
        q0_batch = p0_all[v_idx]
        q1_batch = p1_all[v_idx]

        distances = batch_segment_distance(p0_batch, p1_batch, q0_batch, q1_batch)

        mask = distances <= self.threshold
        contact_indices = np.nonzero(mask)[0]

        contacts: list[Contact] = []
        for k in contact_indices:
            contacts.append(
                Contact(
                    tube_a=int(u_idx[k]),
                    tube_b=int(v_idx[k]),
                    distance=float(distances[k]),
                )
            )

        return contacts


class SpatialHashGrid(SpatialIndex):
    """Sparse 3D spatial hash grid.

    Uses a dictionary mapping 3D integer coordinates (gx, gy, gz) to tube indices.
    Does not require fixed domain boundaries and uses memory strictly proportional
    to occupied space.
    """

    def __init__(
        self,
        tubes: Sequence[Nanotube],
        threshold: float,
        cell_size: float | None = None,
    ) -> None:
        self.tubes = list(tubes)
        self.threshold = float(threshold)
        self.n_tubes = len(self.tubes)

        if cell_size is not None and cell_size > 0:
            self.cell_size = float(cell_size)
        else:
            max_len = max((t.length for t in self.tubes), default=1.0)
            self.cell_size = float(max_len + self.threshold)

        if self.cell_size <= 1e-9:
            self.cell_size = 1.0

        self.table: dict[tuple[int, int, int], list[int]] = {}
        self._build_hash_table()

    def _build_hash_table(self) -> None:
        cs = self.cell_size
        half_th = 0.5 * self.threshold

        for idx, tube in enumerate(self.tubes):
            p0 = tube.p0
            p1 = tube.p1

            x_min = min(p0[0], p1[0]) - half_th
            x_max = max(p0[0], p1[0]) + half_th
            y_min = min(p0[1], p1[1]) - half_th
            y_max = max(p0[1], p1[1]) + half_th
            z_min = min(p0[2], p1[2]) - half_th
            z_max = max(p0[2], p1[2]) + half_th

            gx0 = math.floor(x_min / cs)
            gx1 = math.floor(x_max / cs)
            gy0 = math.floor(y_min / cs)
            gy1 = math.floor(y_max / cs)
            gz0 = math.floor(z_min / cs)
            gz1 = math.floor(z_max / cs)

            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    for gz in range(gz0, gz1 + 1):
                        key = (gx, gy, gz)
                        if key not in self.table:
                            self.table[key] = [idx]
                        else:
                            self.table[key].append(idx)

    def query_candidate_pairs(self) -> list[tuple[int, int]]:
        if self.n_tubes < 2:
            return []

        candidate_set: set[tuple[int, int]] = set()
        for cell_tubes in self.table.values():
            count = len(cell_tubes)
            if count < 2:
                continue
            for a in range(count):
                u = cell_tubes[a]
                for b in range(a + 1, count):
                    v = cell_tubes[b]
                    if u < v:
                        candidate_set.add((u, v))
                    elif u > v:
                        candidate_set.add((v, u))

        return sorted(candidate_set)

    def find_contacts(self) -> list[Contact]:
        candidates = self.query_candidate_pairs()
        if not candidates:
            return []

        pairs_arr = np.array(candidates, dtype=np.int32)
        u_idx = pairs_arr[:, 0]
        v_idx = pairs_arr[:, 1]

        p0_all = np.array([t.p0 for t in self.tubes], dtype=np.float64)
        p1_all = np.array([t.p1 for t in self.tubes], dtype=np.float64)

        p0_batch = p0_all[u_idx]
        p1_batch = p1_all[u_idx]
        q0_batch = p0_all[v_idx]
        q1_batch = p1_all[v_idx]

        distances = batch_segment_distance(p0_batch, p1_batch, q0_batch, q1_batch)

        mask = distances <= self.threshold
        contact_indices = np.nonzero(mask)[0]

        contacts: list[Contact] = []
        for k in contact_indices:
            contacts.append(
                Contact(
                    tube_a=int(u_idx[k]),
                    tube_b=int(v_idx[k]),
                    distance=float(distances[k]),
                )
            )

        return contacts


class KDTreeIndex(SpatialIndex):
    """Spatial index using SciPy's C++ cKDTree on tube bounding spheres.

    Each tube has center C_i and radius R_i = L_i / 2.
    Two tubes can be within distance <= threshold only if:
        ||C_i - C_j|| <= R_i + R_j + threshold
    For monodisperse tubes of length L, the query radius is r = L + threshold.
    """

    def __init__(
        self,
        tubes: Sequence[Nanotube],
        threshold: float,
    ) -> None:
        self.tubes = list(tubes)
        self.threshold = float(threshold)
        self.n_tubes = len(self.tubes)

        if self.n_tubes >= 2:
            self.centers = np.array([t.center for t in self.tubes], dtype=np.float64)
            self.tree = cKDTree(self.centers)
            max_len = max((t.length for t in self.tubes), default=1.0)
            self.search_radius = float(max_len + self.threshold)
        else:
            self.centers = np.empty((0, 3), dtype=np.float64)
            self.tree = None
            self.search_radius = 0.0

    def query_candidate_pairs(self) -> list[tuple[int, int]]:
        if self.n_tubes < 2 or self.tree is None:
            return []

        pairs_set = self.tree.query_pairs(self.search_radius)
        return sorted(pairs_set)

    def find_contacts(self) -> list[Contact]:
        candidates = self.query_candidate_pairs()
        if not candidates:
            return []

        pairs_arr = np.array(candidates, dtype=np.int32)
        u_idx = pairs_arr[:, 0]
        v_idx = pairs_arr[:, 1]

        p0_all = np.array([t.p0 for t in self.tubes], dtype=np.float64)
        p1_all = np.array([t.p1 for t in self.tubes], dtype=np.float64)

        p0_batch = p0_all[u_idx]
        p1_batch = p1_all[u_idx]
        q0_batch = p0_all[v_idx]
        q1_batch = p1_all[v_idx]

        distances = batch_segment_distance(p0_batch, p1_batch, q0_batch, q1_batch)

        mask = distances <= self.threshold
        contact_indices = np.nonzero(mask)[0]

        contacts: list[Contact] = []
        for k in contact_indices:
            contacts.append(
                Contact(
                    tube_a=int(u_idx[k]),
                    tube_b=int(v_idx[k]),
                    distance=float(distances[k]),
                )
            )

        return contacts


def find_contacts_brute_force(
    tubes: Sequence[Nanotube],
    threshold: float,
) -> list[Contact]:
    """Reference brute-force pairwise contact detection with bounding rejection."""
    contacts: list[Contact] = []
    n = len(tubes)
    if n < 2:
        return contacts

    centers = [t.center for t in tubes]
    half_lengths = [0.5 * t.length for t in tubes]

    for i in range(n):
        ci = centers[i]
        hl_i = half_lengths[i]
        tube_i = tubes[i]
        for j in range(i + 1, n):
            reach = hl_i + half_lengths[j] + threshold
            cj = centers[j]
            dx = abs(ci[0] - cj[0])
            if dx > reach:
                continue
            dy = abs(ci[1] - cj[1])
            if dy > reach:
                continue
            dz = abs(ci[2] - cj[2])
            if dz > reach:
                continue

            if (dx * dx + dy * dy + dz * dz) > reach * reach:
                continue

            d = nanotube_distance(tube_i, tubes[j])
            if d <= threshold:
                contacts.append(Contact(tube_a=i, tube_b=j, distance=d))

    return sorted(contacts, key=lambda c: (c.tube_a, c.tube_b))


def compare_contact_methods(
    tubes: Sequence[Nanotube],
    threshold: float,
    box: MaterialBox | None = None,
    methods: Sequence[str] | None = None,
    cell_size: float | None = None,
) -> dict[str, SpatialBenchmarkResult]:
    """Benchmark and cross-verify contact detection algorithms.

    Runs each selected method, asserts identical contact sets, and computes
    execution speed and candidate pair reduction metrics.

    Parameters
    ----------
    tubes     : Sequence of Nanotubes
    threshold : Contact distance threshold
    box       : Simulation domain (optional)
    methods   : Methods to benchmark. Defaults to:
                ["brute_force", "grid", "hash", "kdtree"]
    cell_size : Custom cell size for grid and hash methods

    Returns
    -------
    dict[str, SpatialBenchmarkResult]
    """
    if methods is None:
        methods = ["brute_force", "grid", "hash", "kdtree"]

    n = len(tubes)
    total_pairs = (n * (n - 1)) // 2 if n >= 2 else 0

    results: dict[str, SpatialBenchmarkResult] = {}
    contacts_by_method: dict[str, list[Contact]] = {}

    # 1. Run Brute Force first if requested, to serve as reference
    ref_time = 0.0
    if "brute_force" in methods:
        t0 = time.perf_counter()
        c_bf = find_contacts_brute_force(tubes, threshold)
        t_bf = time.perf_counter() - t0
        ref_time = t_bf
        contacts_by_method["brute_force"] = c_bf
        results["brute_force"] = SpatialBenchmarkResult(
            method="brute_force",
            execution_time_sec=t_bf,
            candidate_pairs_count=total_pairs,
            contacts_found=len(c_bf),
            total_possible_pairs=total_pairs,
            candidate_reduction_pct=0.0,
            speedup_vs_brute_force=1.0,
        )

    # 2. Run Spatial Grid
    if "grid" in methods:
        t0 = time.perf_counter()
        grid = UniformSpatialGrid(tubes, threshold, box=box, cell_size=cell_size)
        candidates = grid.query_candidate_pairs()
        c_grid = grid.find_contacts()
        t_grid = time.perf_counter() - t0
        contacts_by_method["grid"] = c_grid
        reduc = 100.0 * (1.0 - len(candidates) / total_pairs) if total_pairs > 0 else 0.0
        speedup = (ref_time / t_grid) if t_grid > 0 and ref_time > 0 else 1.0
        results["grid"] = SpatialBenchmarkResult(
            method="grid",
            execution_time_sec=t_grid,
            candidate_pairs_count=len(candidates),
            contacts_found=len(c_grid),
            total_possible_pairs=total_pairs,
            candidate_reduction_pct=reduc,
            speedup_vs_brute_force=speedup,
        )

    # 3. Run Spatial Hash
    if "hash" in methods:
        t0 = time.perf_counter()
        s_hash = SpatialHashGrid(tubes, threshold, cell_size=cell_size)
        candidates = s_hash.query_candidate_pairs()
        c_hash = s_hash.find_contacts()
        t_hash = time.perf_counter() - t0
        contacts_by_method["hash"] = c_hash
        reduc = 100.0 * (1.0 - len(candidates) / total_pairs) if total_pairs > 0 else 0.0
        speedup = (ref_time / t_hash) if t_hash > 0 and ref_time > 0 else 1.0
        results["hash"] = SpatialBenchmarkResult(
            method="hash",
            execution_time_sec=t_hash,
            candidate_pairs_count=len(candidates),
            contacts_found=len(c_hash),
            total_possible_pairs=total_pairs,
            candidate_reduction_pct=reduc,
            speedup_vs_brute_force=speedup,
        )

    # 4. Run KDTree
    if "kdtree" in methods:
        t0 = time.perf_counter()
        tree_idx = KDTreeIndex(tubes, threshold)
        candidates = tree_idx.query_candidate_pairs()
        c_kd = tree_idx.find_contacts()
        t_kd = time.perf_counter() - t0
        contacts_by_method["kdtree"] = c_kd
        reduc = 100.0 * (1.0 - len(candidates) / total_pairs) if total_pairs > 0 else 0.0
        speedup = (ref_time / t_kd) if t_kd > 0 and ref_time > 0 else 1.0
        results["kdtree"] = SpatialBenchmarkResult(
            method="kdtree",
            execution_time_sec=t_kd,
            candidate_pairs_count=len(candidates),
            contacts_found=len(c_kd),
            total_possible_pairs=total_pairs,
            candidate_reduction_pct=reduc,
            speedup_vs_brute_force=speedup,
        )

    # Cross-verification of contacts
    # Check that every method found the exact same contact pairs
    first_method = next(iter(contacts_by_method))
    ref_contacts_set = {(c.tube_a, c.tube_b) for c in contacts_by_method[first_method]}

    for m, clist in contacts_by_method.items():
        m_set = {(c.tube_a, c.tube_b) for c in clist}
        if m_set != ref_contacts_set:
            diff_missing = ref_contacts_set - m_set
            diff_extra = m_set - ref_contacts_set
            raise ValueError(
                f"Mismatch between {first_method} and {m}! "
                f"Missing: {diff_missing}, Extra: {diff_extra}"
            )

    return results


def main() -> None:
    """CLI runner to benchmark contact detection algorithms."""
    parser = argparse.ArgumentParser(
        description="Benchmark spatial contact detection algorithms on CNT networks."
    )
    parser.add_argument("--count", type=int, default=1000, help="Number of CNTs (default: 1000)")
    parser.add_argument("--box-x", type=float, default=10.0, help="Box X size (default: 10.0)")
    parser.add_argument("--box-y", type=float, default=5.0, help="Box Y size (default: 5.0)")
    parser.add_argument("--box-z", type=float, default=5.0, help="Box Z size (default: 5.0)")
    parser.add_argument("--length", type=float, default=1.0, help="CNT length L (default: 1.0)")
    parser.add_argument("--diameter", type=float, default=0.1, help="CNT diameter D (default: 0.1)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Contact threshold (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")

    args = parser.parse_args()

    from percolation.generation.cnt_generator import generate_nanotubes

    box = MaterialBox(size_x=args.box_x, size_y=args.box_y, size_z=args.box_z)
    rng = np.random.default_rng(args.seed)

    print("=" * 80)
    print("  CNT Contact Detection Benchmark - Phase 5 Spatial Optimization")
    print("=" * 80)
    print(f"Nanotube Count (N)  : {args.count}")
    print(f"Domain Box          : {box.size_x} x {box.size_y} x {box.size_z} um")
    print(f"CNT Geometry        : L={args.length} um, D={args.diameter} um")
    print(f"Contact Threshold   : {args.threshold} um")
    total_pairs = (args.count * (args.count - 1)) // 2
    print(f"Total O(N^2) Pairs  : {total_pairs:,}")
    print("-" * 80)

    tubes = generate_nanotubes(args.count, box, args.length, args.diameter, rng)

    results = compare_contact_methods(tubes, args.threshold, box=box)

    print(
        f"{'Method':<14} | {'Time (ms)':>10} | {'Candidates':>12} | "
        f"{'Reduction':>10} | {'Contacts':>9} | {'Speedup':>9}"
    )
    print("-" * 80)

    for m, res in results.items():
        t_ms = res.execution_time_sec * 1000
        print(
            f"{m:<14} | {t_ms:>10.2f} | {res.candidate_pairs_count:>12,} | "
            f"{res.candidate_reduction_pct:>9.1f}% | {res.contacts_found:>9,} | "
            f"{res.speedup_vs_brute_force:>8.1f}x"
        )
    print("=" * 80)
    print("All methods verified: 100% identical contacts identified!\n")


if __name__ == "__main__":
    main()

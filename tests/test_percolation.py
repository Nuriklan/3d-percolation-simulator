"""
Integration test: full pipeline for a single sample.

Note on scaling
---------------
With the default config (box 10×5×5 μm, L=1 μm, D=0.01 μm), even φ=0.005
produces ~16,000 CNTs, making O(N²) contact detection impractical for tests.
These tests therefore use a *small* box (2×1×1 μm, L=0.5 μm, D=0.05 μm)
so that realistic concentrations give N ≈ 10–100 tubes, which runs in
milliseconds under the brute-force algorithm.

Phase 5 (spatial optimisation) will make the full-scale config feasible.
"""

import numpy as np
import pytest

from percolation.domain.cnt import MaterialBox
from percolation.simulation.experiment import run_sample

# Small test domain for fast unit tests (runs in milliseconds under O(N²))
TEST_BOX = MaterialBox(size_x=1.0, size_y=0.5, size_z=0.5)
CNT_L = 0.3
CNT_D = 0.05
THRESHOLD = 0.08


def test_run_sample_returns_result():
    """Pipeline runs and returns a valid SimulationResult."""
    rng = np.random.default_rng(42)
    result = run_sample(
        concentration=0.05,
        box=TEST_BOX,
        cnt_length=CNT_L,
        cnt_diameter=CNT_D,
        contact_threshold=THRESHOLD,
        rng=rng,
    )
    assert result.nanotube_count > 0
    assert isinstance(result.percolates, bool)
    assert result.contact_count >= 0


def test_high_concentration_likely_percolates():
    """At high φ the majority of samples must percolate."""
    successes = 0
    n_trials = 5
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        result = run_sample(
            concentration=0.25,      # gives ~100 tubes in TEST_BOX
            box=TEST_BOX,
            cnt_length=CNT_L,
            cnt_diameter=CNT_D,
            contact_threshold=THRESHOLD,
            rng=rng,
        )
        if result.percolates:
            successes += 1
    assert successes >= 3, f"Only {successes}/{n_trials} samples percolated at high φ"


def test_very_low_concentration_does_not_percolate():
    """At φ≈0 (N=1) a single short tube cannot span the box."""
    rng = np.random.default_rng(0)
    result = run_sample(
        concentration=1e-6,          # → N=1
        box=TEST_BOX,
        cnt_length=CNT_L,
        cnt_diameter=CNT_D,
        contact_threshold=0.001,
        rng=rng,
    )
    assert result.percolates is False

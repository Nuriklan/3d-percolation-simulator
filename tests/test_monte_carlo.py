"""
Unit and integration tests for Phase 2: Monte Carlo simulation.
"""

from pathlib import Path
import json
import csv
import pytest
import yaml

from percolation.domain.cnt import MaterialBox
from percolation.simulation.monte_carlo import (
    run_concentration,
    run_monte_carlo,
    run_from_config,
)

# Fast domain for unit testing
TEST_BOX = MaterialBox(size_x=1.0, size_y=0.5, size_z=0.5)
CNT_L = 0.3
CNT_D = 0.05
THRESHOLD = 0.08


def test_run_concentration_basic():
    """run_concentration returns valid statistics."""
    samples = 5
    res = run_concentration(
        concentration=0.05,
        samples=samples,
        box=TEST_BOX,
        cnt_length=CNT_L,
        cnt_diameter=CNT_D,
        contact_threshold=THRESHOLD,
        seed_sequence=42,
        record_samples=True,
    )

    assert res.concentration == 0.05
    assert res.samples == samples
    assert 0 <= res.percolating_samples <= samples
    assert 0.0 <= res.probability <= 1.0
    assert res.probability == res.percolating_samples / samples
    assert res.avg_contact_count >= 0.0
    assert res.avg_nanotube_count > 0.0
    assert len(res.sample_results) == samples


def test_run_concentration_invalid_samples():
    """Invalid sample count raises ValueError."""
    with pytest.raises(ValueError, match="Number of samples must be positive"):
        run_concentration(
            concentration=0.01,
            samples=0,
            box=TEST_BOX,
            cnt_length=CNT_L,
            cnt_diameter=CNT_D,
            contact_threshold=THRESHOLD,
            seed_sequence=42,
        )


def test_run_concentration_reproducibility():
    """Identical seeds produce identical Monte Carlo statistics."""
    res1 = run_concentration(
        concentration=0.1,
        samples=8,
        box=TEST_BOX,
        cnt_length=CNT_L,
        cnt_diameter=CNT_D,
        contact_threshold=THRESHOLD,
        seed_sequence=12345,
    )
    res2 = run_concentration(
        concentration=0.1,
        samples=8,
        box=TEST_BOX,
        cnt_length=CNT_L,
        cnt_diameter=CNT_D,
        contact_threshold=THRESHOLD,
        seed_sequence=12345,
    )

    assert res1.percolating_samples == res2.percolating_samples
    assert res1.probability == res2.probability
    assert res1.avg_contact_count == res2.avg_contact_count
    assert res1.avg_nanotube_count == res2.avg_nanotube_count


def test_run_monte_carlo_empty_concentrations():
    """Empty concentration list raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        run_monte_carlo(
            concentrations=[],
            samples_per_concentration=5,
            box=TEST_BOX,
            cnt_length=CNT_L,
            cnt_diameter=CNT_D,
            contact_threshold=THRESHOLD,
            random_seed=42,
        )


def test_run_monte_carlo_transition_trend():
    """Percolation probability increases from near 0 to near 1 across wide concentration range."""
    concs = [0.001, 0.30]
    mc_result = run_monte_carlo(
        concentrations=concs,
        samples_per_concentration=6,
        box=TEST_BOX,
        cnt_length=CNT_L,
        cnt_diameter=CNT_D,
        contact_threshold=THRESHOLD,
        random_seed=100,
    )

    assert len(mc_result.concentration_results) == 2
    assert mc_result.concentrations == concs
    p_low = mc_result.probabilities[0]
    p_high = mc_result.probabilities[1]

    assert p_low == 0.0
    assert p_high >= 0.5


def test_monte_carlo_save_json_and_csv(tmp_path: Path):
    """MonteCarloResult saves correctly to JSON and CSV formats."""
    concs = [0.01, 0.05]
    mc_result = run_monte_carlo(
        concentrations=concs,
        samples_per_concentration=3,
        box=TEST_BOX,
        cnt_length=CNT_L,
        cnt_diameter=CNT_D,
        contact_threshold=THRESHOLD,
        random_seed=7,
    )

    json_file = tmp_path / "test_mc.json"
    csv_file = tmp_path / "test_mc.csv"

    mc_result.save_json(json_file)
    mc_result.save_csv(csv_file)

    assert json_file.exists()
    assert csv_file.exists()

    # Verify JSON content
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "parameters" in data
    assert data["parameters"]["random_seed"] == 7
    assert len(data["results"]) == 2
    assert data["results"][0]["concentration"] == 0.01

    # Verify CSV content
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0] == [
        "concentration",
        "samples",
        "percolating_samples",
        "probability",
        "avg_contact_count",
        "avg_nanotube_count",
    ]
    assert len(reader) == 3  # Header + 2 data rows


def test_run_from_config(tmp_path: Path):
    """run_from_config successfully parses YAML and executes Monte Carlo."""
    cfg = {
        "box": {"x": 1.0, "y": 0.5, "z": 0.5},
        "cnt": {"length": 0.3, "diameter": 0.05},
        "contact": {"threshold": 0.08},
        "simulation": {
            "concentrations": [0.01, 0.05],
            "samples_per_concentration": 4,
            "random_seed": 99,
        },
    }
    cfg_file = tmp_path / "temp_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    result = run_from_config(cfg_file, verbose=False)
    assert len(result.concentration_results) == 2
    assert result.random_seed == 99
    assert result.box.size_x == 1.0


def test_run_from_config_missing_file():
    """run_from_config raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        run_from_config("non_existent_config.yaml", verbose=False)

"""
Domain model: core data structures.

Physical assumptions documented here:
- CNT modelled as a 3D line segment (capsule) with fixed length and diameter.
- MaterialBox is a rectangular parallelepiped.  Current flows along X axis.
- A Contact exists when the minimum distance between two CNT segments ≤ threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Primitive: 3-component vector
# ---------------------------------------------------------------------------

Vector3D = np.ndarray  # shape (3,), dtype float64


# ---------------------------------------------------------------------------
# CNT (Carbon Nanotube) — line segment in 3D
# ---------------------------------------------------------------------------

@dataclass
class Nanotube:
    """Single CNT represented as a line segment.

    Physical model:
        centre + direction * t,  t ∈ [-length/2, +length/2]

    Attributes
    ----------
    center    : midpoint of the tube, μm
    direction : unit vector along tube axis
    length    : tube length L, μm
    diameter  : tube diameter D, μm
    """

    center: Vector3D
    direction: Vector3D
    length: float
    diameter: float

    # Derived endpoints (cached on first access via property)
    @property
    def p0(self) -> Vector3D:
        """Start endpoint."""
        return self.center - 0.5 * self.length * self.direction

    @property
    def p1(self) -> Vector3D:
        """End endpoint."""
        return self.center + 0.5 * self.length * self.direction

    def __post_init__(self) -> None:
        """Validate and normalise direction vector."""
        self.center = np.asarray(self.center, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm < 1e-12:
            raise ValueError("Nanotube direction vector must be non-zero.")
        self.direction = self.direction / norm


# ---------------------------------------------------------------------------
# Material box — the simulation domain
# ---------------------------------------------------------------------------

@dataclass
class MaterialBox:
    """Rectangular 3D domain.

    Current transport direction: X  (LEFT electrode at x=0, RIGHT at x=size_x).

    Attributes
    ----------
    size_x, size_y, size_z : domain dimensions, μm
    """

    size_x: float
    size_y: float
    size_z: float

    def volume(self) -> float:
        return self.size_x * self.size_y * self.size_z

    def contains_segment(self, p0: Vector3D, p1: Vector3D) -> bool:
        """Return True if both endpoints lie strictly inside the box."""
        for p in (p0, p1):
            if not (
                0.0 <= p[0] <= self.size_x
                and 0.0 <= p[1] <= self.size_y
                and 0.0 <= p[2] <= self.size_z
            ):
                return False
        return True


# ---------------------------------------------------------------------------
# Contact — an electrical junction between two tubes
# ---------------------------------------------------------------------------

@dataclass
class Contact:
    """Electrical contact between two CNTs.

    Attributes
    ----------
    tube_a, tube_b : indices into the nanotube list
    distance       : minimum surface-to-surface distance, μm
                     (currently approximated as segment-to-segment distance)
    """

    tube_a: int
    tube_b: int
    distance: float


# ---------------------------------------------------------------------------
# Simulation result for a single sample
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Outcome of a single Monte Carlo sample.

    Attributes
    ----------
    concentration    : volumetric CNT fraction φ
    nanotube_count   : number of CNTs placed
    percolates       : True if a LEFT→RIGHT conductive path exists
    contact_count    : number of CNT–CNT contacts found
    """

    concentration: float
    nanotube_count: int
    percolates: bool
    contact_count: int = field(default=0)


# ---------------------------------------------------------------------------
# Monte Carlo results
# ---------------------------------------------------------------------------

@dataclass
class ConcentrationResult:
    """Aggregated outcome of Monte Carlo samples for a single concentration.

    Attributes
    ----------
    concentration       : volumetric CNT fraction φ
    samples             : total number of Monte Carlo realizations
    percolating_samples : number of realizations that percolated
    probability         : percolation probability P(φ) = percolating_samples / samples
    avg_contact_count   : mean number of contacts across all samples
    avg_nanotube_count  : mean number of nanotubes across all samples
    sample_results      : optional list of individual SimulationResult objects
    """

    concentration: float
    samples: int
    percolating_samples: int
    probability: float
    avg_contact_count: float = 0.0
    avg_nanotube_count: float = 0.0
    sample_results: list[SimulationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = {
            "concentration": self.concentration,
            "samples": self.samples,
            "percolating_samples": self.percolating_samples,
            "probability": self.probability,
            "avg_contact_count": self.avg_contact_count,
            "avg_nanotube_count": self.avg_nanotube_count,
        }
        if self.sample_results:
            data["sample_results"] = [
                {
                    "concentration": s.concentration,
                    "nanotube_count": s.nanotube_count,
                    "percolates": s.percolates,
                    "contact_count": s.contact_count,
                }
                for s in self.sample_results
            ]
        return data


@dataclass
class MonteCarloResult:
    """Complete results for a multi-concentration Monte Carlo study.

    Attributes
    ----------
    box                   : simulation domain Box
    cnt_length            : CNT length L, μm
    cnt_diameter          : CNT diameter D, μm
    contact_threshold     : contact distance threshold, μm
    random_seed           : base random seed
    concentration_results : list of ConcentrationResult for each tested concentration
    """

    box: MaterialBox
    cnt_length: float
    cnt_diameter: float
    contact_threshold: float
    random_seed: int
    concentration_results: list[ConcentrationResult] = field(default_factory=list)

    @property
    def concentrations(self) -> list[float]:
        return [r.concentration for r in self.concentration_results]

    @property
    def probabilities(self) -> list[float]:
        return [r.probability for r in self.concentration_results]

    def estimate_threshold(
        self,
        confidence_level: float = 0.95,
        target_probability: float = 0.5,
        prefer_method: str = "logistic",
    ):
        """Estimate percolation threshold from this Monte Carlo dataset."""
        from percolation.analysis.threshold import estimate_threshold

        return estimate_threshold(
            self,
            confidence_level=confidence_level,
            target_probability=target_probability,
            prefer_method=prefer_method,
        )

    def to_dict(self) -> dict:
        return {
            "parameters": {
                "box": {
                    "size_x": self.box.size_x,
                    "size_y": self.box.size_y,
                    "size_z": self.box.size_z,
                },
                "cnt": {
                    "length": self.cnt_length,
                    "diameter": self.cnt_diameter,
                    "aspect_ratio": self.cnt_length / self.cnt_diameter if self.cnt_diameter > 0 else 0.0,
                },
                "contact_threshold": self.contact_threshold,
                "random_seed": self.random_seed,
            },
            "results": [r.to_dict() for r in self.concentration_results],
        }

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save results to a JSON file."""
        import json
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def save_csv(self, path: str | Path) -> None:
        """Save P(φ) table to a CSV file."""
        import csv
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "concentration",
                "samples",
                "percolating_samples",
                "probability",
                "avg_contact_count",
                "avg_nanotube_count",
            ])
            for r in self.concentration_results:
                writer.writerow([
                    r.concentration,
                    r.samples,
                    r.percolating_samples,
                    r.probability,
                    f"{r.avg_contact_count:.2f}",
                    f"{r.avg_nanotube_count:.2f}",
                ])

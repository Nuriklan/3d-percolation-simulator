"""
Domain model: core data structures.

Physical assumptions documented here:
- CNT modelled as a 3D line segment (capsule) with fixed length and diameter.
- MaterialBox is a rectangular parallelepiped. Current flows along X axis.
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

# Valid orientation distribution modes (Phase 7)
VALID_ORIENTATION_MODES = ("isotropic", "aligned", "partially_aligned", "fisher")

_AXIS_ALIASES: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def _normalize_axis(axis: str | tuple[float, float, float] | list[float] | np.ndarray) -> tuple[float, float, float]:
    """Resolve an axis alias ('x', 'y', 'z') or 3D vector to a unit tuple."""
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key not in _AXIS_ALIASES:
            raise ValueError(f"Unknown axis alias {axis!r}. Expected one of 'x', 'y', 'z' or a 3D vector.")
        return _AXIS_ALIASES[key]

    arr = np.asarray(axis, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Orientation axis must be a 3-element vector, got shape {arr.shape}.")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("Orientation axis vector must be non-zero.")
    unit = arr / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _axis_short_name(axis: tuple[float, float, float]) -> str:
    """Return a concise axis name ('X', 'Y', 'Z', or formatted vector)."""
    vec = np.asarray(axis, dtype=float)
    for name, ref in (("X", (1.0, 0.0, 0.0)), ("Y", (0.0, 1.0, 0.0)), ("Z", (0.0, 0.0, 1.0))):
        if abs(float(np.dot(vec, ref))) > 1.0 - 1e-6:
            return name
    return f"({axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f})"


@dataclass
class OrientationConfig:
    """Configuration for CNT 3D orientation distribution (Phase 7).

    Supported modes:
    - ``isotropic``: Uniform random directions on the unit sphere S^2 (S = 0).
    - ``aligned``: All CNTs strictly parallel to ``axis`` (theta_max = 0 deg, S_axis = 1).
    - ``partially_aligned``: Uniform solid-angle distribution inside a cone of
      half-angle ``max_angle_deg`` in [0, 90] around ``axis``.
      Theoretical Hermans orientation factor along ``axis``:
          S_axis = cos(theta_max) * (1 + cos(theta_max)) / 2
    - ``fisher``: von Mises-Fisher distribution around ``axis`` with
      concentration parameter ``kappa >= 0`` (kappa = 0 -> isotropic;
      kappa -> inf -> aligned).

    Attributes
    ----------
    mode          : orientation distribution mode
    axis          : preferred alignment axis (unit vector or 'x'/'y'/'z')
    max_angle_deg : cone half-angle in degrees [0, 90] for 'partially_aligned'
    kappa         : concentration parameter >= 0 for 'fisher'
    """

    mode: str = "isotropic"
    axis: tuple[float, float, float] | str = (1.0, 0.0, 0.0)
    max_angle_deg: float = 90.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode == "cone":
            self.mode = "partially_aligned"
        if self.mode not in VALID_ORIENTATION_MODES:
            raise ValueError(
                f"Invalid orientation mode {self.mode!r}. "
                f"Expected one of {VALID_ORIENTATION_MODES}."
            )

        self.axis = _normalize_axis(self.axis)
        self.max_angle_deg = float(self.max_angle_deg)
        self.kappa = float(self.kappa)

        if self.mode == "isotropic":
            self.max_angle_deg = 90.0
            self.kappa = 0.0
        elif self.mode == "aligned":
            self.max_angle_deg = 0.0
        elif self.mode == "partially_aligned":
            if not (0.0 <= self.max_angle_deg <= 90.0):
                raise ValueError(
                    f"max_angle_deg must be in [0, 90], got {self.max_angle_deg}."
                )
        elif self.mode == "fisher":
            if self.kappa < 0.0 or not np.isfinite(self.kappa):
                raise ValueError(f"kappa must be finite and >= 0, got {self.kappa}.")

    @property
    def axis_vector(self) -> Vector3D:
        """Unit axis vector as a NumPy ndarray of shape (3,)."""
        return np.asarray(self.axis, dtype=float)

    @property
    def label(self) -> str:
        """Human-readable ASCII-safe label describing this orientation."""
        ax_name = _axis_short_name(self.axis)
        if self.mode == "isotropic":
            return "Isotropic"
        if self.mode == "aligned":
            return f"Aligned ({ax_name})"
        if self.mode == "partially_aligned":
            if self.max_angle_deg <= 1e-9:
                return f"Aligned ({ax_name})"
            if self.max_angle_deg >= 90.0 - 1e-9:
                return "Isotropic (90 deg)"
            return f"Cone {self.max_angle_deg:g} deg ({ax_name})"
        if self.mode == "fisher":
            return f"Fisher kappa={self.kappa:g} ({ax_name})"
        return self.mode

    def theoretical_order_parameter(
        self,
        reference_axis: tuple[float, float, float] | str = (1.0, 0.0, 0.0),
    ) -> float:
        """Return analytical 3D Hermans orientation factor S along ``reference_axis``.

        Defined as:
            S_ref = (3 * <cos^2(theta_ref)> - 1) / 2  in [-0.5, 1.0]

        By the Legendre addition theorem for azimuthally symmetric distributions
        around ``self.axis``:
            S_ref = S_axis * (3 * (a . r)^2 - 1) / 2
        """
        if self.mode == "isotropic":
            s_axis = 0.0
        elif self.mode == "aligned":
            s_axis = 1.0
        elif self.mode == "partially_aligned":
            cos_max = float(np.cos(np.deg2rad(self.max_angle_deg)))
            s_axis = 0.5 * cos_max * (1.0 + cos_max)
        elif self.mode == "fisher":
            if self.kappa < 1e-6:
                # Series expansion: S = kappa^2 / 15 - 2 * kappa^4 / 315 + ...
                s_axis = (self.kappa ** 2) / 15.0
            elif self.kappa > 500.0:
                # Asymptotic coth(k) -> 1: <cos^2> = 1 - 2/k + 2/k^2
                mean_cos2 = 1.0 - 2.0 / self.kappa + 2.0 / (self.kappa ** 2)
                s_axis = 0.5 * (3.0 * mean_cos2 - 1.0)
            else:
                coth_k = float(1.0 / np.tanh(self.kappa))
                mean_cos2 = 1.0 - (2.0 / self.kappa) * coth_k + 2.0 / (self.kappa ** 2)
                s_axis = 0.5 * (3.0 * mean_cos2 - 1.0)
        else:
            s_axis = 0.0

        ref = np.asarray(_normalize_axis(reference_axis), dtype=float)
        dot = float(np.dot(self.axis_vector, ref))
        p2 = 0.5 * (3.0 * dot * dot - 1.0)
        return float(s_axis * p2)

    def to_dict(self) -> dict:
        """Serialize orientation configuration to a plain dictionary."""
        return {
            "mode": self.mode,
            "axis": list(self.axis),
            "max_angle_deg": self.max_angle_deg,
            "kappa": self.kappa,
            "label": self.label,
            "theoretical_order_parameter_x": self.theoretical_order_parameter("x"),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> OrientationConfig:
        """Deserialize OrientationConfig from a dictionary (or None -> isotropic)."""
        if not data:
            return cls.isotropic()
        return cls(
            mode=data.get("mode", "isotropic"),
            axis=data.get("axis", (1.0, 0.0, 0.0)),
            max_angle_deg=float(data.get("max_angle_deg", 90.0)),
            kappa=float(data.get("kappa", 0.0)),
        )

    @classmethod
    def isotropic(cls) -> OrientationConfig:
        """Uniform isotropic orientation on S^2."""
        return cls(mode="isotropic")

    @classmethod
    def aligned(cls, axis: tuple[float, float, float] | str = "x") -> OrientationConfig:
        """Strictly aligned CNTs along ``axis``."""
        return cls(mode="aligned", axis=axis, max_angle_deg=0.0)

    @classmethod
    def partially_aligned(
        cls,
        max_angle_deg: float = 30.0,
        axis: tuple[float, float, float] | str = "x",
    ) -> OrientationConfig:
        """Partially aligned CNTs inside a cone of half-angle ``max_angle_deg``."""
        return cls(mode="partially_aligned", axis=axis, max_angle_deg=max_angle_deg)

    @classmethod
    def fisher(
        cls,
        kappa: float = 5.0,
        axis: tuple[float, float, float] | str = "x",
    ) -> OrientationConfig:
        """von Mises-Fisher orientation distribution around ``axis``."""
        return cls(mode="fisher", axis=axis, kappa=kappa)


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
# Full sample state for inspection and 3D visualization (Phase 4)
# ---------------------------------------------------------------------------

@dataclass
class SampleState:
    """Complete geometrical and topological state of a single CNT sample.

    Attributes
    ----------
    box                 : material domain box
    tubes               : list of Nanotubes placed inside the box
    contacts            : list of pairwise Contact instances
    graph               : adjacency list representation of contact network
    percolates          : True if LEFT to RIGHT path exists
    percolating_cluster : set of nanotube indices in percolating network
    percolating_path    : sequence of nodes from LEFT to RIGHT, or None
    concentration       : volumetric CNT fraction φ
    cnt_length          : length L of each nanotube, μm
    cnt_diameter        : diameter D of each nanotube, μm
    contact_threshold   : distance threshold for contact, μm
    """

    box: MaterialBox
    tubes: list[Nanotube]
    contacts: list[Contact]
    graph: dict[int | str, list[int | str]]
    percolates: bool
    percolating_cluster: set[int] = field(default_factory=set)
    percolating_path: list[int | str] | None = None
    concentration: float = 0.0
    cnt_length: float = 0.0
    cnt_diameter: float = 0.0
    contact_threshold: float = 0.0
    orientation: OrientationConfig = field(default_factory=OrientationConfig.isotropic)

    @property
    def nanotube_count(self) -> int:
        return len(self.tubes)

    @property
    def contact_count(self) -> int:
        return len(self.contacts)

    def to_simulation_result(self) -> SimulationResult:
        return SimulationResult(
            concentration=self.concentration,
            nanotube_count=self.nanotube_count,
            percolates=self.percolates,
            contact_count=self.contact_count,
        )


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
    orientation: OrientationConfig = field(default_factory=OrientationConfig.isotropic)

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
                    "orientation": self.orientation.to_dict(),
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

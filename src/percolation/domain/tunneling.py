"""
Tunneling contact conductance model — Phase 8 domain layer.

Physical model
--------------
Quantum mechanical electron tunneling across an insulating polymer gap between
two Carbon Nanotubes (CNTs) or between a CNT and a metallic electrode is
modeled by an exponential decay law (Plan Sections 8.2 & 21):

    G_ij = G_0 * exp(-alpha * d_eff)
    R_ij = R_0 * exp(+alpha * d_eff) = 1 / G_ij

where:
- G_0 = 1 / R_0 is the direct-contact conductance (at d_eff = 0), in Siemens (S)
  or normalized dimensionless conductance units;
- alpha >= 0 is the tunneling decay constant, in inverse simulation length units
  (e.g. 1/um when lengths are in um);
- d_eff = max(0, d - d_min) is the effective tunneling barrier thickness beyond
  the minimum direct-contact / van der Waals distance d_min >= 0.

Optional Simmons (1963) parameterization
----------------------------------------
For physically grounded polymer barriers of height lambda (in eV):

    alpha_nm = (4 * pi / h) * sqrt(2 * m_e * e * lambda) * 1e-9  [1/nm]
             ~ 10.2463 * sqrt(lambda [eV])  [1/nm]

and Landauer quantum contact conductance:

    G_0 = 2 * e^2 / h ~ 7.748091729e-5 S  (R_0 ~ 12.906 kOhm).

Physical assumptions documented:
- Gap distance d is approximated by the minimum 3D line-segment distance between
  CNT axes (or segment-to-plane distance for electrodes), consistent with Phases 1-7.
- Contacts beyond ``effective_cutoff`` (tightest bound from contact threshold,
  ``cutoff_distance``, and optional ``min_conductance``) are omitted from the
  contact network.
- Intrinsic CNT resistance along tube segments is deferred to the full resistor
  network stage (Phase 9); Phase 8 focuses on junction (contact) tunneling
  conductance and resistance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from percolation.domain.cnt import Contact

# Fundamental physical constants (SI, CODATA 2018 exact definitions)
ELECTRON_MASS_KG: float = 9.1093837015e-31
PLANCK_H_JS: float = 6.62607015e-34
ELEMENTARY_CHARGE_C: float = 1.602176634e-19
LANDAUER_CONDUCTANCE_S: float = 2.0 * (ELEMENTARY_CHARGE_C ** 2) / PLANCK_H_JS  # ~7.7481e-5 S
LANDAUER_RESISTANCE_OHM: float = 1.0 / LANDAUER_CONDUCTANCE_S  # ~12906.4 Ohm

# Safe exponent limit to prevent float64 overflow in exp(+x)
_MAX_SAFE_EXP: float = 700.0


def simmons_decay_constant_nm(barrier_ev: float) -> float:
    """Compute Simmons tunneling decay constant alpha in 1/nm for barrier height (eV).

    Formula:
        alpha = (4 * pi / h) * sqrt(2 * m_e * e * barrier_ev) * 1e-9

    For a typical polymer barrier lambda = 1.0 eV, alpha ~ 10.2463 nm^-1.
    """
    barrier_ev = float(barrier_ev)
    if barrier_ev <= 0.0 or not math.isfinite(barrier_ev):
        raise ValueError(f"barrier_ev must be finite and > 0, got {barrier_ev}.")
    alpha_per_m = (4.0 * math.pi / PLANCK_H_JS) * math.sqrt(
        2.0 * ELECTRON_MASS_KG * ELEMENTARY_CHARGE_C * barrier_ev
    )
    return alpha_per_m * 1e-9


@dataclass
class TunnelingConfig:
    """Configuration for exponential tunneling contact conductance (Phase 8).

    Conductance & resistance laws:
        d_eff = max(0.0, d - d_min)
        G(d)  = g0 * exp(-alpha * d_eff)
        R(d)  = (1 / g0) * exp(+alpha * d_eff)

    Attributes
    ----------
    enabled         : whether tunneling conductance model is active
    g0              : direct-contact conductance G_0 at d <= d_min (> 0), S
    alpha           : exponential tunneling decay constant (>= 0), 1/length_unit
    d_min           : direct-contact / van der Waals clamp distance (>= 0), length_unit
    cutoff_distance : optional maximum tunneling distance overriding contact threshold
    min_conductance : optional minimum conductance G_min (> 0) for active contacts
    """

    enabled: bool = True
    g0: float = 1.0
    alpha: float = 50.0
    d_min: float = 0.0
    cutoff_distance: float | None = None
    min_conductance: float | None = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.g0 = float(self.g0)
        self.alpha = float(self.alpha)
        self.d_min = float(self.d_min)

        if self.g0 <= 0.0 or not math.isfinite(self.g0):
            raise ValueError(f"g0 must be finite and > 0, got {self.g0}.")
        if self.alpha < 0.0 or not math.isfinite(self.alpha):
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha}.")
        if self.d_min < 0.0 or not math.isfinite(self.d_min):
            raise ValueError(f"d_min must be finite and >= 0, got {self.d_min}.")

        if self.cutoff_distance is not None:
            self.cutoff_distance = float(self.cutoff_distance)
            if self.cutoff_distance < 0.0 or not math.isfinite(self.cutoff_distance):
                raise ValueError(
                    f"cutoff_distance must be finite and >= 0, got {self.cutoff_distance}."
                )

        if self.min_conductance is not None:
            self.min_conductance = float(self.min_conductance)
            if self.min_conductance <= 0.0 or not math.isfinite(self.min_conductance):
                raise ValueError(
                    f"min_conductance must be finite and > 0, got {self.min_conductance}."
                )

    @property
    def r0(self) -> float:
        """Base direct-contact resistance R_0 = 1 / G_0."""
        return 1.0 / self.g0

    @property
    def label(self) -> str:
        """Human-readable ASCII-safe label describing this tunneling configuration."""
        if not self.enabled:
            return "Binary (no tunneling)"
        parts = [f"alpha={self.alpha:g}", f"G0={self.g0:g}"]
        if self.d_min > 0.0:
            parts.append(f"d_min={self.d_min:g}")
        if self.min_conductance is not None:
            parts.append(f"G_min={self.min_conductance:.2e}")
        return f"Tunneling ({', '.join(parts)})"

    def effective_distance(self, distance: float) -> float:
        """Return effective tunneling gap d_eff = max(0, d - d_min)."""
        d = float(distance)
        if d < 0.0:
            raise ValueError(f"Contact distance must be >= 0, got {d}.")
        return max(0.0, d - self.d_min)

    def conductance(self, distance: float) -> float:
        """Evaluate tunneling conductance G(d) = G_0 * exp(-alpha * max(0, d - d_min))."""
        d_eff = self.effective_distance(distance)
        exponent = self.alpha * d_eff
        if exponent >= _MAX_SAFE_EXP:
            return 0.0
        return self.g0 * math.exp(-exponent)

    def resistance(self, distance: float) -> float:
        """Evaluate tunneling resistance R(d) = (1 / G_0) * exp(+alpha * max(0, d - d_min))."""
        d_eff = self.effective_distance(distance)
        exponent = self.alpha * d_eff
        if exponent >= _MAX_SAFE_EXP:
            return float("inf")
        return self.r0 * math.exp(exponent)

    def conductance_array(self, distances: np.ndarray | Sequence[float]) -> np.ndarray:
        """Vectorized evaluation of tunneling conductance G(d)."""
        d = np.asarray(distances, dtype=np.float64)
        if d.size == 0:
            return np.zeros(0, dtype=np.float64)
        if np.any(d < 0.0):
            raise ValueError("Contact distances must be non-negative.")
        d_eff = np.maximum(0.0, d - self.d_min)
        exponent = self.alpha * d_eff
        return np.where(
            exponent < _MAX_SAFE_EXP,
            self.g0 * np.exp(-np.minimum(exponent, _MAX_SAFE_EXP)),
            0.0,
        )

    def resistance_array(self, distances: np.ndarray | Sequence[float]) -> np.ndarray:
        """Vectorized evaluation of tunneling resistance R(d)."""
        d = np.asarray(distances, dtype=np.float64)
        if d.size == 0:
            return np.zeros(0, dtype=np.float64)
        if np.any(d < 0.0):
            raise ValueError("Contact distances must be non-negative.")
        d_eff = np.maximum(0.0, d - self.d_min)
        exponent = self.alpha * d_eff
        return np.where(
            exponent < _MAX_SAFE_EXP,
            self.r0 * np.exp(np.minimum(exponent, _MAX_SAFE_EXP)),
            np.inf,
        )

    def max_distance_for_min_conductance(self) -> float | None:
        """Maximum distance d satisfying G(d) >= min_conductance, or None if unbounded.

        Returns -1.0 if min_conductance > g0 (no distance can satisfy G(d) >= G_min).
        Returns +inf if alpha == 0 and min_conductance <= g0.
        """
        if self.min_conductance is None:
            return None
        if self.min_conductance > self.g0:
            return -1.0
        if self.alpha == 0.0:
            return float("inf")
        return self.d_min + math.log(self.g0 / self.min_conductance) / self.alpha

    def effective_cutoff(self, contact_threshold: float) -> float:
        """Tightest distance upper bound combining threshold, cutoff_distance, and min_conductance.

        Returns a negative value (-1.0) if ``min_conductance > g0`` so that no
        non-negative contact distance passes ``d <= effective_cutoff``.
        """
        d_cut = (
            self.cutoff_distance
            if self.cutoff_distance is not None
            else float(contact_threshold)
        )
        if not self.enabled:
            return float(contact_threshold)

        d_gmin = self.max_distance_for_min_conductance()
        if d_gmin is not None:
            if d_gmin < 0.0:
                return -1.0
            d_cut = min(d_cut, d_gmin)
        return float(d_cut)

    def is_contact_active(self, distance: float, contact_threshold: float) -> bool:
        """Return True if a contact at *distance* passes tunneling cutoffs."""
        d = float(distance)
        if d < 0.0:
            return False
        d_cut = self.effective_cutoff(contact_threshold)
        if d_cut < 0.0 or d > d_cut + 1e-15:
            return False
        if self.enabled and self.min_conductance is not None:
            return self.conductance(d) >= self.min_conductance * (1.0 - 1e-12)
        return True

    def to_dict(self) -> dict:
        """Serialize tunneling configuration to a plain dictionary."""
        return {
            "enabled": self.enabled,
            "g0": self.g0,
            "r0": self.r0,
            "alpha": self.alpha,
            "d_min": self.d_min,
            "cutoff_distance": self.cutoff_distance,
            "min_conductance": self.min_conductance,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> TunnelingConfig:
        """Deserialize TunnelingConfig from a dictionary (None -> disabled)."""
        if not data:
            return cls.disabled()
        g0_val = data.get("g0")
        r0_val = data.get("r0")
        if g0_val is None and r0_val is not None:
            r0_f = float(r0_val)
            if r0_f <= 0.0:
                raise ValueError(f"r0 must be > 0, got {r0_f}.")
            g0_val = 1.0 / r0_f
        elif g0_val is None:
            g0_val = 1.0

        cutoff_raw = data.get("cutoff_distance")
        min_g_raw = data.get("min_conductance")
        return cls(
            enabled=bool(data.get("enabled", True)),
            g0=float(g0_val),
            alpha=float(data.get("alpha", 50.0)),
            d_min=float(data.get("d_min", 0.0)),
            cutoff_distance=float(cutoff_raw) if cutoff_raw is not None else None,
            min_conductance=float(min_g_raw) if min_g_raw is not None else None,
        )

    @classmethod
    def disabled(cls) -> TunnelingConfig:
        """Return a disabled tunneling configuration (pure binary contact model)."""
        return cls(enabled=False)

    @classmethod
    def from_resistance(
        cls,
        r0: float,
        alpha: float = 50.0,
        d_min: float = 0.0,
        cutoff_distance: float | None = None,
        min_conductance: float | None = None,
    ) -> TunnelingConfig:
        """Construct TunnelingConfig parameterized by direct-contact resistance R_0."""
        r0_f = float(r0)
        if r0_f <= 0.0 or not math.isfinite(r0_f):
            raise ValueError(f"r0 must be finite and > 0, got {r0_f}.")
        return cls(
            enabled=True,
            g0=1.0 / r0_f,
            alpha=alpha,
            d_min=d_min,
            cutoff_distance=cutoff_distance,
            min_conductance=min_conductance,
        )

    @classmethod
    def from_simmons_barrier(
        cls,
        barrier_ev: float = 1.0,
        g0: float = LANDAUER_CONDUCTANCE_S,
        length_unit_nm: float = 1000.0,
        d_min_nm: float = 0.34,
        cutoff_nm: float | None = None,
        min_conductance: float | None = None,
    ) -> TunnelingConfig:
        """Construct physically parameterized TunnelingConfig using Simmons (1963) model.

        Parameters
        ----------
        barrier_ev     : polymer tunneling barrier height lambda in eV (typical 0.5-5.0 eV)
        g0             : direct-contact conductance G_0 in S (default: Landauer 2e^2/h)
        length_unit_nm : nanometers per simulation length unit (1000.0 for um, 1.0 for nm)
        d_min_nm       : van der Waals minimum contact separation in nm (default: 0.34 nm)
        cutoff_nm      : optional tunneling cutoff distance in nm (e.g. 1.4-2.0 nm)
        min_conductance: optional minimum conductance cutoff in S
        """
        if length_unit_nm <= 0.0 or not math.isfinite(length_unit_nm):
            raise ValueError(f"length_unit_nm must be finite and > 0, got {length_unit_nm}.")
        if d_min_nm < 0.0 or not math.isfinite(d_min_nm):
            raise ValueError(f"d_min_nm must be finite and >= 0, got {d_min_nm}.")
        if cutoff_nm is not None and (cutoff_nm < 0.0 or not math.isfinite(cutoff_nm)):
            raise ValueError(f"cutoff_nm must be finite and >= 0, got {cutoff_nm}.")

        alpha_nm = simmons_decay_constant_nm(barrier_ev)
        alpha_sim = alpha_nm * length_unit_nm
        d_min_sim = d_min_nm / length_unit_nm
        cutoff_sim = (cutoff_nm / length_unit_nm) if cutoff_nm is not None else None
        return cls(
            enabled=True,
            g0=g0,
            alpha=alpha_sim,
            d_min=d_min_sim,
            cutoff_distance=cutoff_sim,
            min_conductance=min_conductance,
        )


@dataclass
class ElectrodeContact:
    """Electrical contact between a CNT and a planar electrode (LEFT or RIGHT).

    Attributes
    ----------
    electrode   : 'LEFT' (x = 0) or 'RIGHT' (x = box.size_x)
    tube_idx    : index of the nanotube
    distance    : minimum distance from the tube segment to the electrode plane, um
    conductance : tunneling conductance G_e (S), or None in binary mode
    resistance  : tunneling resistance R_e = 1/G_e (Ohm), or None in binary mode
    """

    electrode: str
    tube_idx: int
    distance: float
    conductance: float | None = None
    resistance: float | None = None

    def to_dict(self) -> dict:
        data: dict = {
            "electrode": self.electrode,
            "tube_idx": self.tube_idx,
            "distance": self.distance,
        }
        if self.conductance is not None:
            data["conductance"] = self.conductance
        if self.resistance is not None:
            data["resistance"] = self.resistance
        return data


@dataclass
class ContactConductanceStats:
    """Summary statistics of contact distances, conductances, and resistances."""

    count: int = 0
    mean_distance: float | None = None
    median_distance: float | None = None
    min_distance: float | None = None
    max_distance: float | None = None
    mean_conductance: float | None = None
    median_conductance: float | None = None
    geom_mean_conductance: float | None = None
    min_conductance: float | None = None
    max_conductance: float | None = None
    total_conductance: float | None = None
    log10_conductance_std: float | None = None
    mean_resistance: float | None = None
    median_resistance: float | None = None
    geom_mean_resistance: float | None = None
    min_resistance: float | None = None
    max_resistance: float | None = None

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_distance": self.mean_distance,
            "median_distance": self.median_distance,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "mean_conductance": self.mean_conductance,
            "median_conductance": self.median_conductance,
            "geom_mean_conductance": self.geom_mean_conductance,
            "min_conductance": self.min_conductance,
            "max_conductance": self.max_conductance,
            "total_conductance": self.total_conductance,
            "log10_conductance_std": self.log10_conductance_std,
            "mean_resistance": self.mean_resistance,
            "median_resistance": self.median_resistance,
            "geom_mean_resistance": self.geom_mean_resistance,
            "min_resistance": self.min_resistance,
            "max_resistance": self.max_resistance,
        }


def compute_contact_conductance(distance: float, config: TunnelingConfig) -> tuple[float, float]:
    """Compute ``(conductance, resistance)`` for a single contact at *distance*."""
    return config.conductance(distance), config.resistance(distance)


def apply_tunneling_to_contacts(
    contacts: Sequence[Contact],
    config: TunnelingConfig,
    contact_threshold: float | None = None,
) -> list[Contact]:
    """Populate conductance and resistance on contacts using *config*.

    If ``config.enabled`` is False, returns shallow copies without conductance/resistance.
    If ``contact_threshold``, ``config.cutoff_distance``, or ``config.min_conductance``
    imposes a stricter distance/conductance bound, non-qualifying contacts are filtered out.
    """
    from percolation.domain.cnt import Contact

    if not contacts:
        return []

    if not config.enabled:
        return [Contact(tube_a=c.tube_a, tube_b=c.tube_b, distance=c.distance) for c in contacts]

    distances = np.array([c.distance for c in contacts], dtype=np.float64)
    conductances = config.conductance_array(distances)
    resistances = config.resistance_array(distances)

    mask = np.ones(len(contacts), dtype=bool)
    if contact_threshold is not None or config.cutoff_distance is not None:
        base_th = (
            float(contact_threshold)
            if contact_threshold is not None
            else float(config.cutoff_distance)
        )
        eff_cut = config.effective_cutoff(base_th)
        if eff_cut < 0.0:
            return []
        mask &= distances <= eff_cut + 1e-15

    if config.min_conductance is not None:
        mask &= conductances >= config.min_conductance * (1.0 - 1e-12)

    enriched: list[Contact] = []
    for idx in np.nonzero(mask)[0]:
        c = contacts[int(idx)]
        enriched.append(
            Contact(
                tube_a=c.tube_a,
                tube_b=c.tube_b,
                distance=float(c.distance),
                conductance=float(conductances[idx]),
                resistance=float(resistances[idx]),
            )
        )
    return enriched


def summarize_contact_conductance(contacts: Sequence[Contact]) -> ContactConductanceStats:
    """Compute distance, conductance, and resistance summary statistics for contacts."""
    if not contacts:
        return ContactConductanceStats(count=0)

    dists = np.array([c.distance for c in contacts], dtype=np.float64)
    stats = ContactConductanceStats(
        count=len(contacts),
        mean_distance=float(np.mean(dists)),
        median_distance=float(np.median(dists)),
        min_distance=float(np.min(dists)),
        max_distance=float(np.max(dists)),
    )

    g_vals = [c.conductance for c in contacts if c.conductance is not None]
    if g_vals:
        g_arr = np.array(g_vals, dtype=np.float64)
        stats.mean_conductance = float(np.mean(g_arr))
        stats.median_conductance = float(np.median(g_arr))
        stats.min_conductance = float(np.min(g_arr))
        stats.max_conductance = float(np.max(g_arr))
        stats.total_conductance = float(np.sum(g_arr))

        pos_g = g_arr[g_arr > 0.0]
        if pos_g.size > 0:
            log_g = np.log(pos_g)
            stats.geom_mean_conductance = float(np.exp(np.mean(log_g)))
            stats.log10_conductance_std = float(np.std(np.log10(pos_g)))
        else:
            stats.geom_mean_conductance = 0.0
            stats.log10_conductance_std = 0.0

    r_vals = [c.resistance for c in contacts if c.resistance is not None]
    if r_vals:
        r_arr = np.array(r_vals, dtype=np.float64)
        finite_r = r_arr[np.isfinite(r_arr)]
        if finite_r.size > 0:
            stats.mean_resistance = float(np.mean(finite_r))
            stats.median_resistance = float(np.median(finite_r))
            stats.min_resistance = float(np.min(finite_r))
            stats.max_resistance = float(np.max(finite_r))
            pos_r = finite_r[finite_r > 0.0]
            if pos_r.size > 0:
                stats.geom_mean_resistance = float(np.exp(np.mean(np.log(pos_r))))
        else:
            stats.mean_resistance = float("inf")
            stats.median_resistance = float("inf")
            stats.min_resistance = float("inf")
            stats.max_resistance = float("inf")
            stats.geom_mean_resistance = float("inf")

    return stats

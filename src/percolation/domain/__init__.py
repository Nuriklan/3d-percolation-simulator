"""Domain models for CNT composite percolation simulation."""

from __future__ import annotations

from percolation.domain.cnt import (
    ConcentrationResult,
    Contact,
    MaterialBox,
    MonteCarloResult,
    Nanotube,
    OrientationConfig,
    SampleState,
    SimulationResult,
    Vector3D,
)
from percolation.domain.tunneling import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    LANDAUER_CONDUCTANCE_S,
    LANDAUER_RESISTANCE_OHM,
    PLANCK_H_JS,
    ContactConductanceStats,
    ElectrodeContact,
    TunnelingConfig,
    apply_tunneling_to_contacts,
    compute_contact_conductance,
    simmons_decay_constant_nm,
    summarize_contact_conductance,
)

__all__ = [
    "ELECTRON_MASS_KG",
    "ELEMENTARY_CHARGE_C",
    "LANDAUER_CONDUCTANCE_S",
    "LANDAUER_RESISTANCE_OHM",
    "PLANCK_H_JS",
    "ConcentrationResult",
    "Contact",
    "ContactConductanceStats",
    "ElectrodeContact",
    "MaterialBox",
    "MonteCarloResult",
    "Nanotube",
    "OrientationConfig",
    "SampleState",
    "SimulationResult",
    "TunnelingConfig",
    "Vector3D",
    "apply_tunneling_to_contacts",
    "compute_contact_conductance",
    "simmons_decay_constant_nm",
    "summarize_contact_conductance",
]

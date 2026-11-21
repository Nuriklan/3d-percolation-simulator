"""
Tunneling Contact Conductance Study — Phase 8 analysis module.

Provides:
1. Single-sample tunneling diagnostics (``analyze_sample_tunneling``):
   - All-contact vs percolating-cluster conductance/resistance statistics.
   - Comparison of BFS shortest-hop path vs Dijkstra minimum-resistance path.
   - Critical-path (Ambegaokar-Halperin-Langer bottleneck) conductance G_c.

2. Parametric tunneling studies:
   - ``run_tunneling_alpha_study``: sweeps tunneling decay parameter alpha at a
     fixed conductance threshold G_min to quantify how barrier height shrinks the
     effective tunneling range d_eff(alpha), shifts phi_c(alpha), and governs
     bottleneck conductance G_c(phi).
   - ``run_contact_cutoff_study`` (Plan Section 25, Experiment 3): sweeps contact
     cutoff distance d_contact at fixed (G_0, alpha) to quantify phi_c(d_contact)
     and tunneling conductance trends.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from percolation.analysis.threshold import ThresholdResult, estimate_threshold
from percolation.domain.cnt import MaterialBox, OrientationConfig, SampleState
from percolation.domain.tunneling import (
    TunnelingConfig,
    summarize_contact_conductance,
)
from percolation.graph.percolation import path_series_resistance
from percolation.simulation.monte_carlo import run_concentration, run_monte_carlo

if TYPE_CHECKING:
    from percolation.domain.cnt import MonteCarloResult


# ---------------------------------------------------------------------------
# Single-sample tunneling diagnostic
# ---------------------------------------------------------------------------

def analyze_sample_tunneling(sample: SampleState) -> dict:
    """Compute comprehensive tunneling conductance diagnostics for a SampleState.

    Returns a dictionary containing:
    - ``percolates``
    - ``tunneling`` (config dict or None)
    - ``all_contacts`` (ContactConductanceStats dict)
    - ``cluster_contacts`` (ContactConductanceStats dict for contacts inside percolating cluster)
    - ``shortest_hop_path_hops`` & ``shortest_hop_path_resistance`` (if percolating & weighted)
    - ``min_resistance_path_hops`` & ``min_path_resistance``
    - ``resistance_improvement_ratio`` (R_bfs / R_dijkstra >= 1.0)
    - ``bottleneck_conductance``
    """
    all_stats = summarize_contact_conductance(sample.contacts)

    cluster_set = sample.percolating_cluster if sample.percolates else set()
    cluster_contacts = [
        c for c in sample.contacts if c.tube_a in cluster_set and c.tube_b in cluster_set
    ]
    cluster_stats = summarize_contact_conductance(cluster_contacts)

    bfs_hops: int | None = None
    bfs_resistance: float | None = None
    if sample.percolates and sample.percolating_path is not None:
        bfs_hops = max(0, len(sample.percolating_path) - 2)
        if sample.weighted_graph is not None:
            try:
                bfs_resistance = path_series_resistance(
                    sample.weighted_graph, sample.percolating_path
                )
            except ValueError:
                bfs_resistance = None

    min_r_hops: int | None = None
    if sample.min_resistance_path is not None:
        min_r_hops = max(0, len(sample.min_resistance_path) - 2)

    improvement: float | None = None
    if (
        bfs_resistance is not None
        and sample.min_path_resistance is not None
        and sample.min_path_resistance > 0.0
        and np.isfinite(bfs_resistance)
    ):
        improvement = float(bfs_resistance / sample.min_path_resistance)

    return {
        "percolates": sample.percolates,
        "nanotube_count": sample.nanotube_count,
        "contact_count": sample.contact_count,
        "percolating_cluster_size": len(sample.percolating_cluster),
        "tunneling": sample.tunneling.to_dict() if sample.tunneling is not None else None,
        "all_contacts": all_stats.to_dict(),
        "cluster_contacts": cluster_stats.to_dict(),
        "shortest_hop_path_hops": bfs_hops,
        "shortest_hop_path_resistance": bfs_resistance,
        "min_resistance_path_hops": min_r_hops,
        "min_path_resistance": sample.min_path_resistance,
        "resistance_improvement_ratio": improvement,
        "bottleneck_conductance": sample.bottleneck_conductance,
    }


# ---------------------------------------------------------------------------
# Parametric Study Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TunnelingPoint:
    """Monte Carlo & threshold results for a single tunneling parameter point.

    Attributes
    ----------
    label                       : ASCII-safe label (e.g. 'alpha=50 1/um' or 'd_cut=0.05 um')
    parameter_value             : swept scalar value (alpha or cutoff_distance)
    tunneling                   : TunnelingConfig used for this point
    effective_cutoff            : effective contact cutoff distance used, um
    phi_c                       : estimated percolation threshold phi_c
    phi_c_stderr                : standard error of phi_c estimate (if available)
    r_squared                   : logistic fit R^2 (if available)
    concentrations              : evaluated concentrations phi
    probabilities               : empirical percolation probabilities P(phi)
    avg_contact_conductances    : mean contact conductance per concentration
    avg_geom_conductances       : mean geometric-mean contact conductance per concentration
    avg_min_path_resistances    : mean lowest-resistance path across percolating samples
    avg_bottleneck_conductances : mean critical-path conductance across percolating samples
    mc_result                   : optional full MonteCarloResult
    threshold_result            : optional full ThresholdResult
    """

    label: str
    parameter_value: float
    tunneling: TunnelingConfig
    effective_cutoff: float
    phi_c: float | None
    phi_c_stderr: float | None = None
    r_squared: float | None = None
    concentrations: list[float] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    avg_contact_conductances: list[float | None] = field(default_factory=list)
    avg_geom_conductances: list[float | None] = field(default_factory=list)
    avg_min_path_resistances: list[float | None] = field(default_factory=list)
    avg_bottleneck_conductances: list[float | None] = field(default_factory=list)
    mc_result: MonteCarloResult | None = None
    threshold_result: ThresholdResult | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "parameter_value": self.parameter_value,
            "tunneling": self.tunneling.to_dict(),
            "effective_cutoff": self.effective_cutoff,
            "phi_c": self.phi_c,
            "phi_c_stderr": self.phi_c_stderr,
            "r_squared": self.r_squared,
            "concentrations": list(self.concentrations),
            "probabilities": list(self.probabilities),
            "avg_contact_conductances": list(self.avg_contact_conductances),
            "avg_geom_conductances": list(self.avg_geom_conductances),
            "avg_min_path_resistances": list(self.avg_min_path_resistances),
            "avg_bottleneck_conductances": list(self.avg_bottleneck_conductances),
            "threshold_result": self.threshold_result.to_dict() if self.threshold_result else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TunnelingPoint:
        tunn = TunnelingConfig.from_dict(data.get("tunneling"))
        thresh_dict = data.get("threshold_result")
        thresh_res = None
        if thresh_dict:
            thresh_res = ThresholdResult(
                phi_c=thresh_dict.get("phi_c"),
                method=thresh_dict.get("method", "unknown"),
                phi_c_linear=thresh_dict.get("phi_c_linear"),
                confidence_level=thresh_dict.get("confidence_level", 0.95),
                target_probability=thresh_dict.get("target_probability", 0.5),
            )
        return cls(
            label=str(data["label"]),
            parameter_value=float(data["parameter_value"]),
            tunneling=tunn,
            effective_cutoff=float(data["effective_cutoff"]),
            phi_c=float(data["phi_c"]) if data.get("phi_c") is not None else None,
            phi_c_stderr=(
                float(data["phi_c_stderr"]) if data.get("phi_c_stderr") is not None else None
            ),
            r_squared=float(data["r_squared"]) if data.get("r_squared") is not None else None,
            concentrations=[float(c) for c in data.get("concentrations", [])],
            probabilities=[float(p) for p in data.get("probabilities", [])],
            avg_contact_conductances=list(data.get("avg_contact_conductances", [])),
            avg_geom_conductances=list(data.get("avg_geom_conductances", [])),
            avg_min_path_resistances=list(data.get("avg_min_path_resistances", [])),
            avg_bottleneck_conductances=list(data.get("avg_bottleneck_conductances", [])),
            threshold_result=thresh_res,
        )


@dataclass
class TunnelingStudyResult:
    """Complete result of a Phase 8 parametric tunneling conductance study."""

    study_mode: str  # 'alpha' or 'cutoff'
    points: list[TunnelingPoint]
    box: MaterialBox
    cnt_length: float
    cnt_diameter: float
    contact_threshold: float
    orientation: OrientationConfig = field(default_factory=OrientationConfig.isotropic)
    samples_per_concentration: int = 20
    random_seed: int = 42

    def to_dict(self) -> dict:
        return {
            "study_mode": self.study_mode,
            "parameters": {
                "box": {
                    "size_x": self.box.size_x,
                    "size_y": self.box.size_y,
                    "size_z": self.box.size_z,
                },
                "cnt": {
                    "length": self.cnt_length,
                    "diameter": self.cnt_diameter,
                    "aspect_ratio": (
                        self.cnt_length / self.cnt_diameter if self.cnt_diameter > 0 else 0.0
                    ),
                    "orientation": self.orientation.to_dict(),
                },
                "contact_threshold": self.contact_threshold,
                "samples_per_concentration": self.samples_per_concentration,
                "random_seed": self.random_seed,
            },
            "points": [pt.to_dict() for pt in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TunnelingStudyResult:
        params = data["parameters"]
        box_d = params["box"]
        cnt_d = params["cnt"]
        return cls(
            study_mode=str(data.get("study_mode", "alpha")),
            points=[TunnelingPoint.from_dict(p) for p in data.get("points", [])],
            box=MaterialBox(
                size_x=float(box_d["size_x"]),
                size_y=float(box_d["size_y"]),
                size_z=float(box_d["size_z"]),
            ),
            cnt_length=float(cnt_d["length"]),
            cnt_diameter=float(cnt_d["diameter"]),
            contact_threshold=float(params["contact_threshold"]),
            orientation=OrientationConfig.from_dict(cnt_d.get("orientation")),
            samples_per_concentration=int(params.get("samples_per_concentration", 20)),
            random_seed=int(params.get("random_seed", 42)),
        )

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def load_json(cls, path: str | Path) -> TunnelingStudyResult:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_csv(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label",
                "study_mode",
                "parameter_value",
                "alpha",
                "g0",
                "d_min",
                "min_conductance",
                "effective_cutoff",
                "phi_c",
                "phi_c_stderr",
                "r_squared",
                "peak_bottleneck_conductance",
                "min_observed_path_resistance",
            ])
            for pt in self.points:
                valid_bn = [g for g in pt.avg_bottleneck_conductances if g is not None]
                peak_bn = max(valid_bn) if valid_bn else None
                valid_r = [r for r in pt.avg_min_path_resistances if r is not None]
                best_r = min(valid_r) if valid_r else None
                writer.writerow([
                    pt.label,
                    self.study_mode,
                    f"{pt.parameter_value:g}",
                    f"{pt.tunneling.alpha:g}",
                    f"{pt.tunneling.g0:g}",
                    f"{pt.tunneling.d_min:g}",
                    (
                        f"{pt.tunneling.min_conductance:.6e}"
                        if pt.tunneling.min_conductance is not None
                        else ""
                    ),
                    f"{pt.effective_cutoff:.6f}",
                    f"{pt.phi_c:.6f}" if pt.phi_c is not None else "",
                    f"{pt.phi_c_stderr:.6f}" if pt.phi_c_stderr is not None else "",
                    f"{pt.r_squared:.4f}" if pt.r_squared is not None else "",
                    f"{peak_bn:.6e}" if peak_bn is not None else "",
                    f"{best_r:.6e}" if best_r is not None else "",
                ])


# ---------------------------------------------------------------------------
# Adaptive Concentration Bracketing
# ---------------------------------------------------------------------------

def _bracket_concentration_grid(
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    contact_threshold: float,
    tunneling: TunnelingConfig,
    orientation: OrientationConfig,
    n_concentrations: int,
    seed_sequence: np.random.SeedSequence,
    contact_method: str = "auto",
    probe_samples: int = 5,
    initial_phi: float = 0.005,
    max_phi: float = 0.35,
) -> list[float]:
    """Adaptively find a concentration grid bracketing the percolation transition."""
    eff_cut = tunneling.effective_cutoff(contact_threshold)
    if eff_cut < 0.0:
        return list(np.linspace(0.002, 0.02, max(2, n_concentrations)))

    # Initial heuristic scaled by excluded-volume diameter (D + eff_cut)
    ar = cnt_length / cnt_diameter if cnt_diameter > 0 else 10.0
    scale_factor = cnt_diameter / max(cnt_diameter + eff_cut, 1e-6)
    phi_guess = min(max(initial_phi, 0.4 * scale_factor / max(ar, 1.0)), 0.08)

    probe_seeds = seed_sequence.spawn(24)
    seed_idx = 0

    def probe(phi_val: float) -> float:
        nonlocal seed_idx
        sseq = probe_seeds[seed_idx % len(probe_seeds)]
        seed_idx += 1
        res = run_concentration(
            concentration=phi_val,
            samples=probe_samples,
            box=box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            contact_threshold=contact_threshold,
            seed_sequence=sseq,
            contact_method=contact_method,
            orientation=orientation,
            tunneling=tunneling,
        )
        return res.probability

    phi_low = phi_guess
    phi_high = phi_guess

    p_init = probe(phi_guess)
    if p_init >= 0.5:
        # Walk downward until P < 0.2
        phi_high = phi_guess
        cur = phi_guess
        for _ in range(8):
            cur *= 0.55
            if cur < 1e-4:
                phi_low = cur
                break
            p = probe(cur)
            phi_low = cur
            if p <= 0.15:
                break
        # Ensure upper bracket reaches near 1.0
        if p_init < 0.85:
            phi_high = min(max_phi, phi_guess * 1.6)
    else:
        # Walk upward until P > 0.8
        phi_low = phi_guess
        cur = phi_guess
        for _ in range(10):
            cur = min(max_phi, cur * 1.55)
            p = probe(cur)
            phi_high = cur
            if p >= 0.85 or cur >= max_phi:
                break
        if p_init > 0.15:
            phi_low = max(1e-4, phi_guess * 0.5)

    # Slight padding around bracket
    span = max(phi_high - phi_low, 0.002)
    grid_lo = max(1e-4, phi_low - 0.1 * span)
    grid_hi = min(max_phi, phi_high + 0.15 * span)
    return [float(x) for x in np.linspace(grid_lo, grid_hi, max(3, n_concentrations))]


def _evaluate_tunneling_point(
    label: str,
    parameter_value: float,
    tunneling: TunnelingConfig,
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    contact_threshold: float,
    orientation: OrientationConfig,
    samples_per_concentration: int,
    n_concentrations: int,
    point_seed: int,
    concentrations: Sequence[float] | None,
    contact_method: str,
) -> TunnelingPoint:
    """Run Monte Carlo & threshold estimation for one TunnelingConfig point."""
    eff_cut = tunneling.effective_cutoff(contact_threshold)
    sseq = np.random.SeedSequence(point_seed)
    bracket_seq, mc_seq = sseq.spawn(2)

    if concentrations is not None:
        conc_grid = [float(c) for c in concentrations]
    else:
        conc_grid = _bracket_concentration_grid(
            box=box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            contact_threshold=contact_threshold,
            tunneling=tunneling,
            orientation=orientation,
            n_concentrations=n_concentrations,
            seed_sequence=bracket_seq,
            contact_method=contact_method,
            probe_samples=max(3, min(6, samples_per_concentration // 3)),
        )

    mc_seed = int(mc_seq.generate_state(1)[0])
    mc_res = run_monte_carlo(
        concentrations=conc_grid,
        samples_per_concentration=samples_per_concentration,
        box=box,
        cnt_length=cnt_length,
        cnt_diameter=cnt_diameter,
        contact_threshold=contact_threshold,
        random_seed=mc_seed,
        contact_method=contact_method,
        orientation=orientation,
        tunneling=tunneling,
    )

    thresh_res = estimate_threshold(mc_res)
    phi_c = thresh_res.phi_c
    phi_c_stderr = thresh_res.logistic_phi_c_stderr
    r_squared = (
        thresh_res.logistic_fit.r_squared
        if thresh_res.logistic_fit is not None and thresh_res.logistic_fit.converged
        else None
    )

    return TunnelingPoint(
        label=label,
        parameter_value=float(parameter_value),
        tunneling=tunneling,
        effective_cutoff=eff_cut,
        phi_c=phi_c,
        phi_c_stderr=phi_c_stderr,
        r_squared=r_squared,
        concentrations=mc_res.concentrations,
        probabilities=mc_res.probabilities,
        avg_contact_conductances=[r.avg_contact_conductance for r in mc_res.concentration_results],
        avg_geom_conductances=[r.avg_geom_mean_conductance for r in mc_res.concentration_results],
        avg_min_path_resistances=[r.avg_min_path_resistance for r in mc_res.concentration_results],
        avg_bottleneck_conductances=[
            r.avg_bottleneck_conductance for r in mc_res.concentration_results
        ],
        mc_result=mc_res,
        threshold_result=thresh_res,
    )


# ---------------------------------------------------------------------------
# Public Study Runners
# ---------------------------------------------------------------------------

def run_tunneling_alpha_study(
    alpha_values: Sequence[float] = (30.0, 50.0, 80.0, 120.0, 180.0),
    g0: float = 1.0,
    d_min: float = 0.0,
    min_conductance: float | None = 1e-2,
    box: MaterialBox | None = None,
    cnt_length: float = 1.0,
    cnt_diameter: float = 0.1,
    contact_threshold: float = 0.15,
    orientation: OrientationConfig | None = None,
    samples_per_concentration: int = 20,
    n_concentrations: int = 7,
    concentrations: Sequence[float] | None = None,
    random_seed: int = 42,
    contact_method: str = "auto",
    progress_callback: Callable[[int, int, TunnelingPoint], None] | None = None,
) -> TunnelingStudyResult:
    """Study effect of tunneling decay constant alpha on phi_c and bottleneck conductance.

    For each alpha in ``alpha_values``, constructs a ``TunnelingConfig`` with
    ``(g0, alpha, d_min, cutoff_distance=contact_threshold, min_conductance=min_conductance)``,
    runs Monte Carlo across concentrations, and estimates phi_c(alpha) along with
    mean bottleneck conductance and minimum-resistance path statistics.
    """
    if not alpha_values:
        raise ValueError("alpha_values must not be empty.")

    sim_box = box if box is not None else MaterialBox(size_x=6.0, size_y=4.0, size_z=4.0)
    orient_cfg = orientation if orientation is not None else OrientationConfig.isotropic()

    root_seq = np.random.SeedSequence(random_seed)
    point_seqs = root_seq.spawn(len(alpha_values))

    points: list[TunnelingPoint] = []
    total = len(alpha_values)

    for idx, (alpha_val, pseq) in enumerate(zip(alpha_values, point_seqs)):
        alpha_f = float(alpha_val)
        tunn_cfg = TunnelingConfig(
            enabled=True,
            g0=g0,
            alpha=alpha_f,
            d_min=d_min,
            cutoff_distance=contact_threshold,
            min_conductance=min_conductance,
        )
        point_seed = int(pseq.generate_state(1)[0])
        pt = _evaluate_tunneling_point(
            label=f"alpha={alpha_f:g} 1/um",
            parameter_value=alpha_f,
            tunneling=tunn_cfg,
            box=sim_box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            contact_threshold=contact_threshold,
            orientation=orient_cfg,
            samples_per_concentration=samples_per_concentration,
            n_concentrations=n_concentrations,
            point_seed=point_seed,
            concentrations=concentrations,
            contact_method=contact_method,
        )
        points.append(pt)
        if progress_callback is not None:
            progress_callback(idx, total, pt)

    return TunnelingStudyResult(
        study_mode="alpha",
        points=points,
        box=sim_box,
        cnt_length=cnt_length,
        cnt_diameter=cnt_diameter,
        contact_threshold=contact_threshold,
        orientation=orient_cfg,
        samples_per_concentration=samples_per_concentration,
        random_seed=random_seed,
    )


def run_contact_cutoff_study(
    cutoff_values: Sequence[float] = (0.01, 0.03, 0.05, 0.08, 0.12),
    g0: float = 1.0,
    alpha: float = 50.0,
    d_min: float = 0.0,
    box: MaterialBox | None = None,
    cnt_length: float = 1.0,
    cnt_diameter: float = 0.1,
    orientation: OrientationConfig | None = None,
    samples_per_concentration: int = 20,
    n_concentrations: int = 7,
    concentrations: Sequence[float] | None = None,
    random_seed: int = 42,
    contact_method: str = "auto",
    progress_callback: Callable[[int, int, TunnelingPoint], None] | None = None,
) -> TunnelingStudyResult:
    """Study effect of contact cutoff distance d_contact on phi_c (Plan Experiment 3).

    Evaluates phi_c(d_contact) and tunneling conductance metrics across multiple
    cutoff distances at fixed (g0, alpha, d_min).
    """
    if not cutoff_values:
        raise ValueError("cutoff_values must not be empty.")

    sim_box = box if box is not None else MaterialBox(size_x=6.0, size_y=4.0, size_z=4.0)
    orient_cfg = orientation if orientation is not None else OrientationConfig.isotropic()

    root_seq = np.random.SeedSequence(random_seed)
    point_seqs = root_seq.spawn(len(cutoff_values))

    points: list[TunnelingPoint] = []
    total = len(cutoff_values)
    max_cutoff = max(float(c) for c in cutoff_values)

    for idx, (cut_val, pseq) in enumerate(zip(cutoff_values, point_seqs)):
        cut_f = float(cut_val)
        tunn_cfg = TunnelingConfig(
            enabled=True,
            g0=g0,
            alpha=alpha,
            d_min=d_min,
            cutoff_distance=cut_f,
            min_conductance=None,
        )
        point_seed = int(pseq.generate_state(1)[0])
        pt = _evaluate_tunneling_point(
            label=f"d_cut={cut_f:g} um",
            parameter_value=cut_f,
            tunneling=tunn_cfg,
            box=sim_box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            contact_threshold=cut_f,
            orientation=orient_cfg,
            samples_per_concentration=samples_per_concentration,
            n_concentrations=n_concentrations,
            point_seed=point_seed,
            concentrations=concentrations,
            contact_method=contact_method,
        )
        points.append(pt)
        if progress_callback is not None:
            progress_callback(idx, total, pt)

    return TunnelingStudyResult(
        study_mode="cutoff",
        points=points,
        box=sim_box,
        cnt_length=cnt_length,
        cnt_diameter=cnt_diameter,
        contact_threshold=max_cutoff,
        orientation=orient_cfg,
        samples_per_concentration=samples_per_concentration,
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tunneling Contact Conductance Parametric Study (Phase 8)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="alpha",
        choices=["alpha", "cutoff"],
        help="Study sweep mode: 'alpha' (decay constant) or 'cutoff' (d_contact) (default: alpha)",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        default="30,50,80,120,180",
        help="Comma-separated alpha values in 1/um for mode=alpha (default: 30,50,80,120,180)",
    )
    parser.add_argument(
        "--cutoff-values",
        type=str,
        default="0.01,0.03,0.05,0.08,0.12",
        help="Comma-separated cutoff distances in um for mode=cutoff (default: 0.01,0.03,0.05,0.08,0.12)",
    )
    parser.add_argument(
        "--g0",
        type=float,
        default=1.0,
        help="Direct-contact conductance G0 in S (default: 1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Fixed alpha in 1/um for mode=cutoff (default: 50.0)",
    )
    parser.add_argument(
        "--d-min",
        type=float,
        default=0.0,
        help="Minimum direct-contact clamp distance d_min in um (default: 0.0)",
    )
    parser.add_argument(
        "--min-conductance",
        type=float,
        default=1e-2,
        help="Minimum conductance cutoff G_min in S for mode=alpha (default: 1e-2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Maximum geometric contact threshold in um for mode=alpha (default: 0.15)",
    )
    parser.add_argument(
        "--box-size",
        type=str,
        default="6.0,4.0,4.0",
        help="Box dimensions Lx,Ly,Lz in um (default: 6.0,4.0,4.0)",
    )
    parser.add_argument(
        "--cnt-length",
        type=float,
        default=1.0,
        help="CNT length L in um (default: 1.0)",
    )
    parser.add_argument(
        "--cnt-diameter",
        type=float,
        default=0.1,
        help="CNT diameter D in um (default: 0.1)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Monte Carlo samples per concentration (default: 20)",
    )
    parser.add_argument(
        "--n-concentrations",
        type=int,
        default=7,
        help="Number of concentration points per sweep value (default: 7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42)",
    )
    parser.add_argument(
        "--contact-method",
        type=str,
        default="auto",
        choices=["auto", "grid", "hash", "kdtree", "brute_force"],
        help="Spatial contact detection method (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Output directory for JSON/CSV/PNG files (default: results)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating tunneling study plot",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Plot DPI (default: 300)",
    )
    args = parser.parse_args()

    box_dims = [float(x.strip()) for x in args.box_size.split(",")]
    if len(box_dims) != 3:
        raise ValueError(f"--box-size expects 3 comma-separated floats, got {args.box_size!r}")
    box = MaterialBox(size_x=box_dims[0], size_y=box_dims[1], size_z=box_dims[2])

    print("=" * 78, flush=True)
    print(f"  Phase 8: Tunneling Contact Conductance Study (mode={args.mode})", flush=True)
    print("=" * 78, flush=True)
    print(f"Domain Box          : {box.size_x} x {box.size_y} x {box.size_z} um", flush=True)
    print(f"CNT Geometry        : L={args.cnt_length} um, D={args.cnt_diameter} um (AR={args.cnt_length / args.cnt_diameter:.1f})", flush=True)
    print(f"Samples / conc      : {args.samples} x {args.n_concentrations} points", flush=True)
    print("-" * 78, flush=True)
    print(
        f"{'Point':<18} | {'d_eff (um)':>10} | {'phi_c':>10} | {'R^2':>7} | {'Peak G_c (S)':>12}",
        flush=True,
    )
    print("-" * 78, flush=True)

    def _on_point(idx: int, total: int, pt: TunnelingPoint) -> None:
        phi_str = f"{pt.phi_c:10.5f}" if pt.phi_c is not None else f"{'N/A':>10}"
        r2_str = f"{pt.r_squared:7.4f}" if pt.r_squared is not None else f"{'N/A':>7}"
        valid_bn = [g for g in pt.avg_bottleneck_conductances if g is not None]
        bn_str = f"{max(valid_bn):12.3e}" if valid_bn else f"{'N/A':>12}"
        print(
            f"{pt.label:<18} | {pt.effective_cutoff:10.5f} | {phi_str} | {r2_str} | {bn_str}",
            flush=True,
        )

    if args.mode == "alpha":
        alphas = [float(x.strip()) for x in args.alpha_values.split(",") if x.strip()]
        study = run_tunneling_alpha_study(
            alpha_values=alphas,
            g0=args.g0,
            d_min=args.d_min,
            min_conductance=args.min_conductance,
            box=box,
            cnt_length=args.cnt_length,
            cnt_diameter=args.cnt_diameter,
            contact_threshold=args.threshold,
            samples_per_concentration=args.samples,
            n_concentrations=args.n_concentrations,
            random_seed=args.seed,
            contact_method=args.contact_method,
            progress_callback=_on_point,
        )
    else:
        cutoffs = [float(x.strip()) for x in args.cutoff_values.split(",") if x.strip()]
        study = run_contact_cutoff_study(
            cutoff_values=cutoffs,
            g0=args.g0,
            alpha=args.alpha,
            d_min=args.d_min,
            box=box,
            cnt_length=args.cnt_length,
            cnt_diameter=args.cnt_diameter,
            samples_per_concentration=args.samples,
            n_concentrations=args.n_concentrations,
            random_seed=args.seed,
            contact_method=args.contact_method,
            progress_callback=_on_point,
        )

    out_dir = Path(args.output_dir)
    json_path = out_dir / "tunneling_study.json"
    csv_path = out_dir / "tunneling_study.csv"
    study.save_json(json_path)
    study.save_csv(csv_path)

    print("-" * 78, flush=True)
    print(f"Saved JSON : {json_path}", flush=True)
    print(f"Saved CSV  : {csv_path}", flush=True)

    if not args.no_plot:
        from percolation.visualization.tunneling_plot import plot_tunneling_study

        plot_path = out_dir / "tunneling_study.png"
        plot_tunneling_study(study, save_path=plot_path, dpi=args.dpi)
        print(f"Saved Plot : {plot_path}", flush=True)

    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()

"""
CNT Orientation Study - Phase 7 implementation.

Investigates how the 3D spatial orientation distribution of CNTs affects
the percolation threshold phi_c = f(orientation):
- Isotropic (uniform random on S^2, Hermans order parameter S_x = 0)
- Partially aligned (uniform solid-angle cone of half-angle theta_max around axis)
- Strictly aligned (parallel to transport axis X: S_x = 1, or transverse Y/Z: S_x = -0.5)
- von Mises-Fisher angular concentration (kappa >= 0)

Captures the classic non-monotonic dependence of phi_c on alignment:
moderate alignment along the transport axis can optimize connectivity,
while extreme alignment sharply decreases excluded volume and raises phi_c.
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
from percolation.domain.cnt import MaterialBox, OrientationConfig, _normalize_axis
from percolation.generation.cnt_generator import (
    compute_order_parameter,
    count_from_concentration,
    generate_nanotubes,
)
from percolation.simulation.monte_carlo import run_concentration, run_monte_carlo

if TYPE_CHECKING:
    from percolation.domain.cnt import MonteCarloResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrientationPoint:
    """Percolation threshold result for a single CNT orientation configuration.

    Attributes
    ----------
    orientation                 : OrientationConfig used for this experiment
    phi_c                       : estimated percolation threshold phi_c
    phi_c_stderr                : standard error of phi_c estimate (if available)
    r_squared                   : goodness-of-fit R^2 of logistic sigmoid (if available)
    theoretical_order_parameter : analytical Hermans order parameter S along reference axis
    measured_order_parameter    : empirical Hermans order parameter S measured on generated CNTs
    concentrations              : sampled concentration values
    probabilities               : observed percolation probabilities P(phi)
    mc_result                   : optional full MonteCarloResult
    threshold_result            : optional full ThresholdResult
    """

    orientation: OrientationConfig
    phi_c: float
    phi_c_stderr: float | None = None
    r_squared: float | None = None
    theoretical_order_parameter: float = 0.0
    measured_order_parameter: float = 0.0
    concentrations: list[float] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    mc_result: MonteCarloResult | None = None
    threshold_result: ThresholdResult | None = None

    @property
    def label(self) -> str:
        return self.orientation.label

    @property
    def mode(self) -> str:
        return self.orientation.mode

    @property
    def max_angle_deg(self) -> float:
        return self.orientation.max_angle_deg

    @property
    def order_parameter(self) -> float:
        """Alias for theoretical_order_parameter."""
        return self.theoretical_order_parameter

    def to_dict(self) -> dict:
        """Serialize orientation point data to a dictionary."""
        return {
            "label": self.label,
            "orientation": self.orientation.to_dict(),
            "theoretical_order_parameter": self.theoretical_order_parameter,
            "measured_order_parameter": self.measured_order_parameter,
            "phi_c": self.phi_c,
            "phi_c_stderr": self.phi_c_stderr,
            "r_squared": self.r_squared,
            "concentrations": list(self.concentrations),
            "probabilities": list(self.probabilities),
            "threshold_result": self.threshold_result.to_dict() if self.threshold_result else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OrientationPoint:
        """Deserialize OrientationPoint from a dictionary."""
        orient = OrientationConfig.from_dict(data.get("orientation"))
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
            orientation=orient,
            phi_c=float(data["phi_c"]),
            phi_c_stderr=float(data["phi_c_stderr"]) if data.get("phi_c_stderr") is not None else None,
            r_squared=float(data["r_squared"]) if data.get("r_squared") is not None else None,
            theoretical_order_parameter=float(
                data.get("theoretical_order_parameter", orient.theoretical_order_parameter("x"))
            ),
            measured_order_parameter=float(
                data.get("measured_order_parameter", orient.theoretical_order_parameter("x"))
            ),
            concentrations=list(data.get("concentrations", [])),
            probabilities=list(data.get("probabilities", [])),
            threshold_result=thresh_res,
        )


@dataclass
class OrientationStudyResult:
    """Aggregated results from a multi-orientation CNT percolation study."""

    points: list[OrientationPoint]
    box_size: tuple[float, float, float] = (4.0, 2.0, 2.0)
    cnt_length: float = 1.0
    cnt_diameter: float = 0.1
    contact_threshold: float = 0.05
    reference_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    notes: str = "Effect of CNT 3D orientation on percolation threshold phi_c(orientation)"

    @property
    def aspect_ratio(self) -> float:
        return self.cnt_length / self.cnt_diameter if self.cnt_diameter > 0 else 0.0

    @property
    def isotropic_point(self) -> OrientationPoint | None:
        """Return the isotropic reference point if present in the study."""
        for pt in self.points:
            if pt.orientation.mode == "isotropic" or (
                pt.orientation.mode == "partially_aligned"
                and pt.orientation.max_angle_deg >= 90.0 - 1e-6
            ):
                return pt
        return None

    @property
    def isotropic_phi_c(self) -> float | None:
        """Percolation threshold of the isotropic baseline, if available."""
        iso = self.isotropic_point
        return iso.phi_c if iso is not None else None

    @property
    def optimal_point(self) -> OrientationPoint | None:
        """Point achieving the lowest percolation threshold phi_c."""
        valid = [p for p in self.points if np.isfinite(p.phi_c) and p.phi_c > 0]
        if not valid:
            return None
        return min(valid, key=lambda p: p.phi_c)

    def relative_threshold(self, point: OrientationPoint) -> float | None:
        """Return phi_c / phi_c_isotropic for *point*, or None if no isotropic baseline."""
        iso_phi = self.isotropic_phi_c
        if iso_phi is None or iso_phi <= 0:
            return None
        return float(point.phi_c / iso_phi)

    def to_dict(self) -> dict:
        """Serialize complete study result to a dictionary."""
        opt = self.optimal_point
        return {
            "box_size": list(self.box_size),
            "cnt_length": self.cnt_length,
            "cnt_diameter": self.cnt_diameter,
            "aspect_ratio": self.aspect_ratio,
            "contact_threshold": self.contact_threshold,
            "reference_axis": list(self.reference_axis),
            "isotropic_phi_c": self.isotropic_phi_c,
            "optimal_orientation": opt.label if opt is not None else None,
            "optimal_phi_c": opt.phi_c if opt is not None else None,
            "notes": self.notes,
            "points": [p.to_dict() for p in self.points],
        }

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save study results to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def save_csv(self, path: str | Path) -> None:
        """Save orientation summary table to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label",
                "mode",
                "max_angle_deg",
                "kappa",
                "theoretical_order_parameter",
                "measured_order_parameter",
                "phi_c",
                "phi_c_stderr",
                "phi_c_relative_to_isotropic",
                "r_squared",
            ])
            for pt in self.points:
                rel = self.relative_threshold(pt)
                writer.writerow([
                    pt.label,
                    pt.mode,
                    pt.max_angle_deg,
                    pt.orientation.kappa,
                    f"{pt.theoretical_order_parameter:.4f}",
                    f"{pt.measured_order_parameter:.4f}",
                    pt.phi_c,
                    pt.phi_c_stderr if pt.phi_c_stderr is not None else "",
                    f"{rel:.4f}" if rel is not None else "",
                    pt.r_squared if pt.r_squared is not None else "",
                ])

    @classmethod
    def from_dict(cls, data: dict) -> OrientationStudyResult:
        """Deserialize OrientationStudyResult from a dictionary."""
        pts = [OrientationPoint.from_dict(p) for p in data.get("points", [])]
        box = tuple(data.get("box_size", (4.0, 2.0, 2.0)))
        ref = tuple(data.get("reference_axis", (1.0, 0.0, 0.0)))
        return cls(
            points=pts,
            box_size=(float(box[0]), float(box[1]), float(box[2])),
            cnt_length=float(data.get("cnt_length", 1.0)),
            cnt_diameter=float(data.get("cnt_diameter", 0.1)),
            contact_threshold=float(data.get("contact_threshold", 0.05)),
            reference_axis=(float(ref[0]), float(ref[1]), float(ref[2])),
            notes=data.get("notes", ""),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> OrientationStudyResult:
        """Load study result from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Helpers: default orientation schedules & adaptive concentration bracketing
# ---------------------------------------------------------------------------

def build_default_orientations(
    max_angles_deg: Sequence[float] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
    axis: tuple[float, float, float] | str = "x",
    include_transverse: bool = False,
) -> list[OrientationConfig]:
    """Construct a standard sequence of OrientationConfig instances from cone half-angles.

    - ``angle <= 0``  -> ``OrientationConfig.aligned(axis=axis)``
    - ``angle >= 90`` -> ``OrientationConfig.isotropic()``
    - ``0 < angle < 90`` -> ``OrientationConfig.partially_aligned(angle, axis=axis)``
    - If ``include_transverse=True``, appends a transverse aligned state along Y (or Z if axis is Y).
    """
    configs: list[OrientationConfig] = []
    for ang in max_angles_deg:
        a = float(ang)
        if a <= 1e-9:
            configs.append(OrientationConfig.aligned(axis=axis))
        elif a >= 90.0 - 1e-9:
            configs.append(OrientationConfig.isotropic())
        else:
            configs.append(OrientationConfig.partially_aligned(max_angle_deg=a, axis=axis))

    if include_transverse:
        norm_ax = _normalize_axis(axis)
        trans_axis = "y" if abs(norm_ax[1]) < 0.9 else "z"
        configs.append(OrientationConfig.aligned(axis=trans_axis))

    return configs


def _bracket_concentration_grid(
    orientation: OrientationConfig,
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    contact_threshold: float,
    n_concentrations: int,
    seed_sequence: np.random.SeedSequence,
    contact_method: str = "auto",
    probe_samples: int = 5,
    min_phi: float = 1e-4,
    max_phi: float = 0.95,
    max_probe_steps: int = 9,
) -> list[float]:
    """Adaptively bracket the P(phi) percolation transition for *orientation*.

    Because strongly aligned or transverse CNTs can have phi_c shifted by 2x-5x
    relative to isotropic networks, probing guarantees that the concentration grid
    covers the actual sigmoid transition from P ~ 0 to P ~ 1.
    """
    ar = cnt_length / cnt_diameter if cnt_diameter > 0 else 10.0
    # Effective diameter accounts for contact threshold
    d_eff = max(cnt_diameter, contact_threshold)
    ar_eff = cnt_length / d_eff if d_eff > 0 else ar
    # Initial heuristic guess based on excluded volume scaling
    phi_guess = float(np.clip(1.0 / max(ar_eff, 1.0) * (cnt_diameter / d_eff) ** 1.0, min_phi * 2, max_phi * 0.5))
    # Ensure a reasonable minimum count so at least ~10 tubes exist at phi_guess
    v_cnt = np.pi * cnt_diameter**2 * cnt_length / 4.0
    phi_min_tubes = 10.0 * v_cnt / box.volume()
    phi_guess = float(np.clip(max(phi_guess, phi_min_tubes), min_phi, max_phi * 0.5))

    probe_seeds = seed_sequence.spawn(max_probe_steps * 2 + 2)
    seed_idx = 0

    def probe_prob(phi_val: float) -> float:
        nonlocal seed_idx
        res = run_concentration(
            concentration=phi_val,
            samples=probe_samples,
            box=box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            contact_threshold=contact_threshold,
            seed_sequence=probe_seeds[seed_idx],
            contact_method=contact_method,
            orientation=orientation,
        )
        seed_idx += 1
        return res.probability

    p0 = probe_prob(phi_guess)
    phi_low: float | None = None
    phi_high: float | None = None

    if p0 <= 0.15:
        phi_low = phi_guess
        cur = phi_guess
        for _ in range(max_probe_steps):
            nxt = min(max_phi, cur * 1.8)
            if nxt <= cur * 1.01:
                phi_high = max_phi
                break
            p = probe_prob(nxt)
            if p <= 0.15:
                phi_low = nxt
            elif p >= 0.85:
                phi_high = nxt
                break
            cur = nxt
        if phi_high is None:
            phi_high = min(max_phi, cur * 1.3)
    elif p0 >= 0.85:
        phi_high = phi_guess
        cur = phi_guess
        for _ in range(max_probe_steps):
            nxt = max(min_phi, cur * 0.55)
            if nxt >= cur * 0.99:
                phi_low = min_phi
                break
            p = probe_prob(nxt)
            if p >= 0.85:
                phi_high = nxt
            elif p <= 0.15:
                phi_low = nxt
                break
            cur = nxt
        if phi_low is None:
            phi_low = max(min_phi, cur * 0.7)
    else:
        # Already inside transition: expand both directions
        cur_down = phi_guess
        for _ in range(max_probe_steps // 2 + 1):
            nxt = max(min_phi, cur_down * 0.6)
            p = probe_prob(nxt)
            cur_down = nxt
            if p <= 0.15 or nxt <= min_phi * 1.01:
                break
        phi_low = cur_down

        cur_up = phi_guess
        for _ in range(max_probe_steps // 2 + 1):
            nxt = min(max_phi, cur_up * 1.6)
            p = probe_prob(nxt)
            cur_up = nxt
            if p >= 0.85 or nxt >= max_phi * 0.99:
                break
        phi_high = cur_up

    assert phi_low is not None and phi_high is not None
    if phi_high <= phi_low * 1.05:
        phi_high = min(max_phi, phi_low * 2.0)

    # Slight padding so logistic tails are cleanly observed
    lo = max(min_phi, phi_low * 0.85)
    hi = min(max_phi, phi_high * 1.15)
    return [float(x) for x in np.linspace(lo, hi, n_concentrations)]


# ---------------------------------------------------------------------------
# Orientation study runner
# ---------------------------------------------------------------------------

def run_orientation_study(
    orientations: Sequence[OrientationConfig] | None = None,
    max_angles_deg: Sequence[float] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
    axis: tuple[float, float, float] | str = "x",
    include_transverse: bool = False,
    box: MaterialBox | None = None,
    cnt_length: float = 1.0,
    cnt_diameter: float = 0.1,
    contact_threshold: float = 0.05,
    samples_per_concentration: int = 25,
    concentrations: Sequence[float] | None = None,
    concentrations_by_label: dict[str, Sequence[float]] | None = None,
    n_concentrations: int = 7,
    random_seed: int = 42,
    contact_method: str = "auto",
    reference_axis: tuple[float, float, float] | str = "x",
    progress_callback: Callable[[int, int, OrientationPoint], None] | None = None,
) -> OrientationStudyResult:
    """Run a comparative study of CNT orientation effects on percolation threshold phi_c.

    Parameters
    ----------
    orientations              : explicit sequence of OrientationConfig objects.
                                If None, built from ``max_angles_deg``, ``axis``,
                                and ``include_transverse``.
    max_angles_deg            : cone half-angles (degrees) when ``orientations`` is None
    axis                      : preferred alignment axis when ``orientations`` is None
    include_transverse        : whether to include a transverse-aligned baseline (along Y)
    box                       : simulation domain (default: MaterialBox(4.0, 2.0, 2.0))
    cnt_length                : CNT length L, um
    cnt_diameter              : CNT diameter D, um
    contact_threshold         : contact distance threshold, um
    samples_per_concentration : number of Monte Carlo realizations per concentration
    concentrations            : optional shared concentration grid for all orientations
    concentrations_by_label   : optional dict mapping orientation.label -> concentration grid
    n_concentrations          : number of concentration points when bracketing adaptively
    random_seed               : master random seed for full reproducibility
    contact_method            : contact detection algorithm ('auto', 'grid', 'hash', 'kdtree', 'brute_force')
    reference_axis            : transport axis for computing Hermans order parameter S (default: 'x')
    progress_callback         : optional callback invoked as (done, total, OrientationPoint)

    Returns
    -------
    OrientationStudyResult
    """
    if orientations is None:
        orient_list = build_default_orientations(
            max_angles_deg=max_angles_deg,
            axis=axis,
            include_transverse=include_transverse,
        )
    else:
        orient_list = list(orientations)

    if not orient_list:
        raise ValueError("orientations list must not be empty.")

    if box is None:
        box = MaterialBox(size_x=4.0, size_y=2.0, size_z=2.0)

    ref_ax_tuple = _normalize_axis(reference_axis)
    root_seed_seq = np.random.SeedSequence(random_seed)
    orient_seeds = root_seed_seq.spawn(len(orient_list))

    points: list[OrientationPoint] = []
    total = len(orient_list)

    for idx, (orient, sseq) in enumerate(zip(orient_list, orient_seeds)):
        probe_seq, mc_seq, order_seq = sseq.spawn(3)

        # Resolve concentration grid for this orientation
        if concentrations_by_label is not None and orient.label in concentrations_by_label:
            phis = [float(c) for c in concentrations_by_label[orient.label]]
        elif concentrations is not None:
            phis = [float(c) for c in concentrations]
        else:
            phis = _bracket_concentration_grid(
                orientation=orient,
                box=box,
                cnt_length=cnt_length,
                cnt_diameter=cnt_diameter,
                contact_threshold=contact_threshold,
                n_concentrations=n_concentrations,
                seed_sequence=probe_seq,
                contact_method=contact_method,
            )

        mc_result = run_monte_carlo(
            concentrations=phis,
            samples_per_concentration=samples_per_concentration,
            box=box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            contact_threshold=contact_threshold,
            random_seed=int(mc_seq.generate_state(1)[0]),
            contact_method=contact_method,
            orientation=orient,
        )

        thresh_res = estimate_threshold(mc_result)
        phi_c = thresh_res.phi_c if thresh_res.phi_c is not None else float(np.median(phis))
        phi_c_err = thresh_res.logistic_phi_c_stderr
        r2 = thresh_res.logistic_r2

        # Measure realized Hermans order parameter S on a representative ensemble inside box
        n_measure = max(200, min(1000, count_from_concentration(phi_c, box, cnt_length, cnt_diameter)))
        meas_rng = np.random.default_rng(order_seq)
        meas_tubes = generate_nanotubes(
            n=n_measure,
            box=box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            rng=meas_rng,
            orientation=orient,
        )
        s_theory = orient.theoretical_order_parameter(ref_ax_tuple)
        s_measured = compute_order_parameter(meas_tubes, reference_axis=ref_ax_tuple)

        pt = OrientationPoint(
            orientation=orient,
            phi_c=phi_c,
            phi_c_stderr=phi_c_err,
            r_squared=r2,
            theoretical_order_parameter=s_theory,
            measured_order_parameter=s_measured,
            concentrations=[r.concentration for r in mc_result.concentration_results],
            probabilities=[r.probability for r in mc_result.concentration_results],
            mc_result=mc_result,
            threshold_result=thresh_res,
        )
        points.append(pt)

        if progress_callback is not None:
            progress_callback(idx + 1, total, pt)

    return OrientationStudyResult(
        points=points,
        box_size=(box.size_x, box.size_y, box.size_z),
        cnt_length=cnt_length,
        cnt_diameter=cnt_diameter,
        contact_threshold=contact_threshold,
        reference_axis=ref_ax_tuple,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args: Sequence[str] | None = None) -> int:
    """Command-line interface for the Phase 7 Orientation Study."""
    parser = argparse.ArgumentParser(
        description="CNT Orientation Study: Effect of 3D alignment on percolation threshold phi_c."
    )
    parser.add_argument(
        "--angles",
        type=str,
        default="0,15,30,45,60,75,90",
        help="Comma-separated cone half-angles in degrees [0..90] (default: 0,15,30,45,60,75,90)",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="x",
        help="Preferred alignment axis: 'x', 'y', or 'z' (default: x)",
    )
    parser.add_argument(
        "--include-transverse",
        action="store_true",
        help="Also evaluate a transverse-aligned baseline perpendicular to transport axis",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Monte Carlo realizations per concentration point (default: 25)",
    )
    parser.add_argument(
        "--n-concentrations",
        type=int,
        default=7,
        help="Number of concentration points per orientation (default: 7)",
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
        "--threshold",
        type=float,
        default=0.05,
        help="Contact distance threshold in um (default: 0.05)",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=[4.0, 2.0, 2.0],
        metavar=("X", "Y", "Z"),
        help="Domain dimensions in um (default: 4.0 2.0 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--contact-method",
        type=str,
        default="auto",
        choices=["auto", "grid", "hash", "kdtree", "brute_force"],
        help="Contact search acceleration method (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save JSON, CSV, and figures (default: results)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the publication plot",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for saved plot (default: 300)",
    )

    parsed = parser.parse_args(args)
    angle_list = [float(v.strip()) for v in parsed.angles.split(",") if v.strip()]
    box = MaterialBox(size_x=parsed.box_size[0], size_y=parsed.box_size[1], size_z=parsed.box_size[2])
    ar = parsed.cnt_length / parsed.cnt_diameter if parsed.cnt_diameter > 0 else 0.0

    print("=" * 78)
    print("CNT Percolation Simulator - Phase 7: Orientation Study")
    print(f"Cone half-angles : {angle_list} deg (axis={parsed.axis}, transverse={parsed.include_transverse})")
    print(f"Domain Box       : {box.size_x} x {box.size_y} x {box.size_z} um")
    print(f"CNT Geometry     : L={parsed.cnt_length} um, D={parsed.cnt_diameter} um (AR={ar:.1f}), d_c={parsed.threshold} um")
    print(f"Monte Carlo      : {parsed.samples} samples x {parsed.n_concentrations} concentrations | seed={parsed.seed}")
    print("=" * 78)

    def on_progress(done: int, total: int, pt: OrientationPoint) -> None:
        r2_str = f"{pt.r_squared:.3f}" if pt.r_squared is not None else "N/A"
        err_str = f" +/- {pt.phi_c_stderr:.5f}" if pt.phi_c_stderr is not None else ""
        print(
            f"[{done}/{total}] {pt.label:<20} | "
            f"S_x={pt.theoretical_order_parameter:+.3f} (meas {pt.measured_order_parameter:+.3f}) | "
            f"phi_c={pt.phi_c:.5f}{err_str} | R^2={r2_str}"
        )

    study = run_orientation_study(
        max_angles_deg=angle_list,
        axis=parsed.axis,
        include_transverse=parsed.include_transverse,
        box=box,
        cnt_length=parsed.cnt_length,
        cnt_diameter=parsed.cnt_diameter,
        contact_threshold=parsed.threshold,
        samples_per_concentration=parsed.samples,
        n_concentrations=parsed.n_concentrations,
        random_seed=parsed.seed,
        contact_method=parsed.contact_method,
        progress_callback=on_progress,
    )

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "orientation_study.json"
    csv_path = out_dir / "orientation_study.csv"
    study.save_json(json_path)
    study.save_csv(csv_path)

    print("\n" + "-" * 78)
    print("Orientation Study Summary:")
    if study.isotropic_phi_c is not None:
        print(f"  Isotropic baseline phi_c : {study.isotropic_phi_c:.5f}")
    if study.optimal_point is not None:
        opt = study.optimal_point
        rel = study.relative_threshold(opt)
        rel_txt = f" ({rel:.2f}x isotropic)" if rel is not None else ""
        print(f"  Lowest phi_c orientation : {opt.label} -> phi_c = {opt.phi_c:.5f}{rel_txt}")
    print(f"Saved JSON : {json_path}")
    print(f"Saved CSV  : {csv_path}")

    if not parsed.no_plot:
        from percolation.visualization.orientation_plot import plot_orientation_study

        plot_path = out_dir / "orientation_study.png"
        plot_orientation_study(study, save_path=plot_path, dpi=parsed.dpi)
        print(f"Saved Plot : {plot_path} ({parsed.dpi} DPI)")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

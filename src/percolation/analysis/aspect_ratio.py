"""
Aspect Ratio Study - Phase 6 implementation.

Investigates percolation threshold scaling against CNT geometric aspect ratio:
    AR = L / D
Verifies Balberg's excluded volume theory:
    phi_c(AR) = C * AR^(-beta), where beta ~ 1.0.
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
from scipy import stats

from percolation.analysis.threshold import ThresholdResult, estimate_threshold
from percolation.domain.cnt import MaterialBox
from percolation.simulation.monte_carlo import run_monte_carlo

if TYPE_CHECKING:
    from percolation.domain.cnt import MonteCarloResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AspectRatioPoint:
    """Percolation threshold result for a single Aspect Ratio (AR).

    Attributes
    ----------
    aspect_ratio     : CNT aspect ratio AR = L / D
    length           : CNT length L, um
    diameter         : CNT diameter D, um
    phi_c            : estimated percolation threshold phi_c
    phi_c_stderr     : standard error of phi_c estimate (if available)
    r_squared        : goodness-of-fit R^2 of logistic sigmoid (if available)
    concentrations   : sampled concentration values
    probabilities    : observed percolation probabilities P(phi)
    mc_result        : optional full MonteCarloResult
    threshold_result : optional full ThresholdResult
    """

    aspect_ratio: float
    length: float
    diameter: float
    phi_c: float
    phi_c_stderr: float | None = None
    r_squared: float | None = None
    concentrations: list[float] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    mc_result: MonteCarloResult | None = None
    threshold_result: ThresholdResult | None = None

    @property
    def ar(self) -> float:
        """Alias for aspect_ratio."""
        return self.aspect_ratio

    @property
    def phi_c_error(self) -> float | None:
        """Alias for phi_c_stderr."""
        return self.phi_c_stderr

    def to_dict(self) -> dict:
        """Serialize point data to a dictionary."""
        return {
            "aspect_ratio": self.aspect_ratio,
            "length": self.length,
            "diameter": self.diameter,
            "phi_c": self.phi_c,
            "phi_c_stderr": self.phi_c_stderr,
            "r_squared": self.r_squared,
            "concentrations": list(self.concentrations),
            "probabilities": list(self.probabilities),
            "threshold_result": self.threshold_result.to_dict() if self.threshold_result else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AspectRatioPoint:
        """Deserialize point data from a dictionary."""
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
            aspect_ratio=float(data["aspect_ratio"]),
            length=float(data["length"]),
            diameter=float(data["diameter"]),
            phi_c=float(data["phi_c"]),
            phi_c_stderr=float(data["phi_c_stderr"]) if data.get("phi_c_stderr") is not None else None,
            r_squared=float(data["r_squared"]) if data.get("r_squared") is not None else None,
            concentrations=list(data.get("concentrations", [])),
            probabilities=list(data.get("probabilities", [])),
            threshold_result=thresh_res,
        )


@dataclass
class PowerLawFitResult:
    """Outcome of power law regression: phi_c = C * AR^(-beta).

    Attributes
    ----------
    beta          : scaling power exponent (theoretically ~ 1.0)
    c_coefficient : pre-exponential constant C
    r_squared     : coefficient of determination R^2 in log-log space
    beta_stderr   : standard error of exponent beta
    c_stderr      : standard error of constant C
    p_value       : p-value from two-sided Wald test
    """

    beta: float
    c_coefficient: float
    r_squared: float
    beta_stderr: float | None = None
    c_stderr: float | None = None
    p_value: float | None = None

    def predict(self, ar: float | Sequence[float] | np.ndarray) -> np.ndarray | float:
        """Evaluate power law prediction phi_c = C * AR^(-beta)."""
        ar_arr = np.asarray(ar, dtype=float)
        pred = self.c_coefficient * np.power(ar_arr, -self.beta)
        if np.isscalar(ar):
            return float(pred.item())
        return pred

    def to_dict(self) -> dict:
        """Serialize fit result to a dictionary."""
        return {
            "beta": self.beta,
            "c_coefficient": self.c_coefficient,
            "r_squared": self.r_squared,
            "beta_stderr": self.beta_stderr,
            "c_stderr": self.c_stderr,
            "p_value": self.p_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PowerLawFitResult:
        """Deserialize fit result from a dictionary."""
        return cls(
            beta=float(data["beta"]),
            c_coefficient=float(data["c_coefficient"]),
            r_squared=float(data["r_squared"]),
            beta_stderr=float(data["beta_stderr"]) if data.get("beta_stderr") is not None else None,
            c_stderr=float(data["c_stderr"]) if data.get("c_stderr") is not None else None,
            p_value=float(data["p_value"]) if data.get("p_value") is not None else None,
        )


@dataclass
class AspectRatioStudyResult:
    """Complete results from an aspect ratio study."""

    points: list[AspectRatioPoint]
    power_law: PowerLawFitResult
    box_size: tuple[float, float, float] = (3.0, 2.0, 2.0)
    contact_threshold: float | None = None
    notes: str = "Balberg excluded volume scaling: phi_c ~ AR^(-beta)"

    @property
    def beta(self) -> float:
        return self.power_law.beta

    @property
    def c_coefficient(self) -> float:
        return self.power_law.c_coefficient

    @property
    def r_squared(self) -> float:
        return self.power_law.r_squared

    @property
    def beta_stderr(self) -> float | None:
        return self.power_law.beta_stderr

    @property
    def c_stderr(self) -> float | None:
        return self.power_law.c_stderr

    def predict(self, ar: float | Sequence[float] | np.ndarray) -> np.ndarray | float:
        """Evaluate power law prediction."""
        return self.power_law.predict(ar)

    def to_dict(self) -> dict:
        """Serialize complete study result to a dictionary."""
        return {
            "beta": self.beta,
            "c_coefficient": self.c_coefficient,
            "r_squared": self.r_squared,
            "beta_stderr": self.beta_stderr,
            "c_stderr": self.c_stderr,
            "power_law": self.power_law.to_dict(),
            "box_size": list(self.box_size),
            "contact_threshold": self.contact_threshold,
            "notes": self.notes,
            "points": [p.to_dict() for p in self.points],
        }

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save results to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def save_csv(self, path: str | Path) -> None:
        """Save points summary to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["aspect_ratio", "length", "diameter", "phi_c", "phi_c_stderr", "r_squared"])
            for pt in self.points:
                writer.writerow([
                    pt.aspect_ratio,
                    pt.length,
                    pt.diameter,
                    pt.phi_c,
                    pt.phi_c_stderr if pt.phi_c_stderr is not None else "",
                    pt.r_squared if pt.r_squared is not None else "",
                ])

    @classmethod
    def from_dict(cls, data: dict) -> AspectRatioStudyResult:
        """Deserialize study result from a dictionary."""
        pts = [AspectRatioPoint.from_dict(p) for p in data.get("points", [])]
        pl = PowerLawFitResult.from_dict(data["power_law"]) if "power_law" in data else PowerLawFitResult(
            beta=float(data.get("beta", 1.0)),
            c_coefficient=float(data.get("c_coefficient", 1.0)),
            r_squared=float(data.get("r_squared", 0.0)),
            beta_stderr=float(data["beta_stderr"]) if data.get("beta_stderr") is not None else None,
            c_stderr=float(data["c_stderr"]) if data.get("c_stderr") is not None else None,
        )
        box = tuple(data.get("box_size", (3.0, 2.0, 2.0)))
        return cls(
            points=pts,
            power_law=pl,
            box_size=(float(box[0]), float(box[1]), float(box[2])),
            contact_threshold=data.get("contact_threshold"),
            notes=data.get("notes", ""),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> AspectRatioStudyResult:
        """Load study result from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Power law fitting function
# ---------------------------------------------------------------------------

def fit_power_law(
    ar_values: Sequence[float],
    phi_c_values: Sequence[float],
) -> PowerLawFitResult:
    """Fit power law scaling phi_c = C * AR^(-beta) using log-log linear regression.

    Model:
        ln(phi_c) = ln(C) - beta * ln(AR)

    Parameters
    ----------
    ar_values    : Sequence of Aspect Ratio values (AR = L / D > 0)
    phi_c_values : Corresponding estimated percolation thresholds (phi_c > 0)

    Returns
    -------
    PowerLawFitResult
        Fitted exponent beta, prefactor C, R^2, and standard errors.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, fewer than 2 valid positive points,
        or zero variance.
    """
    if len(ar_values) != len(phi_c_values):
        raise ValueError(
            f"ar_values and phi_c_values must have the same length ({len(ar_values)} != {len(phi_c_values)})."
        )

    ar_arr = np.asarray(ar_values, dtype=float)
    phi_arr = np.asarray(phi_c_values, dtype=float)

    # Filter invalid points (non-positive or NaN/inf)
    valid_mask = (ar_arr > 0) & (phi_arr > 0) & np.isfinite(ar_arr) & np.isfinite(phi_arr)
    valid_ar = ar_arr[valid_mask]
    valid_phi = phi_arr[valid_mask]

    if len(valid_ar) < 2:
        raise ValueError("At least 2 positive valid data points are required to fit a power law.")

    log_x = np.log(valid_ar)
    log_y = np.log(valid_phi)

    if np.allclose(log_x, log_x[0]):
        raise ValueError("Cannot fit power law: all AR values are identical.")

    reg = stats.linregress(log_x, log_y)

    slope = float(reg.slope)
    intercept = float(reg.intercept)
    r_val = float(reg.rvalue)
    r_squared = float(r_val * r_val)
    beta = -slope
    c_coeff = float(np.exp(intercept))
    beta_stderr = float(reg.stderr) if reg.stderr is not None else None
    c_stderr = None
    if getattr(reg, "intercept_stderr", None) is not None:
        c_stderr = float(c_coeff * reg.intercept_stderr)

    return PowerLawFitResult(
        beta=beta,
        c_coefficient=c_coeff,
        r_squared=r_squared,
        beta_stderr=beta_stderr,
        c_stderr=c_stderr,
        p_value=float(reg.pvalue) if reg.pvalue is not None else None,
    )


# ---------------------------------------------------------------------------
# Aspect Ratio study runner
# ---------------------------------------------------------------------------

def run_aspect_ratio_study(
    ar_values: Sequence[float] = (15.0, 25.0, 40.0, 60.0, 100.0),
    box: MaterialBox | None = None,
    cnt_length: float = 1.0,
    cnt_diameter: float | None = None,
    contact_threshold: float | None = None,
    contact_threshold_ratio: float = 0.5,
    samples_per_concentration: int = 25,
    concentrations_by_ar: dict[float, Sequence[float]] | None = None,
    n_concentrations: int = 7,
    concentration_span: tuple[float, float] = (0.4, 1.6),
    random_seed: int = 42,
    contact_method: str = "auto",
    progress_callback: Callable[[int, int, AspectRatioPoint], None] | None = None,
) -> AspectRatioStudyResult:
    """Run a comprehensive aspect ratio scaling study.

    For each specified AR = L / D:
    1. Determines tube geometry (L, D) and contact threshold.
    2. Runs Monte Carlo simulations across an appropriate concentration grid.
    3. Estimates the percolation threshold phi_c using logistic sigmoid fitting / linear interpolation.
    4. Aggregates data and fits the scaling law phi_c = C * AR^(-beta).

    Parameters
    ----------
    ar_values                 : list or tuple of aspect ratios to evaluate
    box                       : simulation domain (default: MaterialBox(3.0, 2.0, 2.0))
    cnt_length                : fixed CNT length L (if cnt_diameter is None)
    cnt_diameter              : fixed CNT diameter D (if specified, L = AR * D)
    contact_threshold         : distance threshold for contacts; if None, uses contact_threshold_ratio * D
    contact_threshold_ratio   : ratio of contact threshold to tube diameter (default 0.5)
    samples_per_concentration : number of Monte Carlo realizations per concentration
    concentrations_by_ar      : optional explicit map of {ar: [phi1, phi2, ...]}
    n_concentrations          : number of concentration points per AR (default 7)
    concentration_span        : fraction range around expected threshold (default (0.4, 1.6))
    random_seed               : master random seed for reproducible runs
    contact_method            : contact detection algorithm ('auto', 'grid', 'hash', 'kdtree', 'brute_force')
    progress_callback         : optional callback (completed_count, total_count, point)

    Returns
    -------
    AspectRatioStudyResult
    """
    if not ar_values:
        raise ValueError("ar_values must not be empty.")

    if box is None:
        box = MaterialBox(size_x=3.0, size_y=2.0, size_z=2.0)

    root_seed_seq = np.random.SeedSequence(random_seed)
    ar_seeds = root_seed_seq.spawn(len(ar_values))

    points: list[AspectRatioPoint] = []
    total_ars = len(ar_values)

    for idx, (ar_val, sseq) in enumerate(zip(ar_values, ar_seeds)):
        ar = float(ar_val)
        if cnt_diameter is not None:
            d = float(cnt_diameter)
            l_tube = ar * d
        else:
            l_tube = float(cnt_length)
            d = l_tube / ar

        # Contact threshold
        if contact_threshold is not None:
            th = float(contact_threshold)
        else:
            th = float(contact_threshold_ratio * d)

        # Concentration sequence for this AR
        if concentrations_by_ar is not None and ar in concentrations_by_ar:
            phis = list(concentrations_by_ar[ar])
        else:
            # Theoretical Balberg center: phi_c ~ 1.2 / AR
            phi_center = 1.2 / ar
            phis = list(np.linspace(
                concentration_span[0] * phi_center,
                concentration_span[1] * phi_center,
                n_concentrations,
            ))

        # Run Monte Carlo for this AR
        mc_result = run_monte_carlo(
            concentrations=phis,
            samples_per_concentration=samples_per_concentration,
            box=box,
            cnt_length=l_tube,
            cnt_diameter=d,
            contact_threshold=th,
            random_seed=int(sseq.generate_state(1)[0]),
            contact_method=contact_method,
        )

        # Estimate percolation threshold
        thresh_res = estimate_threshold(mc_result)
        phi_c = thresh_res.phi_c if thresh_res.phi_c is not None else float(np.median(phis))
        phi_c_err = thresh_res.logistic_phi_c_stderr
        r2 = thresh_res.logistic_r2

        pt = AspectRatioPoint(
            aspect_ratio=ar,
            length=l_tube,
            diameter=d,
            phi_c=phi_c,
            phi_c_stderr=phi_c_err,
            r_squared=r2,
            concentrations=[r.concentration for r in mc_result.concentration_results],
            probabilities=[r.probability for r in mc_result.concentration_results],
            mc_result=mc_result,
            threshold_result=thresh_res,
        )
        points.append(pt)

        if progress_callback is not None:
            progress_callback(idx + 1, total_ars, pt)

    # Fit scaling law across all evaluated points
    ar_list = [p.aspect_ratio for p in points]
    phi_c_list = [p.phi_c for p in points]
    power_law = fit_power_law(ar_list, phi_c_list)

    box_dims = (box.size_x, box.size_y, box.size_z)
    return AspectRatioStudyResult(
        points=points,
        power_law=power_law,
        box_size=box_dims,
        contact_threshold=contact_threshold,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args: Sequence[str] | None = None) -> int:
    """Command-line interface for the Aspect Ratio Study."""
    parser = argparse.ArgumentParser(
        description="CNT Aspect Ratio Study: Scaling of percolation threshold phi_c(AR) and Balberg power law fit."
    )
    parser.add_argument(
        "--ar-values",
        type=str,
        default="15,25,40,60,100",
        help="Comma-separated list of aspect ratios (default: 15,25,40,60,100)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of Monte Carlo realizations per concentration (default: 25)",
    )
    parser.add_argument(
        "--cnt-length",
        type=float,
        default=1.0,
        help="Fixed CNT length L (default: 1.0 um)",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=[3.0, 2.0, 2.0],
        metavar=("X", "Y", "Z"),
        help="Domain dimensions in um (default: 3.0 2.0 2.0)",
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
        help="Directory to save JSON and figures (default: results)",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        default=True,
        help="Generate and save high-resolution publication plot (default: True)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for saved plot (default: 300)",
    )

    parsed = parser.parse_args(args)
    ar_list = [float(v.strip()) for v in parsed.ar_values.split(",") if v.strip()]
    box = MaterialBox(size_x=parsed.box_size[0], size_y=parsed.box_size[1], size_z=parsed.box_size[2])

    print("=" * 70)
    print("CNT Percolation Simulator - Phase 6: Aspect Ratio Study")
    print(f"Evaluating AR values: {ar_list}")
    print(f"Domain Box: {box.size_x} x {box.size_y} x {box.size_z} um, CNT Length: {parsed.cnt_length} um")
    print(f"Monte Carlo samples: {parsed.samples} per concentration point")
    print("=" * 70)

    def on_progress(done: int, total: int, pt: AspectRatioPoint):
        r2_str = f"{pt.r_squared:.3f}" if pt.r_squared is not None else "N/A"
        print(f"[{done}/{total}] AR={pt.aspect_ratio:5.1f} | D={pt.diameter:.4f} um | phi_c={pt.phi_c:.5f} | R^2={r2_str}")

    study = run_aspect_ratio_study(
        ar_values=ar_list,
        box=box,
        cnt_length=parsed.cnt_length,
        samples_per_concentration=parsed.samples,
        random_seed=parsed.seed,
        contact_method=parsed.contact_method,
        progress_callback=on_progress,
    )

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "aspect_ratio_study.json"
    csv_path = out_dir / "aspect_ratio_study.csv"
    study.save_json(json_path)
    study.save_csv(csv_path)

    print("\n" + "-" * 70)
    print("Scaling Law Fit: phi_c = C * AR^(-beta)")
    err_str = f" +/- {study.beta_stderr:.3f}" if study.beta_stderr is not None else ""
    print(f"  Exponent beta        : {study.beta:.3f}{err_str}  (Theory: beta ~ 1.00)")
    print(f"  Prefactor C       : {study.c_coefficient:.4f}")
    print(f"  Goodness-of-fit R^2: {study.r_squared:.4f}")
    print(f"Saved JSON : {json_path}")
    print(f"Saved CSV  : {csv_path}")

    if parsed.save_plot:
        from percolation.visualization.aspect_ratio_plot import plot_aspect_ratio_study
        plot_path = out_dir / "aspect_ratio_scaling.png"
        plot_aspect_ratio_study(study, save_path=plot_path, dpi=parsed.dpi)
        print(f"Saved Plot : {plot_path} ({parsed.dpi} DPI)")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

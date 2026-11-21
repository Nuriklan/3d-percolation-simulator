"""
Monte Carlo simulation runner for CNT percolation.

Phases 2, 5, 7 & 8 implementation:
- Loop over a range of concentrations φ.
- Multiple independent, reproducible samples per concentration.
- Statistical aggregation: P(φ) = N_percolating / N_samples.
- Reproducibility via NumPy SeedSequence hierarchy.
- Configuration loading from YAML and export to JSON/CSV.
- Spatial acceleration (UniformSpatialGrid, KDTreeIndex, SpatialHashGrid) via contact_method.
- CNT 3D orientation distributions via OrientationConfig.
- Exponential tunneling conductance statistics via TunnelingConfig (Phase 8).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import yaml

from percolation.domain.cnt import (
    ConcentrationResult,
    MaterialBox,
    MonteCarloResult,
    OrientationConfig,
    SimulationResult,
)
from percolation.domain.tunneling import TunnelingConfig
from percolation.simulation.experiment import run_sample


def run_concentration(
    concentration: float,
    samples: int,
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    contact_threshold: float,
    seed_sequence: np.random.SeedSequence | int,
    electrode_threshold: float | None = None,
    record_samples: bool = False,
    sample_callback: Callable[[int, SimulationResult], None] | None = None,
    contact_method: str = "auto",
    orientation: OrientationConfig | None = None,
    tunneling: TunnelingConfig | None = None,
) -> ConcentrationResult:
    """Run Monte Carlo simulation for a single CNT concentration.

    Parameters
    ----------
    concentration       : volumetric CNT fraction φ
    samples             : number of Monte Carlo realizations
    box                 : simulation domain box
    cnt_length          : CNT length L, μm
    cnt_diameter        : CNT diameter D, μm
    contact_threshold   : distance threshold for CNT-CNT contact, μm
    seed_sequence       : SeedSequence or integer seed for spawning reproducible sample RNGs
    electrode_threshold : distance threshold to electrodes (defaults to contact_threshold)
    record_samples      : whether to retain individual SimulationResult instances
    sample_callback     : optional callback invoked after each sample (index, result)
    contact_method      : contact detection algorithm ('auto', 'grid', 'hash', 'kdtree', 'brute_force')
    orientation         : optional OrientationConfig (defaults to isotropic)
    tunneling           : optional TunnelingConfig (Phase 8)

    Returns
    -------
    ConcentrationResult
    """
    if samples <= 0:
        raise ValueError(f"Number of samples must be positive, got {samples}")

    if isinstance(seed_sequence, int):
        root_seq = np.random.SeedSequence(seed_sequence)
    else:
        root_seq = seed_sequence

    child_seeds = root_seq.spawn(samples)

    percolating_count = 0
    total_contacts = 0
    total_tubes = 0
    recorded_results: list[SimulationResult] = []

    use_tunneling = tunneling is not None and tunneling.enabled
    g_means: list[float] = []
    g_geoms: list[float] = []
    min_r_vals: list[float] = []
    bn_g_vals: list[float] = []

    for idx, sseq in enumerate(child_seeds):
        rng = np.random.default_rng(sseq)
        sample_res = run_sample(
            concentration=concentration,
            box=box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            contact_threshold=contact_threshold,
            rng=rng,
            electrode_threshold=electrode_threshold,
            contact_method=contact_method,
            orientation=orientation,
            tunneling=tunneling if use_tunneling else None,
        )

        if sample_res.percolates:
            percolating_count += 1
        total_contacts += sample_res.contact_count
        total_tubes += sample_res.nanotube_count

        if use_tunneling:
            if sample_res.avg_contact_conductance is not None:
                g_means.append(sample_res.avg_contact_conductance)
            if sample_res.geom_mean_conductance is not None:
                g_geoms.append(sample_res.geom_mean_conductance)
            if sample_res.min_path_resistance is not None and np.isfinite(sample_res.min_path_resistance):
                min_r_vals.append(sample_res.min_path_resistance)
            if sample_res.bottleneck_conductance is not None:
                bn_g_vals.append(sample_res.bottleneck_conductance)

        if record_samples:
            recorded_results.append(sample_res)

        if sample_callback is not None:
            sample_callback(idx, sample_res)

    probability = percolating_count / samples
    avg_contacts = total_contacts / samples
    avg_tubes = total_tubes / samples

    avg_g = float(np.mean(g_means)) if g_means else None
    avg_geom_g = float(np.mean(g_geoms)) if g_geoms else None
    avg_min_r = float(np.mean(min_r_vals)) if min_r_vals else None
    avg_bn_g = float(np.mean(bn_g_vals)) if bn_g_vals else None

    return ConcentrationResult(
        concentration=concentration,
        samples=samples,
        percolating_samples=percolating_count,
        probability=probability,
        avg_contact_count=avg_contacts,
        avg_nanotube_count=avg_tubes,
        sample_results=recorded_results,
        avg_contact_conductance=avg_g,
        avg_geom_mean_conductance=avg_geom_g,
        avg_min_path_resistance=avg_min_r,
        avg_bottleneck_conductance=avg_bn_g,
    )


def run_monte_carlo(
    concentrations: Sequence[float],
    samples_per_concentration: int,
    box: MaterialBox,
    cnt_length: float,
    cnt_diameter: float,
    contact_threshold: float,
    random_seed: int = 42,
    electrode_threshold: float | None = None,
    record_samples: bool = False,
    concentration_callback: Callable[[int, int, ConcentrationResult], None] | None = None,
    contact_method: str = "auto",
    orientation: OrientationConfig | None = None,
    tunneling: TunnelingConfig | None = None,
) -> MonteCarloResult:
    """Run full Monte Carlo study over multiple concentrations.

    Parameters
    ----------
    concentrations            : sequence of volumetric fractions φ
    samples_per_concentration : number of realizations per concentration
    box                       : simulation domain
    cnt_length                : CNT length L, μm
    cnt_diameter              : CNT diameter D, μm
    contact_threshold         : distance threshold for contact, μm
    random_seed               : master random seed for full reproducibility
    electrode_threshold       : distance threshold to electrodes
    record_samples            : whether to retain individual SimulationResult instances
    concentration_callback    : optional callback invoked after each concentration completes
                                (conc_idx, total_concs, result)
    contact_method            : contact search method ('auto', 'grid', 'hash', 'kdtree', 'brute_force')
    orientation               : optional OrientationConfig (defaults to isotropic)
    tunneling                 : optional TunnelingConfig (Phase 8)

    Returns
    -------
    MonteCarloResult
    """
    if not concentrations:
        raise ValueError("concentrations list must not be empty.")

    orient_cfg = orientation if orientation is not None else OrientationConfig.isotropic()

    root_seed_seq = np.random.SeedSequence(random_seed)
    conc_seed_seqs = root_seed_seq.spawn(len(concentrations))

    concentration_results: list[ConcentrationResult] = []
    total_concs = len(concentrations)

    for i, (phi, sseq) in enumerate(zip(concentrations, conc_seed_seqs)):
        conc_res = run_concentration(
            concentration=phi,
            samples=samples_per_concentration,
            box=box,
            cnt_length=cnt_length,
            cnt_diameter=cnt_diameter,
            contact_threshold=contact_threshold,
            seed_sequence=sseq,
            electrode_threshold=electrode_threshold,
            record_samples=record_samples,
            contact_method=contact_method,
            orientation=orient_cfg,
            tunneling=tunneling,
        )
        concentration_results.append(conc_res)

        if concentration_callback is not None:
            concentration_callback(i, total_concs, conc_res)

    return MonteCarloResult(
        box=box,
        cnt_length=cnt_length,
        cnt_diameter=cnt_diameter,
        contact_threshold=contact_threshold,
        random_seed=random_seed,
        concentration_results=concentration_results,
        orientation=orient_cfg,
        tunneling=tunneling,
    )


def run_from_config(
    config_path: str | Path,
    record_samples: bool = False,
    verbose: bool = True,
    contact_method: str | None = None,
    orientation: OrientationConfig | None = None,
    tunneling: TunnelingConfig | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation configured via a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    box = MaterialBox(
        size_x=float(cfg["box"]["x"]),
        size_y=float(cfg["box"]["y"]),
        size_z=float(cfg["box"]["z"]),
    )
    cnt_length = float(cfg["cnt"]["length"])
    cnt_diameter = float(cfg["cnt"]["diameter"])
    contact_threshold = float(cfg["contact"]["threshold"])

    # Orientation priority: explicit argument > config > default isotropic
    if orientation is None:
        orient_raw = cfg.get("cnt", {}).get("orientation")
        orientation = OrientationConfig.from_dict(orient_raw)

    # Tunneling priority: explicit argument > config > disabled (None)
    if tunneling is None:
        tunn_raw = cfg.get("contact", {}).get("tunneling")
        if tunn_raw:
            parsed_tunn = TunnelingConfig.from_dict(tunn_raw)
            tunneling = parsed_tunn if parsed_tunn.enabled else None

    # Method priority: CLI override > config > default "auto"
    if contact_method is None:
        contact_method = str(cfg.get("contact", {}).get("method", "auto"))

    concentrations = [float(c) for c in cfg["simulation"]["concentrations"]]
    samples = int(cfg["simulation"]["samples_per_concentration"])
    seed = int(cfg["simulation"].get("random_seed", 42))

    use_tunneling = tunneling is not None and tunneling.enabled

    if verbose:
        print("=" * 78, flush=True)
        print("  CNT Percolation Simulator - Monte Carlo Simulation", flush=True)
        print("=" * 78, flush=True)
        print(f"Domain Box           : {box.size_x} x {box.size_y} x {box.size_z} um", flush=True)
        print(f"CNT Dimensions       : L = {cnt_length} um, D = {cnt_diameter} um (AR = {cnt_length / cnt_diameter:.1f})", flush=True)
        print(f"CNT Orientation      : {orientation.label} (S_x = {orientation.theoretical_order_parameter('x'):.3f})", flush=True)
        print(f"Contact Threshold    : {contact_threshold} um", flush=True)
        print(f"Contact Method       : {contact_method}", flush=True)
        if use_tunneling and tunneling is not None:
            print(f"Tunneling Model      : {tunneling.label}", flush=True)
            print(f"Effective Cutoff     : {tunneling.effective_cutoff(contact_threshold):.5f} um", flush=True)
        print(f"Concentrations (phi) : {concentrations}", flush=True)
        print(f"Samples per phi      : {samples}", flush=True)
        print(f"Master Seed          : {seed}", flush=True)
        print("-" * 78, flush=True)
        if use_tunneling:
            print(
                f"{'phi':>8} | {'N (avg)':>8} | {'Contacts':>9} | {'Percolated':>11} | "
                f"{'P(phi)':>7} | {'G_c (avg)':>10} | {'Elapsed':>7}",
                flush=True,
            )
        else:
            print(f"{'phi':>8} | {'N (avg)':>8} | {'Contacts':>10} | {'Percolated':>12} | {'P(phi)':>8} | {'Elapsed':>8}", flush=True)
        print("-" * 78, flush=True)

    start_time = time.perf_counter()

    def progress_callback(idx: int, total: int, r: ConcentrationResult) -> None:
        if verbose:
            elapsed = time.perf_counter() - start_time
            if use_tunneling:
                gc_str = (
                    f"{r.avg_bottleneck_conductance:10.3e}"
                    if r.avg_bottleneck_conductance is not None
                    else f"{'N/A':>10}"
                )
                print(
                    f"{r.concentration:8.4f} | "
                    f"{r.avg_nanotube_count:8.1f} | "
                    f"{r.avg_contact_count:9.1f} | "
                    f"{r.percolating_samples:4d} / {r.samples:<4d} | "
                    f"{r.probability:7.3f} | "
                    f"{gc_str} | "
                    f"{elapsed:6.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"{r.concentration:8.4f} | "
                    f"{r.avg_nanotube_count:8.1f} | "
                    f"{r.avg_contact_count:10.1f} | "
                    f"{r.percolating_samples:5d} / {r.samples:<5d} | "
                    f"{r.probability:8.3f} | "
                    f"{elapsed:7.2f}s",
                    flush=True,
                )

    result = run_monte_carlo(
        concentrations=concentrations,
        samples_per_concentration=samples,
        box=box,
        cnt_length=cnt_length,
        cnt_diameter=cnt_diameter,
        contact_threshold=contact_threshold,
        random_seed=seed,
        record_samples=record_samples,
        concentration_callback=progress_callback if verbose else None,
        contact_method=contact_method,
        orientation=orientation,
        tunneling=tunneling,
    )

    if verbose:
        total_elapsed = time.perf_counter() - start_time
        print("-" * 78, flush=True)
        print(f"Total simulation time: {total_elapsed:.2f} seconds", flush=True)
        print("=" * 78, flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo CNT Percolation Simulator (Phases 2-8)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Directory to save simulation results (default: results)",
    )
    parser.add_argument(
        "--contact-method",
        type=str,
        default=None,
        choices=["auto", "grid", "hash", "kdtree", "brute_force"],
        help="Override contact detection algorithm (default: from config or auto)",
    )
    parser.add_argument(
        "--orientation-mode",
        type=str,
        default=None,
        choices=["isotropic", "aligned", "partially_aligned", "fisher"],
        help="Override CNT orientation mode (default: from config or isotropic)",
    )
    parser.add_argument(
        "--orientation-axis",
        type=str,
        default="x",
        help="Preferred alignment axis when --orientation-mode is set (default: x)",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=30.0,
        help="Cone half-angle in degrees when --orientation-mode=partially_aligned (default: 30.0)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=5.0,
        help="Concentration parameter kappa when --orientation-mode=fisher (default: 5.0)",
    )
    parser.add_argument(
        "--tunneling",
        action="store_true",
        help="Enable tunneling contact conductance model (overrides config)",
    )
    parser.add_argument(
        "--g0",
        type=float,
        default=1.0,
        help="Direct-contact conductance G0 when --tunneling is set (default: 1.0 S)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Tunneling decay constant alpha (1/um) when --tunneling is set (default: 50.0)",
    )
    parser.add_argument(
        "--d-min",
        type=float,
        default=0.0,
        help="Direct-contact distance clamp d_min (um) when --tunneling is set (default: 0.0)",
    )
    parser.add_argument(
        "--min-conductance",
        type=float,
        default=None,
        help="Minimum conductance cutoff G_min when --tunneling is set",
    )
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Skip automatic percolation threshold estimation",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating percolation probability plot",
    )
    args = parser.parse_args()

    orientation_override = None
    if args.orientation_mode is not None:
        orientation_override = OrientationConfig(
            mode=args.orientation_mode,
            axis=args.orientation_axis,
            max_angle_deg=args.max_angle,
            kappa=args.kappa,
        )

    tunneling_override = None
    if args.tunneling:
        tunneling_override = TunnelingConfig(
            enabled=True,
            g0=args.g0,
            alpha=args.alpha,
            d_min=args.d_min,
            min_conductance=args.min_conductance,
        )

    result = run_from_config(
        args.config,
        verbose=True,
        contact_method=args.contact_method,
        orientation=orientation_override,
        tunneling=tunneling_override,
    )

    out_dir = Path(args.output_dir)
    json_path = out_dir / "monte_carlo_results.json"
    csv_path = out_dir / "monte_carlo_results.csv"

    result.save_json(json_path)
    result.save_csv(csv_path)

    print("\nResults successfully saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")

    if not args.no_threshold:
        from percolation.analysis.threshold import estimate_threshold
        from percolation.visualization.threshold_plot import plot_percolation_probability

        thresh = estimate_threshold(result)
        thresh_path = out_dir / "threshold_results.json"
        thresh.save_json(thresh_path)

        print("\n" + "=" * 70)
        print("  Percolation Threshold Analysis (Phase 3)")
        print("=" * 70)
        print(f"Primary Method        : {thresh.method}")
        if thresh.phi_c is not None:
            if thresh.logistic_phi_c_stderr is not None:
                print(f"Estimated phi_c       : {thresh.phi_c:.6f} +/- {thresh.logistic_phi_c_stderr:.6f}")
            else:
                print(f"Estimated phi_c       : {thresh.phi_c:.6f}")
        else:
            print("Estimated phi_c       : Not reached within simulated range")

        if thresh.phi_c_linear is not None:
            print(f"Linear Interp phi_c   : {thresh.phi_c_linear:.6f}")

        if thresh.logistic_fit is not None and thresh.logistic_fit.converged:
            print(f"Logistic Steepness k  : {thresh.logistic_fit.k:.2f}")
            print(f"Logistic R^2          : {thresh.logistic_fit.r_squared:.4f}")

        print(f"Saved threshold data  : {thresh_path}")

        if not args.no_plot:
            plot_path = out_dir / "percolation_probability.png"
            plot_percolation_probability(
                mc_result=result,
                threshold_result=thresh,
                save_path=plot_path,
            )
            print(f"Saved P(phi) plot     : {plot_path}")

        print("=" * 70)


if __name__ == "__main__":
    main()

# CNT Composite Percolation Simulator

3D Monte Carlo simulator for electrical percolation in Carbon Nanotube (CNT) composites.

---

## 1. Physical Problem & Purpose

The simulator investigates electrical percolation in polymer-nanotube composite materials:
* **Domain**: 3D rectangular matrix box ($L_x \times L_y \times L_z$). Opposing electrodes are located at $x = 0$ (**LEFT**) and $x = L_x$ (**RIGHT**). Current transport is studied along the X axis.
* **Geometry**: Carbon nanotubes (CNT) are modeled as finite 3D line segments with length $L$, diameter $D$, random centers, and isotropic uniform spherical 3D orientations.
* **Contacts**: An electrical junction between two CNTs forms when their minimum 3D segment distance $d(A, B) \le d_{\text{contact}}$.
* **Percolation**: Evaluated via Breadth-First Search (BFS) on the contact graph. A state percolates if a continuous conductive path exists connecting the `LEFT` electrode to the `RIGHT` electrode.
* **Percolation Threshold ($\phi_c$)**: The critical volumetric concentration where percolation probability $P(\phi_c) \approx 0.5$.

---

## 2. Project Status & Roadmap

| Phase | Milestone | Status | Description |
|:-----:|:----------|:------:|:------------|
| **1** | Minimal Percolation Model | ✅ Complete | 3D line segment geometry, contact detection, graph construction, BFS connectivity (`LEFT` $\to$ `RIGHT`). |
| **2** | Monte Carlo Engine | ✅ Complete | Multi-sample runs over concentration range, seed reproducibility (`SeedSequence`), statistical aggregation. |
| **3** | Percolation Threshold ($\phi_c$) | ✅ Complete | Linear interpolation, logistic sigmoid fitting ($R^2$, stderr), Wilson score confidence intervals, $P(\phi)$ curve plotting. |
| **4** | 3D Visualization | ✅ Complete | 3D visualization of material box, electrodes, CNT network, contacts, and percolating cluster. |
| **5** | Spatial Optimization | ✅ Complete | Uniform 3D spatial grid (`UniformSpatialGrid`), sparse hashing (`SpatialHashGrid`), KD-tree (`KDTreeIndex`), and vectorized segment distance (`batch_segment_distance`) replacing $O(N^2)$ brute force with $>15\times$ speedup and zero false negatives. |
| **6** | Aspect Ratio Study | ✅ Complete | Comparative Monte Carlo scaling study $\phi_c(AR) = C \cdot AR^{-\beta}$ confirming Balberg's excluded-volume law ($\beta \approx 1$), JSON/CSV export, and 2-panel publication plot. |
| **7** | CNT Orientation Study | ✅ Complete | Configurable 3D orientation distributions (`isotropic`, `aligned`, `partially_aligned` cone, `fisher`), 3D Hermans order parameter $S_x$, adaptive $\phi_c(S_x)$ study, 2-panel visualization, and CLI/YAML support. |
| **8** | Tunneling Contact Conductance | ✅ Complete | Exponential tunneling model $G_{ij}=G_0 e^{-\alpha d_{\text{eff}}}$, $R_{ij}=1/G_{ij}$, Simmons barrier helper, conductance-weighted contact graph, Dijkstra min-resistance path, AHL bottleneck conductance $G_c$, parametric $\alpha$ / $d_{\text{contact}}$ studies, and 4-panel visualization. |

---

## 3. Installation & Setup

Requirements: **Python $\ge$ 3.11**

```bash
# 1. Clone repository and navigate to root
cd cnt-percolation

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `pyyaml`, and `pytest` for tests.

---

## 4. Usage Guide & CLI Commands

### Command A: Full Monte Carlo Simulation + Threshold Analysis + Plot

Runs Monte Carlo realizations across all concentrations defined in the configuration file, calculates percolation probabilities, estimates $\phi_c$, and outputs publication-ready plots.

```bash
.venv\Scripts\python.exe -m percolation.simulation.monte_carlo --config config.yaml --output-dir results
```

**CLI Flags:**
* `-c, --config PATH`: Path to YAML configuration file (default: `config.yaml`).
* `-o, --output-dir PATH`: Directory where output files are saved (default: `results`).
* `--contact-method`: Acceleration backend: `auto`, `grid`, `hash`, `kdtree`, `brute_force` (default: from config or `auto`).
* `--no-threshold`: Skip automatic threshold estimation.
* `--no-plot`: Skip generating the $P(\phi)$ plot.

**Generated Outputs in `results/`:**
* `monte_carlo_results.json`: Full structured output containing parameters, averages, and sample statistics.
* `monte_carlo_results.csv`: Tabular data ($N_{\text{tubes}}$, contacts, percolating count, $P(\phi)$) per concentration.
* `threshold_results.json`: Estimated threshold $\phi_c$, logistic fit parameters ($k$, $R^2$, stderr), and Wilson confidence intervals.
* `percolation_probability.png`: Plot displaying empirical probabilities with 95% Wilson confidence error bars, the fitted logistic curve, and threshold markers.

---

### Command B: Standalone Threshold Estimation & Plotting

To analyze or re-plot previously saved Monte Carlo data without re-running the simulation:

```bash
.venv\Scripts\python.exe -m percolation.analysis.threshold --input results/monte_carlo_results.json --output results/threshold_results.json --plot results/percolation_probability.png --confidence 0.95
```

**CLI Flags:**
* `-i, --input PATH`: Path to input `monte_carlo_results.json` (**required**).
* `-o, --output PATH`: Path to write calculated `threshold_results.json`.
* `-p, --plot PATH`: Path to save the percolation curve plot (`.png` / `.pdf` / `.svg`).
* `--confidence FLOAT`: Confidence level for error bars (default: `0.95` for 95% CI).

---

### Command C: 3D Visualization of Realized CNT Sample (Phase 4)

Generates and renders a full 3D spatial realization of the CNT network, distinguishing isolated tubes, the percolating cluster, conductive backbone path, electrodes, and contact junctions:

```bash
.venv\Scripts\python.exe -m percolation.visualization.renderer --phi 0.005 --seed 42 --output results/sample_3d.png
```

**CLI Flags:**
* `--phi FLOAT`: Volumetric CNT fraction $\phi$ (default: `0.005`).
* `--seed INT`: Random seed (default: `42`).
* `-o, --output PATH`: Path to save rendered 3D scene (default: `results/sample_3d.png`).
* `--length FLOAT`: CNT length $L$ in $\mu\text{m}$ (default: `1.0`).
* `--diameter FLOAT`: CNT diameter $D$ in $\mu\text{m}$ (default: `0.1`).
* `--threshold FLOAT`: Contact distance threshold in $\mu\text{m}$ (default: `0.05`).
* `--dpi INT`: Output resolution in DPI (default: `300`).

---

### Command D: Spatial Optimization Benchmark (Phase 5)

Runs a side-by-side benchmark verifying that all spatial partitioning algorithms (`grid`, `hash`, `kdtree`) match `brute_force` 100% while reducing candidate pairs by $>95\%$:

```bash
.venv\Scripts\python.exe -m percolation.geometry.spatial --count 1000 --threshold 0.05
```

Example Benchmark Output ($N = 1000$ CNTs, 499,500 pairs):
```text
Method         |  Time (ms) |   Candidates |  Reduction |  Contacts |   Speedup
--------------------------------------------------------------------------------
brute_force    |     527.65 |      499,500 |       0.0% |       246 |      1.0x
grid           |      39.39 |       17,562 |      96.5% |       246 |     13.4x
hash           |      38.24 |       17,562 |      96.5% |       246 |     13.8x
kdtree         |      28.71 |       10,318 |      97.9% |       246 |     18.4x
================================================================================
All methods verified: 100% identical contacts identified!
```

---

### Command E: Single Realization (Smoke Test & Quick Check)

Runs a single sample with detailed metrics, spatial method choice, orientation options, and optional 3D visualization:

```bash
.venv\Scripts\python.exe -m percolation.simulation.experiment --phi 0.005 --contact-method grid --orientation-mode partially_aligned --max-angle 30 --visualize results/sample_3d.png
```

---

### Command F: Aspect Ratio Scaling Study (Phase 6)

Evaluates percolation threshold scaling $\phi_c(AR)$ across multiple aspect ratios and fits Balberg's power law $\phi_c = C \cdot AR^{-\beta}$:

```bash
.venv\Scripts\python.exe -m percolation.analysis.aspect_ratio --ar-values 15,25,40,60,100 --samples 25 --output-dir results
```

---

### Command G: CNT Orientation Study (Phase 7)

Investigates the dependence of the percolation threshold $\phi_c$ on 3D CNT alignment (from transverse and isotropic through partial cone alignment to strictly aligned along the transport axis):

```bash
.venv\Scripts\python.exe -m percolation.analysis.orientation --angles 0,15,30,45,60,75,90 --include-transverse --samples 25 --output-dir results
```

**CLI Flags:**
* `--angles LIST`: Comma-separated cone half-angles in degrees `[0..90]` (default: `0,15,30,45,60,75,90`).
* `--axis AXIS`: Preferred alignment axis: `x`, `y`, or `z` (default: `x`).
* `--include-transverse`: Also evaluate a transverse-aligned baseline ($S_x = -0.5$).
* `--samples INT`: Monte Carlo realizations per concentration point (default: `25`).
* `--n-concentrations INT`: Concentration grid points per orientation (default: `7`).
* `--cnt-length`, `--cnt-diameter`, `--threshold`, `--box-size`, `--seed`, `--contact-method`, `--output-dir`, `--no-plot`, `--dpi`.

**Generated Outputs in `results/`:**
* `orientation_study.json` & `orientation_study.csv`: Threshold $\phi_c$, Hermans order parameter ($S_{\text{theory}}$, $S_{\text{measured}}$), and ratio $\phi_c / \phi_c^{\text{iso}}$ for each orientation.
* `orientation_study.png`: Two-panel publication plot showing $P(\phi)$ S-curves and $\phi_c(S_x)$.

---

### Command H: Tunneling Contact Conductance Study (Phase 8)

Investigates exponential quantum tunneling contacts $G_{ij} = G_0 e^{-\alpha \max(0, d_{ij} - d_{\min})}$, $R_{ij} = 1/G_{ij}$, sweeping either the tunneling decay constant $\alpha$ (`--mode alpha`) or the contact cutoff distance $d_{\text{contact}}$ (`--mode cutoff`), computing $\phi_c$, Dijkstra minimum-resistance paths, and Ambegaokar-Halperin-Langer bottleneck conductance $G_c$:

```bash
# Sweep tunneling decay constant alpha (at fixed G_min)
.venv\Scripts\python.exe -m percolation.analysis.tunneling --mode alpha --alpha-values 30,50,80,120,180 --min-conductance 0.01 --samples 20 --output-dir results

# Sweep contact cutoff distance d_contact (Experiment 3)
.venv\Scripts\python.exe -m percolation.analysis.tunneling --mode cutoff --cutoff-values 0.01,0.03,0.05,0.08,0.12 --alpha 50 --samples 20 --output-dir results
```

**Generated Outputs in `results/`:**
* `tunneling_study.json` & `tunneling_study.csv`: Threshold $\phi_c$, effective cutoff $d_{\text{eff}}$, peak bottleneck conductance $G_c$, and minimum series path resistance per configuration.
* `tunneling_study.png`: Four-panel publication plot showing $G(d)/G_0$, $P(\phi)$ S-curves, $\phi_c$ trend, and bottleneck conductance $G_c(\phi)$.

---

## 5. Python API Usage

You can also use the simulator directly in Python scripts or Jupyter notebooks:

### Running Monte Carlo Study & Estimating Threshold

```python
from percolation.domain.cnt import MaterialBox
from percolation.simulation.monte_carlo import run_monte_carlo
from percolation.visualization.threshold_plot import plot_percolation_probability

# 1. Define physical domain
box = MaterialBox(size_x=10.0, size_y=5.0, size_z=5.0)

# 2. Run Monte Carlo simulation with spatial acceleration
result = run_monte_carlo(
    concentrations=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
    samples_per_concentration=50,
    box=box,
    cnt_length=1.0,
    cnt_diameter=0.1,
    contact_threshold=0.05,
    random_seed=42,
    contact_method="grid",  # 'grid', 'hash', 'kdtree', or 'auto'
)

# 3. Estimate percolation threshold
thresh = result.estimate_threshold(confidence_level=0.95)
print(f"Percolation threshold phi_c: {thresh.phi_c:.5f}")
print(f"Logistic steepness k       : {thresh.logistic_k:.1f}")
print(f"Logistic R^2               : {thresh.logistic_r2:.4f}")

# 4. Save visualization plot
fig = plot_percolation_probability(result, threshold_result=thresh, save_path="results/my_percolation_curve.png")
```

### Spatial Grid Benchmark in Python

```python
import numpy as np
from percolation.domain.cnt import MaterialBox
from percolation.generation.cnt_generator import generate_nanotubes
from percolation.geometry.spatial import compare_contact_methods

box = MaterialBox(10.0, 5.0, 5.0)
rng = np.random.default_rng(42)
tubes = generate_nanotubes(1000, box, cnt_length=1.0, cnt_diameter=0.1, rng=rng)

# Cross-verifies contacts and benchmarks all methods
bench = compare_contact_methods(tubes, threshold=0.05, box=box)
for name, res in bench.items():
    print(f"{name}: {res.execution_time_sec*1000:.1f} ms, speedup {res.speedup_vs_brute_force:.1f}x")
```

---

## 6. Configuration Guide (`config.yaml`)

Simulation parameters are stored in `config.yaml`:

```yaml
box:
  x: 10.0       # Domain size along X (distance between electrodes), um
  y: 5.0        # Domain size along Y, um
  z: 5.0        # Domain size along Z, um

cnt:
  length: 1.0   # CNT length L, um
  diameter: 0.1 # CNT diameter D, um (Aspect Ratio AR = L / D = 10)
  orientation:
    mode: isotropic    # isotropic, aligned, partially_aligned, fisher
    axis: x            # preferred axis: x, y, z, or [dx, dy, dz]
    max_angle_deg: 90  # cone half-angle in degrees [0..90] (partially_aligned)
    kappa: 0.0         # concentration parameter >= 0 (fisher)

contact:
  threshold: 0.05 # Maximum segment-to-segment distance for electrical contact, um
  method: auto    # Acceleration: auto, grid, hash, kdtree, brute_force
  tunneling:
    enabled: false       # Activate Phase 8 exponential tunneling model
    g0: 1.0              # Direct-contact conductance G0 (S or normalized)
    alpha: 50.0          # Tunneling decay constant alpha (1/um)
    d_min: 0.0           # Direct-contact / van der Waals distance clamp (um)
    cutoff_distance: null
    min_conductance: null

simulation:
  concentrations: # Volumetric CNT fractions phi to evaluate
    - 0.001
    - 0.002
    - 0.003
    - 0.004
    - 0.005
    - 0.006
    - 0.007
    - 0.008
  samples_per_concentration: 100 # Number of independent realizations per phi
  random_seed: 42                # Master seed for SeedSequence hierarchy
```

---

## 7. Verification & Tests

To execute the test suite (geometry, generation, contact graphs, spatial acceleration, Monte Carlo, threshold analysis, 3D visualization, and plotting):

```bash
# Run pytest test suite
.venv\Scripts\python.exe -m pytest

# Run linter
.venv\Scripts\python.exe -m ruff check src tests
```

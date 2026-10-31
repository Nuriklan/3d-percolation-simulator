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
| **4** | 3D Visualization | ⏳ In progress | 3D visualization of material box, electrodes, CNT network, and percolating cluster. |
| **5** | Spatial Optimization | ⬜ Planned | Uniform spatial grid / partitioning to replace $O(N^2)$ contact search for large aspect ratios ($AR > 100$). |

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

### Command C: Single Realization (Smoke Test)

Runs a single sample with detailed contact metrics:

```bash
.venv\Scripts\python.exe -m percolation.simulation.experiment
```

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

# 2. Run Monte Carlo simulation
result = run_monte_carlo(
    concentrations=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
    samples_per_concentration=50,
    box=box,
    cnt_length=1.0,
    cnt_diameter=0.1,
    contact_threshold=0.05,
    random_seed=42,
)

# 3. Estimate percolation threshold
thresh = result.estimate_threshold(confidence_level=0.95)
print(f"Percolation threshold phi_c: {thresh.phi_c:.5f}")
print(f"Logistic steepness k       : {thresh.logistic_k:.1f}")
print(f"Logistic R^2               : {thresh.logistic_r2:.4f}")

# 4. Save visualization plot
fig = plot_percolation_probability(result, threshold_result=thresh, save_path="results/my_percolation_curve.png")
```

### Inspecting an Individual Sample & Finding the Conductive Path

```python
import numpy as np
from percolation.domain.cnt import MaterialBox
from percolation.simulation.experiment import run_sample
from percolation.graph.contact_graph import build_graph
from percolation.graph.percolation import find_percolating_path

box = MaterialBox(size_x=3.0, size_y=2.0, size_z=2.0)
rng = np.random.default_rng(1234)

sample = run_sample(
    concentration=0.04,
    box=box,
    cnt_length=1.0,
    cnt_diameter=0.05,
    contact_threshold=0.05,
    rng=rng,
)

print(f"Placed {sample.nanotube_count} CNTs, found {sample.contact_count} contacts.")
print(f"Does the system percolate? -> {sample.percolates}")
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

contact:
  threshold: 0.05 # Maximum segment-to-segment distance for electrical contact, um

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

To execute the test suite (geometry, generation, contact graphs, Monte Carlo, threshold analysis, and plotting):

```bash
# Run pytest test suite
.venv\Scripts\python.exe -m pytest

# Run linter
.venv\Scripts\python.exe -m ruff check src tests
```

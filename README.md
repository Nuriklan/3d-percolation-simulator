# CNT Percolation Simulator

3D Monte Carlo simulator for electrical percolation in Carbon Nanotube (CNT) composites.

## Physical Problem

Determines the minimum CNT concentration at which a continuous conductive path
forms between opposing electrodes (percolation threshold φ_c).

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Minimal percolation model (3D CNT, contact detection, BFS) | ✅ In progress |
| 2 | Monte Carlo simulation, P(φ) | ⬜ |
| 3 | Percolation threshold φ_c estimation | ⬜ |
| 4 | 3D Visualization | ⬜ |
| 5 | Spatial optimization (O(N²) → grid) | ⬜ |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Run

```bash
python -m percolation.simulation.experiment
```

## Test

```bash
pytest
```

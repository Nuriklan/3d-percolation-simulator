"""Simulation sub-package."""

from percolation.simulation.experiment import run_sample

__all__ = [
    "run_sample",
    "run_concentration",
    "run_monte_carlo",
    "run_from_config",
]


def __getattr__(name: str):
    if name in {"run_concentration", "run_monte_carlo", "run_from_config"}:
        import percolation.simulation.monte_carlo as mc

        return getattr(mc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Visualization sub-package for CNT percolation simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from percolation.visualization.threshold_plot import plot_percolation_probability

__all__ = [
    "plot_percolation_probability",
]


def __getattr__(name: str):
    if name in __all__:
        import percolation.visualization.threshold_plot as _mod
        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Visualization sub-package for CNT percolation simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from percolation.visualization.aspect_ratio_plot import plot_aspect_ratio_study
    from percolation.visualization.renderer import plot_sample_3d
    from percolation.visualization.threshold_plot import plot_percolation_probability

__all__ = [
    "plot_aspect_ratio_study",
    "plot_percolation_probability",
    "plot_sample_3d",
]


def __getattr__(name: str):
    if name == "plot_sample_3d":
        import percolation.visualization.renderer as _ren
        return getattr(_ren, name)
    if name == "plot_percolation_probability":
        import percolation.visualization.threshold_plot as _thresh
        return getattr(_thresh, name)
    if name == "plot_aspect_ratio_study":
        import percolation.visualization.aspect_ratio_plot as _ar_plot
        return getattr(_ar_plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

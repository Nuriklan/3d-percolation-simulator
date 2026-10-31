"""
Analysis sub-package for CNT percolation simulation (Phase 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from percolation.analysis.threshold import (
        ConfidenceInterval,
        LogisticFitResult,
        ThresholdResult,
        compute_confidence_intervals,
        estimate_threshold,
        estimate_threshold_linear,
        fit_logistic,
        logistic_function,
        wilson_score_interval,
    )

__all__ = [
    "ConfidenceInterval",
    "LogisticFitResult",
    "ThresholdResult",
    "compute_confidence_intervals",
    "estimate_threshold",
    "estimate_threshold_linear",
    "fit_logistic",
    "logistic_function",
    "wilson_score_interval",
]


def __getattr__(name: str):
    if name in __all__:
        import percolation.analysis.threshold as _mod
        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

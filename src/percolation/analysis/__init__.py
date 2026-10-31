"""
Analysis sub-package for CNT percolation simulation (Phase 3 & 6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from percolation.analysis.aspect_ratio import (
        AspectRatioPoint,
        AspectRatioStudyResult,
        PowerLawFitResult,
        fit_power_law,
        run_aspect_ratio_study,
    )
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
    "AspectRatioPoint",
    "AspectRatioStudyResult",
    "ConfidenceInterval",
    "LogisticFitResult",
    "PowerLawFitResult",
    "ThresholdResult",
    "compute_confidence_intervals",
    "estimate_threshold",
    "estimate_threshold_linear",
    "fit_logistic",
    "fit_power_law",
    "logistic_function",
    "run_aspect_ratio_study",
    "wilson_score_interval",
]


def __getattr__(name: str):
    if name in {
        "AspectRatioPoint",
        "AspectRatioStudyResult",
        "PowerLawFitResult",
        "fit_power_law",
        "run_aspect_ratio_study",
    }:
        import percolation.analysis.aspect_ratio as _ar_mod
        return getattr(_ar_mod, name)
    if name in __all__:
        import percolation.analysis.threshold as _mod
        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Kernel localization and bandwidth rules in standardized prediction space.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Kernel = Literal["gaussian", "epanechnikov", "uniform"]

DEFAULT_CANDIDATE_CAP = 10_000

DEFAULT_GAUSSIAN_CUTOFF = 5.0

def _validate_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value

def _kernel_values(distance: np.ndarray, kernel: Kernel) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)

    if kernel == "gaussian":
        return np.exp(-0.5 * distance ** 2)
    if kernel == "epanechnikov":
        return np.maximum(1.0 - distance ** 2, 0.0)
    if kernel == "uniform":
        return (distance <= 1.0).astype(float)

    raise ValueError(f"Unknown kernel: {kernel!r}")

def _bandwidth_from_quantile(
    raw_distance: np.ndarray,
    quantile: float,
) -> float:
    quantile = float(quantile)
    if not 0 < quantile <= 1:
        raise ValueError("quantile must lie in (0, 1].")
    return max(float(np.quantile(raw_distance, quantile)), 1e-12)

def _bandwidth_silverman_rate(
    raw_distance: np.ndarray,
    dimension: int,
    count_constant: float,
) -> tuple[float, int]:
    count_constant = _validate_positive(
        "shrinking_count_constant",
        count_constant,
    )

    n = int(raw_distance.size)
    d = int(max(1, dimension))
    m = int(np.ceil(count_constant * n ** (4.0 / (d + 4.0))))
    m = int(min(max(m, d + 2), n))

    if m >= n:
        bandwidth = float(np.max(raw_distance))
    else:
        bandwidth = float(np.partition(raw_distance, m - 1)[m - 1])

    return max(bandwidth, 1e-12), m

def _active_support_count(
    sorted_distance: np.ndarray,
    bandwidth: float,
    kernel: Kernel,
    gaussian_cutoff: float,
) -> int:
    if kernel == "gaussian":
        radius = gaussian_cutoff * bandwidth
        return int(np.searchsorted(sorted_distance, radius, side="right"))
    if kernel == "uniform":
        return int(np.searchsorted(sorted_distance, bandwidth, side="right"))
    return int(np.searchsorted(sorted_distance, bandwidth, side="left"))

def _normalized_kernel_weights(
    raw_distance: np.ndarray,
    bandwidth: float,
    kernel: Kernel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bandwidth = _validate_positive("bandwidth", bandwidth)

    scaled_distance = raw_distance / bandwidth
    scores = _kernel_values(scaled_distance, kernel)
    positive = scores > 0

    if not np.any(positive):
        raise ValueError(
            "The chosen localization has no positive-weight observations."
        )

    score_sum = float(np.sum(scores[positive]))
    if not np.isfinite(score_sum) or score_sum <= 0:
        raise ValueError("Kernel scores could not be normalized.")

    weights = np.zeros_like(scores, dtype=float)
    weights[positive] = scores[positive] / score_sum
    return weights, scaled_distance, scores

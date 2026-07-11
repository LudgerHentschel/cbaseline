"""
Prediction-space geometry shared by all background constructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim == 2:
        return values
    raise ValueError("values must be one- or two-dimensional.")

@dataclass(frozen=True)
class PredictionMetric:
    """Nonredundant whitening map for prediction gaps."""

    center: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    discarded_eigenvectors: np.ndarray
    discarded_eigenvalues: np.ndarray
    ridge: float
    original_dimension: int
    eigen_threshold: float

    @property
    def dimension(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def discarded_dimension(self) -> int:
        return int(self.discarded_eigenvalues.size)

    def coordinates(self, gaps: np.ndarray) -> np.ndarray:
        gaps = _as_2d(gaps)
        if gaps.shape[1] != self.original_dimension:
            raise ValueError(
                f"Expected gap dimension {self.original_dimension}; "
                f"received {gaps.shape[1]}."
            )
        scale = np.sqrt(self.eigenvalues + self.ridge)
        return np.dot(gaps, self.eigenvectors) / scale

    def squared_norm(self, gaps: np.ndarray) -> np.ndarray:
        U = self.coordinates(gaps)
        return np.sum(U ** 2, axis=1)

    def norm(self, gap: np.ndarray) -> float:
        return float(
            np.sqrt(
                self.squared_norm(
                    np.asarray(gap, dtype=float).reshape(1, -1)
                )[0]
            )
        )

    def affine_support_residual(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=float).reshape(1, -1)
        if target.shape[1] != self.original_dimension:
            raise ValueError(
                f"Expected target dimension {self.original_dimension}; "
                f"received {target.shape[1]}."
            )

        if self.discarded_dimension == 0:
            return np.zeros(self.original_dimension, dtype=float)

        difference = target - self.center.reshape(1, -1)
        coeff = np.dot(difference, self.discarded_eigenvectors)
        residual = np.dot(coeff, self.discarded_eigenvectors.T)
        return residual.ravel()

def fit_prediction_metric(
    predictions: np.ndarray,
    *,
    covariance: Optional[np.ndarray] = None,
    ridge: float = 1e-6,
    eigen_tol: float = 1e-10,
) -> PredictionMetric:
    """Fit a whitening metric and remove redundant output directions."""
    ridge = float(ridge)
    eigen_tol = float(eigen_tol)

    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative.")
    if eigen_tol <= 0 or not np.isfinite(eigen_tol):
        raise ValueError("eigen_tol must be finite and positive.")

    F = _as_2d(predictions)
    if F.shape[0] < 2:
        raise ValueError("At least two prediction rows are required.")
    if not np.all(np.isfinite(F)):
        raise ValueError("predictions must be finite.")

    k = F.shape[1]
    center = F.mean(axis=0)

    if covariance is None:
        if k == 1:
            variance = float(np.var(F[:, 0], ddof=1))
            covariance = np.array(
                [[variance if variance > 0 else 1.0]],
                dtype=float,
            )
        else:
            covariance = np.cov(F, rowvar=False)

    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    if covariance.shape != (k, k):
        raise ValueError(
            f"covariance must have shape {(k, k)}; "
            f"received {covariance.shape}."
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite.")

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    largest = float(max(np.max(eigenvalues), 0.0))
    threshold = max(largest * eigen_tol, 1e-14)
    keep = eigenvalues > threshold

    if not np.any(keep):
        raise ValueError(
            "Prediction covariance has no nonredundant positive-variance "
            "direction."
        )

    kept_values = eigenvalues[keep]
    kept_vectors = eigenvectors[:, keep]
    discarded_values = eigenvalues[~keep]
    discarded_vectors = eigenvectors[:, ~keep]
    ridge_abs = ridge * max(float(np.mean(kept_values)), 1e-12)

    return PredictionMetric(
        center=center,
        eigenvectors=kept_vectors,
        eigenvalues=kept_values,
        discarded_eigenvectors=discarded_vectors,
        discarded_eigenvalues=discarded_values,
        ridge=ridge_abs,
        original_dimension=k,
        eigen_threshold=threshold,
    )

def _validate_target_affine_support(
    metric: PredictionMetric,
    target: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[np.ndarray, float, float]:
    if atol < 0 or not np.isfinite(atol):
        raise ValueError("affine_support_atol must be finite and nonnegative.")
    if rtol < 0 or not np.isfinite(rtol):
        raise ValueError("affine_support_rtol must be finite and nonnegative.")

    residual = metric.affine_support_residual(target)
    residual_norm = float(np.linalg.norm(residual))
    scale = max(
        1.0,
        float(np.linalg.norm(metric.center)),
        float(np.linalg.norm(target)),
        float(np.sqrt(np.max(metric.eigenvalues))),
    )
    tolerance = float(atol + rtol * scale)

    if residual_norm > tolerance:
        raise ValueError(
            "f0 does not lie in the affine output support represented by the "
            "reference predictions. Its residual in discarded output "
            f"directions is {residual_norm:.3e}, exceeding the support "
            f"tolerance {tolerance:.3e}. For multiclass use, pass valid "
            "centered logits for both predictions and f0."
        )

    return residual, residual_norm, tolerance

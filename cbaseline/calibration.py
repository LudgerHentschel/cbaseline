"""
Exponential calibration of localized empirical background weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .kernels import _validate_positive

@dataclass(frozen=True)
class CalibrationResult:
    weights: np.ndarray
    success: bool
    optimizer_success: bool
    message: str
    iterations: int
    residual_coordinates: np.ndarray
    residual_norm: float
    objective: float

def exponential_calibration(
    coordinates: np.ndarray,
    kernel_weights: np.ndarray,
    *,
    tolerance: float = 1e-10,
    maxiter: int = 2000,
) -> CalibrationResult:
    """Exponentially tilt kernel weights so their mean coordinate is zero."""
    tolerance = _validate_positive("calibration_tolerance", tolerance)
    maxiter = int(maxiter)
    if maxiter < 1:
        raise ValueError("calibration_maxiter must be at least 1.")

    U = np.asarray(coordinates, dtype=float)
    q = np.asarray(kernel_weights, dtype=float).ravel()

    if U.ndim != 2:
        raise ValueError("coordinates must be two-dimensional.")
    if q.shape != (U.shape[0],):
        raise ValueError("kernel_weights must align with coordinates.")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(q)):
        raise ValueError("coordinates and kernel_weights must be finite.")
    if np.any(q < 0):
        raise ValueError("kernel_weights must be nonnegative.")

    positive = q > 0
    if np.sum(positive) < U.shape[1] + 1:
        residual = np.dot(q, U)
        return CalibrationResult(
            weights=q.copy(),
            success=False,
            optimizer_success=False,
            message=(
                "Too few positive-kernel observations for calibration in "
                f"{U.shape[1]} dimensions."
            ),
            iterations=0,
            residual_coordinates=residual,
            residual_norm=float(np.linalg.norm(residual)),
            objective=np.nan,
        )

    U_pos = U[positive]
    q_pos = q[positive]
    q_pos = q_pos / q_pos.sum()
    logq = np.log(q_pos)

    def objective(lam: np.ndarray) -> float:
        return float(logsumexp(logq + np.dot(U_pos, lam)))

    def gradient(lam: np.ndarray) -> np.ndarray:
        a = logq + np.dot(U_pos, lam)
        w_pos = np.exp(a - logsumexp(a))
        return np.dot(w_pos, U_pos)

    def hessian(lam: np.ndarray) -> np.ndarray:
        a = logq + np.dot(U_pos, lam)
        w_pos = np.exp(a - logsumexp(a))
        mean_u = np.dot(w_pos, U_pos)
        centered = U_pos - mean_u
        return np.dot((centered * w_pos[:, None]).T, centered)

    result = minimize(
        objective,
        np.zeros(U.shape[1], dtype=float),
        method="trust-exact",
        jac=gradient,
        hess=hessian,
        options={"gtol": tolerance, "maxiter": maxiter},
    )

    if not result.success:
        retry = minimize(
            objective,
            np.asarray(result.x, dtype=float),
            method="BFGS",
            jac=gradient,
            options={"gtol": tolerance, "maxiter": maxiter},
        )
        if (
            retry.success
            or np.linalg.norm(gradient(retry.x))
            < np.linalg.norm(gradient(result.x))
        ):
            result = retry

    a = logq + np.dot(U_pos, result.x)
    w_pos = np.exp(a - logsumexp(a))

    weights = np.zeros_like(q)
    weights[positive] = w_pos

    residual = np.dot(weights, U)
    residual_norm = float(np.linalg.norm(residual))

    success = bool(
        np.isfinite(residual_norm)
        and residual_norm <= tolerance
        and np.all(np.isfinite(weights))
        and float(np.sum(weights)) > 0
    )

    message = str(result.message)
    if not success:
        message = (
            f"{message}; calibration residual {residual_norm:.3e} exceeds "
            f"tolerance {tolerance:.3e} or produced invalid weights."
        )

    return CalibrationResult(
        weights=weights,
        success=success,
        optimizer_success=bool(result.success),
        message=message,
        iterations=int(getattr(result, "nit", 0)),
        residual_coordinates=residual,
        residual_norm=residual_norm,
        objective=float(result.fun),
    )

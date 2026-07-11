"""
Kernel-weighted prediction-neutral backgrounds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .calibration import CalibrationResult, exponential_calibration
from .geometry import (
    PredictionMetric,
    _as_2d,
    _validate_target_affine_support,
    fit_prediction_metric,
)
from .kernels import (
    DEFAULT_CANDIDATE_CAP,
    DEFAULT_GAUSSIAN_CUTOFF,
    Kernel,
    _active_support_count,
    _bandwidth_from_quantile,
    _bandwidth_silverman_rate,
    _kernel_values,
    _normalized_kernel_weights,
    _validate_positive,
)

@dataclass
class WeightedBackground:
    """Observed rows with kernel and calibrated localization weights."""

    index: np.ndarray
    X: np.ndarray
    predictions: np.ndarray
    f0: np.ndarray | float
    kernel_weights: np.ndarray
    calibrated_weights: np.ndarray
    metric: PredictionMetric
    bandwidth: float
    kernel: Kernel
    diagnostics: dict = field(default_factory=dict)

    def weighted(
        self,
        *,
        calibrated: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights = (
            self.calibrated_weights
            if calibrated
            else self.kernel_weights
        )
        return self.X, weights

    def weighted_index(self) -> np.ndarray:
        return self.index

    def resampled(
        self,
        n_draws: int,
        *,
        calibrated: bool = True,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        n_draws = int(n_draws)
        if n_draws <= 0:
            raise ValueError("n_draws must be positive.")

        weights = (
            self.calibrated_weights
            if calibrated
            else self.kernel_weights
        )
        rng = np.random.default_rng(random_state)
        draw = rng.choice(
            self.X.shape[0],
            size=n_draws,
            replace=True,
            p=weights,
        )
        return self.X[draw]

    def __repr__(self) -> str:
        d = self.diagnostics
        return (
            "WeightedBackground("
            f"rows={self.X.shape[0]}, "
            f"dimension={self.metric.dimension}, "
            f"bandwidth={self.bandwidth:.4g}, "
            f"kernel_ess={d.get('kernel_ess', np.nan):.1f}, "
            f"calibrated_ess={d.get('calibrated_ess', np.nan):.1f}, "
            f"neutrality={d.get('calibrated_neutrality_norm', np.nan):.2e}"
            ")"
        )

def kernel_weighted_background(
    predictions: np.ndarray,
    f0: np.ndarray | float,
    features: np.ndarray,
    *,
    kernel: Kernel = "gaussian",
    bandwidth: Optional[float] = None,
    quantile: Optional[float] = None,
    shrinking_count_constant: float = 0.5,
    metric: Optional[PredictionMetric] = None,
    candidate_size: Optional[int] = None,
    covariance: Optional[np.ndarray] = None,
    metric_ridge: float = 1e-6,
    eigen_tol: float = 1e-10,
    affine_support_atol: float = 1e-10,
    affine_support_rtol: float = 1e-8,
    calibrate: bool = True,
    calibration_tolerance: float = 1e-10,
    calibration_raw_tolerance: Optional[float] = None,
    calibration_maxiter: int = 2000,
    adaptive_widen: bool = True,
    widening_factor: float = 1.5,
    max_bandwidth: Optional[float] = None,
    max_widening_steps: int = 12,
    gaussian_cutoff: float = DEFAULT_GAUSSIAN_CUTOFF,
) -> WeightedBackground:
    """Construct a kernel-weighted background near a reference output."""
    if kernel not in {"gaussian", "epanechnikov", "uniform"}:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    gaussian_cutoff = _validate_positive(
        "gaussian_cutoff",
        gaussian_cutoff,
    )
    calibration_tolerance = _validate_positive(
        "calibration_tolerance",
        calibration_tolerance,
    )
    widening_factor = float(widening_factor)
    if adaptive_widen and (
        not np.isfinite(widening_factor) or widening_factor <= 1.0
    ):
        raise ValueError(
            "widening_factor must be finite and greater than 1 when "
            "adaptive_widen=True."
        )

    max_widening_steps = int(max_widening_steps)
    if max_widening_steps < 0:
        raise ValueError("max_widening_steps must be nonnegative.")

    calibration_maxiter = int(calibration_maxiter)
    if calibration_maxiter < 1:
        raise ValueError("calibration_maxiter must be at least 1.")

    F_original = np.asarray(predictions, dtype=float)
    F = _as_2d(F_original)
    X = np.asarray(features, dtype=float)
    target = np.asarray(f0, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    if F.shape[0] != X.shape[0]:
        raise ValueError(
            "predictions and features must contain the same number of rows."
        )
    if F.shape[0] < 2:
        raise ValueError("At least two reference rows are required.")
    if target.size != F.shape[1]:
        raise ValueError(
            f"f0 must have dimension {F.shape[1]}; "
            f"received {target.size}."
        )
    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(X)):
        raise ValueError("predictions and features must be finite.")
    if not np.all(np.isfinite(target)):
        raise ValueError("f0 must be finite.")

    if metric is None:
        metric = fit_prediction_metric(
            F,
            covariance=covariance,
            ridge=metric_ridge,
            eigen_tol=eigen_tol,
        )
    elif metric.original_dimension != F.shape[1]:
        raise ValueError(
            f"Supplied metric expects prediction dimension "
            f"{metric.original_dimension}; received {F.shape[1]}."
        )

    support_residual, support_residual_norm, support_tolerance = (
        _validate_target_affine_support(
            metric,
            target,
            atol=float(affine_support_atol),
            rtol=float(affine_support_rtol),
        )
    )

    gaps = F - target
    U = metric.coordinates(gaps)
    raw_distance = np.sqrt(np.einsum("ij,ij->i", U, U))

    shrinking_target_count = None
    if bandwidth is not None:
        current_bandwidth = _validate_positive(
            "bandwidth",
            bandwidth,
        )
        initial_bandwidth_source = "user"
    elif quantile is not None:
        current_bandwidth = _bandwidth_from_quantile(
            raw_distance,
            float(quantile),
        )
        initial_bandwidth_source = f"quantile={float(quantile):.6g}"
    else:
        current_bandwidth, shrinking_target_count = (
            _bandwidth_silverman_rate(
                raw_distance,
                metric.dimension,
                float(shrinking_count_constant),
            )
        )
        initial_bandwidth_source = (
            f"silverman-rate(c={float(shrinking_count_constant):.6g}, "
            f"m={shrinking_target_count})"
        )

    if max_bandwidth is None:
        max_bandwidth_value = max(
            current_bandwidth,
            float(np.max(raw_distance)) + 1e-12,
        )
    else:
        max_bandwidth_value = _validate_positive(
            "max_bandwidth",
            max_bandwidth,
        )
        if max_bandwidth_value < current_bandwidth:
            raise ValueError(
                "max_bandwidth cannot be smaller than the initial bandwidth."
            )

    n_rows = int(F.shape[0])
    if candidate_size is None:
        cap = min(DEFAULT_CANDIDATE_CAP, n_rows)
    else:
        candidate_size = int(candidate_size)
        if candidate_size < metric.dimension + 2:
            raise ValueError(
                "candidate_size must be at least effective dimension + 2 "
                f"({metric.dimension + 2})."
            )
        cap = min(candidate_size, n_rows)

    distance_order = np.argsort(raw_distance, kind="stable")
    sorted_distance = raw_distance[distance_order]

    final_pool = None
    final_q = None
    final_scaled_distance = None
    calibration = None
    widening_steps = 0
    active_count = 0
    cap_binding = False

    for step in range(max_widening_steps + 1):
        active_count = _active_support_count(
            sorted_distance,
            current_bandwidth,
            kernel,
            gaussian_cutoff,
        )
        active_count = max(active_count, metric.dimension + 2)
        active_count = min(active_count, n_rows)

        pool_count = min(active_count, cap)
        cap_binding = bool(active_count > cap)
        pool = distance_order[:pool_count]

        U_pool = U[pool]
        raw_distance_pool = raw_distance[pool]

        q, scaled_distance, _ = _normalized_kernel_weights(
            raw_distance_pool,
            current_bandwidth,
            kernel,
        )

        if calibrate:
            calibration = exponential_calibration(
                U_pool,
                q,
                tolerance=calibration_tolerance,
                maxiter=calibration_maxiter,
            )
            success = calibration.success
        else:
            residual = np.dot(q, U_pool)
            calibration = CalibrationResult(
                weights=q.copy(),
                success=True,
                optimizer_success=True,
                message="Calibration disabled; direct kernel weights returned.",
                iterations=0,
                residual_coordinates=residual,
                residual_norm=float(np.linalg.norm(residual)),
                objective=np.nan,
            )
            success = True

        final_pool = pool
        final_q = q
        final_scaled_distance = scaled_distance
        widening_steps = step

        if success:
            break
        if not adaptive_widen:
            break
        if current_bandwidth >= max_bandwidth_value:
            break

        next_bandwidth = min(
            current_bandwidth * widening_factor,
            max_bandwidth_value,
        )
        if next_bandwidth <= current_bandwidth * (1.0 + 1e-14):
            break
        current_bandwidth = next_bandwidth

    if (
        final_pool is None
        or final_q is None
        or final_scaled_distance is None
        or calibration is None
    ):
        raise RuntimeError("Weighted-background construction failed internally.")

    if not np.all(np.isfinite(calibration.weights)):
        raise ValueError(
            "Calibration returned nonfinite weights. "
            f"Optimizer message: {calibration.message}"
        )
    if float(np.sum(calibration.weights)) <= 0:
        raise ValueError(
            "Calibration returned weights with nonpositive total mass. "
            f"Optimizer message: {calibration.message}"
        )

    positive = final_q > 0
    index = final_pool[positive]
    X_out = X[index]
    F_out = F[index]
    U_out = U[index]
    q_out = final_q[positive]
    w_out = calibration.weights[positive]

    q_out = q_out / q_out.sum()
    w_out = w_out / w_out.sum()

    kernel_mean = np.dot(q_out, F_out)
    calibrated_mean = np.dot(w_out, F_out)

    kernel_gap = kernel_mean - target
    calibrated_gap = calibrated_mean - target

    kernel_coord_gap = np.dot(q_out, U_out)
    calibrated_coord_gap = np.dot(w_out, U_out)

    raw_scale = max(
        1.0,
        float(np.max(np.std(F, axis=0, ddof=1))),
        float(np.max(np.abs(target))),
    )
    if calibration_raw_tolerance is None:
        raw_tolerance = max(
            1e-10,
            10.0 * calibration_tolerance * raw_scale,
        )
    else:
        raw_tolerance = _validate_positive(
            "calibration_raw_tolerance",
            calibration_raw_tolerance,
        )

    calibrated_raw_max_abs_gap = float(
        np.max(np.abs(calibrated_gap))
    )
    calibrated_neutrality_norm = float(
        np.linalg.norm(calibrated_coord_gap)
    )

    if calibrate and (
        not calibration.success
        or calibrated_neutrality_norm > calibration_tolerance
        or calibrated_raw_max_abs_gap > raw_tolerance
    ):
        cap_note = (
            " The candidate cap was binding; increase candidate_size."
            if cap_binding
            else ""
        )
        raise ValueError(
            "Could not calibrate the localized kernel weights to the requested "
            "neutrality tolerances. Widen the bandwidth, increase "
            "max_bandwidth/max_widening_steps, or inspect whether f0 lies in "
            "the convex hull of the available predictions."
            f"{cap_note} Coordinate residual: "
            f"{calibrated_neutrality_norm:.3e} "
            f"(tolerance {calibration_tolerance:.3e}); raw max-absolute gap: "
            f"{calibrated_raw_max_abs_gap:.3e} "
            f"(tolerance {raw_tolerance:.3e}). Optimizer message: "
            f"{calibration.message}"
        )

    positive_w = w_out > 0
    kl_calibrated_to_kernel = float(
        np.sum(
            w_out[positive_w]
            * np.log(
                w_out[positive_w] / q_out[positive_w]
            )
        )
    )

    kernel_ess = float(1.0 / np.sum(q_out ** 2))
    calibrated_ess = float(1.0 / np.sum(w_out ** 2))
    calibrated_max_weight = float(np.max(w_out))
    ess_ratio = (
        float(calibrated_ess / kernel_ess)
        if kernel_ess > 0
        else np.nan
    )
    calibration_degenerate = bool(
        calibrate
        and (
            calibrated_max_weight > 0.5
            or (
                np.isfinite(ess_ratio)
                and ess_ratio < 0.05
            )
            or calibrated_ess < (metric.dimension + 1)
        )
    )

    full_scaled_distance = raw_distance / current_bandwidth
    full_scores = _kernel_values(full_scaled_distance, kernel)
    full_score_sum = float(np.sum(full_scores))
    retained_score_sum = float(np.sum(full_scores[index]))
    retained_kernel_mass = (
        retained_score_sum / full_score_sum
        if full_score_sum > 0
        else np.nan
    )

    diagnostics = {
        "n_full": n_rows,
        "n_active_support": int(active_count),
        "n_pool": int(final_pool.size),
        "n_positive_kernel": int(index.size),
        "candidate_cap": int(cap),
        "candidate_cap_binding": bool(cap_binding),
        "kernel_mass_retained": float(retained_kernel_mass),
        "shrinking_target_count": (
            None
            if shrinking_target_count is None
            else int(shrinking_target_count)
        ),
        "prediction_dimension_original": int(F.shape[1]),
        "prediction_dimension_effective": int(metric.dimension),
        "prediction_dimension_discarded": int(
            metric.discarded_dimension
        ),
        "affine_support_residual": support_residual,
        "affine_support_residual_norm": float(
            support_residual_norm
        ),
        "affine_support_tolerance": float(
            support_tolerance
        ),
        "kernel": kernel,
        "gaussian_cutoff": (
            float(gaussian_cutoff)
            if kernel == "gaussian"
            else None
        ),
        "initial_bandwidth_source": initial_bandwidth_source,
        "final_bandwidth": float(current_bandwidth),
        "adaptive_widen": bool(adaptive_widen),
        "widening_steps": int(widening_steps),
        "calibration_requested": bool(calibrate),
        "calibration_success": bool(calibration.success),
        "optimizer_success": bool(
            calibration.optimizer_success
        ),
        "calibration_message": calibration.message,
        "calibration_iterations": int(calibration.iterations),
        "calibration_degenerate": calibration_degenerate,
        "calibrated_ess_ratio": ess_ratio,
        "kernel_ess": kernel_ess,
        "calibrated_ess": calibrated_ess,
        "kernel_max_weight": float(np.max(q_out)),
        "calibrated_max_weight": calibrated_max_weight,
        "kernel_mean_prediction": (
            float(kernel_mean[0])
            if F.shape[1] == 1
            else kernel_mean
        ),
        "calibrated_mean_prediction": (
            float(calibrated_mean[0])
            if F.shape[1] == 1
            else calibrated_mean
        ),
        "kernel_raw_max_abs_gap": float(
            np.max(np.abs(kernel_gap))
        ),
        "calibrated_raw_max_abs_gap": (
            calibrated_raw_max_abs_gap
        ),
        "calibration_raw_tolerance": float(
            raw_tolerance
        ),
        "kernel_neutrality_norm": float(
            np.linalg.norm(kernel_coord_gap)
        ),
        "calibrated_neutrality_norm": (
            calibrated_neutrality_norm
        ),
        "calibration_kl_to_kernel": (
            kl_calibrated_to_kernel
        ),
        "mean_scaled_distance_kernel": float(
            np.dot(
                q_out,
                final_scaled_distance[positive],
            )
        ),
        "mean_scaled_distance_calibrated": float(
            np.dot(
                w_out,
                final_scaled_distance[positive],
            )
        ),
        "max_scaled_distance_positive_kernel": float(
            np.max(
                final_scaled_distance[positive]
            )
        ),
    }

    if cap_binding and retained_kernel_mass < 0.999:
        warnings.warn(
            "The candidate cap truncates nonnegligible kernel mass "
            f"({100.0 * (1.0 - retained_kernel_mass):.3f}% omitted). "
            "Increase candidate_size if this approximation is material.",
            RuntimeWarning,
            stacklevel=2,
        )

    if calibration_degenerate:
        warnings.warn(
            "Calibration met the neutrality tolerance but the calibrated "
            "weights are degenerate: effective sample size collapsed to "
            f"{calibrated_ess:.2f} (kernel ESS {kernel_ess:.2f}, ratio "
            f"{ess_ratio:.3f}; max weight {calibrated_max_weight:.3f}). "
            "This is the signature of f0 lying at or beyond the edge of the "
            "localized support. Widen the bandwidth or inspect the convex "
            "support of the localized predictions.",
            RuntimeWarning,
            stacklevel=2,
        )

    f0_out: np.ndarray | float
    if F.shape[1] == 1:
        f0_out = float(target[0])
    else:
        f0_out = target.copy()

    predictions_out = (
        F_out[:, 0]
        if F_original.ndim == 1
        else F_out
    )

    return WeightedBackground(
        index=index,
        X=X_out,
        predictions=predictions_out,
        f0=f0_out,
        kernel_weights=q_out,
        calibrated_weights=w_out,
        metric=metric,
        bandwidth=float(current_bandwidth),
        kernel=kernel,
        diagnostics=diagnostics,
    )

weighted_background = kernel_weighted_background

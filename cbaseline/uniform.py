"""
Equal-weight prediction-neutral backgrounds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import (
    PredictionMetric,
    _as_2d,
    _validate_target_affine_support,
    fit_prediction_metric,
)

@dataclass
class UniformBackground:
    """A deterministic equal-weight background selected from observed rows."""

    index: np.ndarray
    X: np.ndarray
    predictions: np.ndarray
    f0: np.ndarray | float
    metric: PredictionMetric
    selected_mask: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def selected(self) -> np.ndarray:
        """Return the selected equal-weight background rows."""
        return self.X[self.selected_mask]

    def selected_index(self) -> np.ndarray:
        """Return selected row indices in the original reference sample."""
        return self.index[self.selected_mask]

    def selected_predictions(self) -> np.ndarray:
        """Return predictions associated with selected background rows."""
        return self.predictions[self.selected_mask]

    def weights(self) -> np.ndarray:
        """Return equal weights for the selected rows."""
        n_selected = int(np.sum(self.selected_mask))
        if n_selected <= 0:
            return np.zeros(0, dtype=float)
        return np.full(n_selected, 1.0 / n_selected, dtype=float)

    def __repr__(self) -> str:
        d = self.diagnostics
        return (
            "UniformBackground("
            f"selected={d.get('n_selected', 0)}, "
            f"dimension={d.get('prediction_dimension_effective', 0)}, "
            f"neutrality={d.get('selected_neutrality_norm', np.nan):.3e}, "
            f"shift={d.get('slide_shift', np.nan):.3e}"
            ")"
        )

def _smallest_k(values: np.ndarray, k: int) -> np.ndarray:
    """Return deterministic indices of the k smallest finite values.

    The partition identifies the kth order statistic in expected O(n) time.
    Boundary ties are resolved by original row index, so the result does not
    depend on implementation-specific ordering inside ``argpartition``.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size

    if n == 0:
        raise ValueError("values must contain at least one entry.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite.")

    k = int(k)
    if not 1 <= k <= n:
        raise ValueError(f"k must lie between 1 and {n}; received {k}.")

    if k == n:
        selected = np.arange(n, dtype=int)
    else:
        kth_value = float(np.partition(values, k - 1)[k - 1])
        below = np.flatnonzero(values < kth_value)
        equal = np.flatnonzero(values == kth_value)

        need = k - below.size
        if need < 0:
            # This should be impossible for an exact kth order statistic.
            raise RuntimeError("Internal order-statistic inconsistency.")
        if need > equal.size:
            raise RuntimeError("Insufficient boundary ties at kth order statistic.")

        boundary = np.sort(equal)[:need]
        selected = np.concatenate([below, boundary])

    order = np.lexsort((selected, values[selected]))
    return selected[order]

def _selection_at_shift(
    coordinates: np.ndarray,
    distance2_zero: np.ndarray,
    projection: np.ndarray,
    shift: float,
    size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate one translated center using precomputed geometry."""
    distance2_center = (
        distance2_zero
        - 2.0 * float(shift) * projection
        + float(shift) ** 2
    )
    # Roundoff can make theoretically nonnegative values slightly negative.
    distance2_center = np.maximum(distance2_center, 0.0)

    selected = _smallest_k(distance2_center, size)
    mean_coordinates = coordinates[selected].mean(axis=0)
    objective = float(np.linalg.norm(mean_coordinates))

    return objective, selected, mean_coordinates

def _candidate_is_better(
    objective: float,
    shift: float,
    selected: np.ndarray,
    *,
    best_objective: float,
    best_shift: float,
    best_selected: Optional[np.ndarray],
) -> bool:
    """Lexicographic comparison for deterministic grid-search results."""
    tolerance = 1e-15

    if objective < best_objective - tolerance:
        return True
    if objective > best_objective + tolerance:
        return False

    if shift < best_shift - tolerance:
        return True
    if shift > best_shift + tolerance:
        return False

    if best_selected is None:
        return True

    return tuple(selected.tolist()) < tuple(best_selected.tolist())

def fixed_size_uniform_slide(
    coordinates: np.ndarray,
    size: int,
    *,
    n_grid: int = 801,
    refinement_rounds: int = 3,
    max_shift: Optional[float] = None,
    max_shift_multiplier: float = 3.0,
    min_direction_norm: float = 1e-12,
    max_pool_size: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """Select a fixed-size local neighborhood by a directed slide.

    Parameters
    ----------
    coordinates
        Whitened prediction gaps, shape ``(n, d)``.

    size
        Exact number of equally weighted rows to select.

    n_grid
        Number of shifts evaluated in each grid-search round.

    refinement_rounds
        Number of local refinements around the best shift.

    max_shift
        Maximum allowed shift along the directed slide. If omitted, a generous
        default is based on the initial neighborhood radius and mean gap.

    max_shift_multiplier
        Multiplier used only when ``max_shift`` is omitted.

    min_direction_norm
        If the initial mean whitened gap is smaller than this value, no slide is
        required.

    max_pool_size
        Optional safety limit on the exact certified candidate pool. If the
        certified pool exceeds this value, the function raises rather than
        silently using an approximate pool.

    Returns
    -------
    mask, diagnostics
        Boolean selection mask over all supplied rows and diagnostics describing
        localization, the certified pool, and the achieved neutrality.
    """
    U = np.asarray(coordinates, dtype=float)

    if U.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array.")
    if U.shape[0] == 0:
        raise ValueError("coordinates must contain at least one row.")
    if not np.all(np.isfinite(U)):
        raise ValueError("coordinates must be finite.")

    n, dimension = U.shape
    size = int(size)
    n_grid = int(n_grid)
    refinement_rounds = int(refinement_rounds)
    max_shift_multiplier = float(max_shift_multiplier)
    min_direction_norm = float(min_direction_norm)

    if not 1 <= size <= n:
        raise ValueError(f"size must lie between 1 and {n}; received {size}.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    if refinement_rounds < 1:
        raise ValueError("refinement_rounds must be at least 1.")
    if (
        not np.isfinite(max_shift_multiplier)
        or max_shift_multiplier <= 0
    ):
        raise ValueError("max_shift_multiplier must be finite and positive.")
    if (
        not np.isfinite(min_direction_norm)
        or min_direction_norm < 0
    ):
        raise ValueError("min_direction_norm must be finite and nonnegative.")

    if max_pool_size is not None:
        max_pool_size = int(max_pool_size)
        if max_pool_size < size:
            raise ValueError("max_pool_size cannot be smaller than size.")

    # Initial nearest-neutral neighborhood over the complete sample.
    distance2_zero_all = np.einsum("ij,ij->i", U, U)
    initial_idx = _smallest_k(distance2_zero_all, size)
    initial_mean = U[initial_idx].mean(axis=0)
    initial_norm = float(np.linalg.norm(initial_mean))
    initial_radius = float(
        np.sqrt(np.max(distance2_zero_all[initial_idx]))
    )

    if initial_norm <= min_direction_norm:
        direction = np.zeros(dimension, dtype=float)
        final_max_shift = 0.0
    else:
        direction = -initial_mean / initial_norm

        if max_shift is None:
            final_max_shift = float(
                max_shift_multiplier
                * max(initial_radius, initial_norm, 1e-8)
            )
        else:
            final_max_shift = float(max_shift)
            if not np.isfinite(final_max_shift) or final_max_shift < 0:
                raise ValueError("max_shift must be finite and nonnegative.")

    # Certified pool:
    # any point that can enter a nearest-m set for a in [0, A] has
    # ||u_i|| <= R0 + 2A.
    certified_radius = float(
        initial_radius + 2.0 * final_max_shift
    )
    certified_mask = (
        distance2_zero_all
        <= certified_radius ** 2 * (1.0 + 1e-14)
    )
    pool = np.flatnonzero(certified_mask)

    # Numerical safety: the initial nearest-m set must always be included.
    pool = np.union1d(pool, initial_idx).astype(int, copy=False)

    if max_pool_size is not None and pool.size > max_pool_size:
        raise ValueError(
            "The exact certified slide pool contains "
            f"{pool.size} rows, exceeding max_pool_size={max_pool_size}. "
            "Increase max_pool_size, reduce max_shift, or use a smaller "
            "max_shift_multiplier. The routine will not silently substitute "
            "an uncertified approximate pool."
        )

    Up = U[pool]
    distance2_zero = distance2_zero_all[pool]
    projection = np.dot(Up, direction)

    if initial_norm <= min_direction_norm:
        best_objective, best_local, best_mean = _selection_at_shift(
            Up,
            distance2_zero,
            projection,
            shift=0.0,
            size=size,
        )
        best_shift = 0.0
        rounds_completed = 0
    else:
        lo = 0.0
        hi = final_max_shift

        best_objective = np.inf
        best_shift = np.inf
        best_local = None
        best_mean = None
        rounds_completed = 0

        for round_index in range(refinement_rounds):
            local_best_objective = np.inf
            local_best_shift = np.inf
            local_best_selected = None
            local_best_mean = None

            for shift in np.linspace(lo, hi, n_grid):
                shift = float(shift)
                objective, selected, mean_coordinates = (
                    _selection_at_shift(
                        Up,
                        distance2_zero,
                        projection,
                        shift=shift,
                        size=size,
                    )
                )

                if _candidate_is_better(
                    objective,
                    shift,
                    selected,
                    best_objective=local_best_objective,
                    best_shift=local_best_shift,
                    best_selected=local_best_selected,
                ):
                    local_best_objective = objective
                    local_best_shift = shift
                    local_best_selected = selected
                    local_best_mean = mean_coordinates

            if _candidate_is_better(
                local_best_objective,
                local_best_shift,
                local_best_selected,
                best_objective=best_objective,
                best_shift=best_shift,
                best_selected=best_local,
            ):
                best_objective = local_best_objective
                best_shift = local_best_shift
                best_local = local_best_selected
                best_mean = local_best_mean

            rounds_completed = round_index + 1

            step = (hi - lo) / (n_grid - 1)
            if step <= np.finfo(float).eps:
                break

            # Refine around the best point found in the current round. The
            # current-round optimum necessarily lies inside [lo, hi].
            lo = max(0.0, local_best_shift - step)
            hi = min(final_max_shift, local_best_shift + step)

            if hi <= lo + np.finfo(float).eps:
                break

    if best_local is None or best_mean is None:
        raise RuntimeError("Uniform-background slide failed internally.")

    selected_idx = pool[np.asarray(best_local, dtype=int)]
    selected_idx = np.sort(selected_idx)

    mask = np.zeros(n, dtype=bool)
    mask[selected_idx] = True

    selected_mean = U[selected_idx].mean(axis=0)
    selected_distance_zero = np.sqrt(
        distance2_zero_all[selected_idx]
    )

    slide_center = float(best_shift) * direction
    selected_distance_center = np.sqrt(
        np.maximum(
            np.sum(
                (U[selected_idx] - slide_center.reshape(1, -1)) ** 2,
                axis=1,
            ),
            0.0,
        )
    )

    hit_max_shift = bool(
        final_max_shift > 0
        and np.isclose(
            best_shift,
            final_max_shift,
            rtol=0.0,
            atol=max(1e-12, final_max_shift * 1e-10),
        )
    )

    diagnostics = {
        "n_candidates": int(n),
        "n_pool": int(pool.size),
        "pool_fraction": float(pool.size / n),
        "n_selected": int(size),
        "selection_rule": (
            "fixed-size one-dimensional slide with certified pool"
        ),
        "pool_certified": True,
        "certified_pool_radius": certified_radius,
        "initial_neutrality_norm": initial_norm,
        "initial_mean_coordinates": initial_mean,
        "initial_mean_distance_to_neutral": float(
            np.mean(np.sqrt(distance2_zero_all[initial_idx]))
        ),
        "initial_median_distance_to_neutral": float(
            np.median(np.sqrt(distance2_zero_all[initial_idx]))
        ),
        "initial_max_distance_to_neutral": initial_radius,
        "slide_direction": direction,
        "slide_direction_norm": float(np.linalg.norm(direction)),
        "slide_shift": float(best_shift),
        "slide_center": slide_center,
        "slide_center_norm": float(np.linalg.norm(slide_center)),
        "slide_max_shift": float(final_max_shift),
        "slide_objective": float(best_objective),
        "slide_n_grid": int(n_grid),
        "slide_refinement_rounds_requested": int(refinement_rounds),
        "slide_refinement_rounds_completed": int(rounds_completed),
        "selected_mean_coordinates": selected_mean,
        "selected_neutrality_norm": float(
            np.linalg.norm(selected_mean)
        ),
        "selected_mean_distance_to_neutral": float(
            np.mean(selected_distance_zero)
        ),
        "selected_median_distance_to_neutral": float(
            np.median(selected_distance_zero)
        ),
        "selected_max_distance_to_neutral": float(
            np.max(selected_distance_zero)
        ),
        "selected_mean_distance_to_slide_center": float(
            np.mean(selected_distance_center)
        ),
        "selected_median_distance_to_slide_center": float(
            np.median(selected_distance_center)
        ),
        "selected_max_distance_to_slide_center": float(
            np.max(selected_distance_center)
        ),
        "hit_max_shift": hit_max_shift,
    }

    if hit_max_shift:
        warnings.warn(
            "The best searched equal-weight background occurs at max_shift. "
            "The one-dimensional search bound may be active; increase "
            "max_shift or max_shift_multiplier and compare the result.",
            RuntimeWarning,
            stacklevel=2,
        )

    return mask, diagnostics

def uniform_background(
    predictions: np.ndarray,
    f0: np.ndarray | float,
    features: np.ndarray,
    *,
    size: int,
    metric: Optional[PredictionMetric] = None,
    covariance: Optional[np.ndarray] = None,
    metric_ridge: float = 1e-6,
    eigen_tol: float = 1e-10,
    affine_support_atol: float = 1e-10,
    affine_support_rtol: float = 1e-8,
    tolerance: Optional[float] = None,
    n_grid: int = 801,
    refinement_rounds: int = 3,
    max_shift: Optional[float] = None,
    max_shift_multiplier: float = 3.0,
    min_direction_norm: float = 1e-12,
    max_pool_size: Optional[int] = None,
) -> UniformBackground:
    """Construct a fixed-size equal-weight background near ``f0``.

    Parameters
    ----------
    predictions
        Model outputs on the reference sample. Shape ``(n,)`` for scalar output
        or ``(n, k)`` for vector output. Multiclass use should pass centered
        logits.

    f0
        Required reference output.

    features
        Observed feature rows aligned with predictions.

    size
        Exact number of rows in the equal-weight background. This fixes the
        downstream attribution budget.

    metric
        Optional precomputed ``PredictionMetric``.

    covariance
        Optional covariance used when fitting the prediction-space metric.

    affine_support_atol, affine_support_rtol
        Tolerances used to verify that ``f0`` belongs to the affine output
        support of the reference predictions after redundant directions are
        removed.

    tolerance
        Optional tolerance for the whitened norm of the selected background's
        mean prediction gap. The construction always returns the best searched
        set. ``tolerance_met`` is ``None`` when no tolerance is supplied and a
        Boolean otherwise.

    max_pool_size
        Optional safety limit on the exact certified candidate pool. Exceeding
        the limit raises an error rather than silently approximating the slide.

    Notes
    -----
    A fixed size is a finite-sample computational choice, not an asymptotic
    bandwidth rule. For asymptotic localization, ``size`` can increase with the
    reference sample while the implied nearest-neighborhood radius shrinks.
    """
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

    (
        affine_support_residual,
        affine_support_residual_norm,
        affine_support_tolerance,
    ) = _validate_target_affine_support(
        metric,
        target,
        atol=affine_support_atol,
        rtol=affine_support_rtol,
    )

    gaps = F - target
    coordinates = metric.coordinates(gaps)

    selected_mask, slide_info = fixed_size_uniform_slide(
        coordinates,
        size=int(size),
        n_grid=n_grid,
        refinement_rounds=refinement_rounds,
        max_shift=max_shift,
        max_shift_multiplier=max_shift_multiplier,
        min_direction_norm=min_direction_norm,
        max_pool_size=max_pool_size,
    )

    selected_predictions_2d = F[selected_mask]
    selected_mean_prediction = selected_predictions_2d.mean(axis=0)
    selected_raw_gap = selected_mean_prediction - target
    selected_neutrality_norm = metric.norm(selected_raw_gap)

    if tolerance is None:
        tolerance_value = None
        tolerance_met = None
    else:
        tolerance_value = float(tolerance)
        if not np.isfinite(tolerance_value) or tolerance_value < 0:
            raise ValueError("tolerance must be finite and nonnegative.")
        tolerance_met = bool(
            selected_neutrality_norm <= tolerance_value
        )

    diagnostics = {
        **slide_info,
        "n_full": int(F.shape[0]),
        "prediction_dimension_original": int(F.shape[1]),
        "prediction_dimension_effective": int(metric.dimension),
        "prediction_dimension_discarded": int(
            metric.discarded_dimension
        ),
        "affine_support_residual": affine_support_residual,
        "affine_support_residual_norm": float(
            affine_support_residual_norm
        ),
        "affine_support_tolerance": float(
            affine_support_tolerance
        ),
        "selected_mean_prediction": (
            float(selected_mean_prediction[0])
            if F.shape[1] == 1
            else selected_mean_prediction
        ),
        "selected_raw_gap": (
            float(selected_raw_gap[0])
            if F.shape[1] == 1
            else selected_raw_gap
        ),
        "selected_raw_l2_gap": float(
            np.linalg.norm(selected_raw_gap)
        ),
        "selected_max_abs_gap": float(
            np.max(np.abs(selected_raw_gap))
        ),
        "selected_neutrality_norm": float(
            selected_neutrality_norm
        ),
        "tolerance": tolerance_value,
        "tolerance_met": tolerance_met,
    }

    predictions_out = (
        F[:, 0]
        if F_original.ndim == 1
        else F
    )

    f0_out: np.ndarray | float
    if F.shape[1] == 1:
        f0_out = float(target[0])
    else:
        f0_out = target.copy()

    return UniformBackground(
        index=np.arange(F.shape[0], dtype=int),
        X=X,
        predictions=predictions_out,
        f0=f0_out,
        metric=metric,
        selected_mask=selected_mask,
        diagnostics=diagnostics,
    )

equal_weight_background = uniform_background

fixed_size_background = uniform_background

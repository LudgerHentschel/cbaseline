"""Public construction API for prediction-neutral backgrounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np

from .uniform import UniformBackground, uniform_background
from .weighted import WeightedBackground, kernel_weighted_background

Weighting = Literal["equal", "kernel", "calibrated"]
UnderlyingBackground = Union[UniformBackground, WeightedBackground]


@dataclass(frozen=True)
class Background:
    """Common user-facing view of an equal- or weighted background.

    Instances are created by :func:`background`.  The underlying construction
    remains available through ``result`` for users who need method-specific
    details.
    """

    result: UnderlyingBackground
    weighting: Weighting

    @property
    def rows(self) -> np.ndarray:
        """Observed feature rows in the constructed background."""
        if isinstance(self.result, UniformBackground):
            return self.result.selected()
        return self.result.X

    @property
    def weights(self) -> np.ndarray:
        """Weights aligned with :attr:`rows`."""
        if isinstance(self.result, UniformBackground):
            return self.result.weights()
        if self.weighting == "kernel":
            return self.result.kernel_weights
        return self.result.calibrated_weights

    @property
    def index(self) -> np.ndarray:
        """Indices of :attr:`rows` in the original reference sample."""
        if isinstance(self.result, UniformBackground):
            return self.result.selected_index()
        return self.result.index

    @property
    def predictions(self) -> np.ndarray:
        """Model outputs associated with :attr:`rows`."""
        if isinstance(self.result, UniformBackground):
            return self.result.selected_predictions()
        return self.result.predictions

    @property
    def f0(self) -> np.ndarray | float:
        """Requested reference output."""
        return self.result.f0

    @property
    def metric(self):
        """Prediction-space metric used by the construction."""
        return self.result.metric

    @property
    def diagnostics(self) -> dict:
        """Construction diagnostics from the underlying method."""
        return self.result.diagnostics

    def resampled(
        self,
        n_draws: int,
        *,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Draw an equal-weight Monte Carlo sample from a weighted background.

        This method is available only for ``weighting='kernel'`` or
        ``weighting='calibrated'``.  For weight-blind software, prefer
        ``weighting='equal'`` unless an approximation to the weighted
        distribution is specifically required.
        """
        if isinstance(self.result, UniformBackground):
            raise TypeError(
                "resampled() applies only to weighted backgrounds. "
                "The equal-weight background is already directly usable."
            )
        return self.result.resampled(
            n_draws,
            calibrated=self.weighting == "calibrated",
            random_state=random_state,
        )

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def __repr__(self) -> str:
        return (
            "Background("
            f"weighting={self.weighting!r}, "
            f"rows={len(self)}, "
            f"dimension={self.metric.dimension}"
            ")"
        )


def background(
    predictions: np.ndarray,
    f0: np.ndarray | float,
    features: np.ndarray,
    *,
    weighting: Weighting = "equal",
    size: Optional[int] = None,
    **kwargs: Any,
) -> Background:
    """Construct a prediction-neutral empirical background.

    Parameters
    ----------
    predictions
        Model outputs on the reference sample.  Use scalar scores for
        regression or binary classification and centered-logit vectors for
        multiclass classification.
    f0
        User-specified reference output.
    features
        Observed reference rows aligned with ``predictions``.
    weighting
        ``"equal"`` (the default) selects a deterministic fixed-size
        equal-weight background. ``"kernel"`` returns direct localization
        weights.
        ``"calibrated"`` exponentially calibrates the kernel weights to the
        requested reference output.
    size
        Number of rows for ``weighting="equal"``. If omitted, defaults to
        100. Invalid for weighted modes.
    **kwargs
        Additional options passed to the selected construction.

    Returns
    -------
    Background
        A common interface exposing ``rows``, ``weights``, ``index``,
        ``predictions``, and ``diagnostics``.
    """
    if weighting not in {"equal", "kernel", "calibrated"}:
        raise ValueError(
            "weighting must be 'equal', 'kernel', or 'calibrated'."
        )

    if weighting == "equal":
        if size is None:
            size = 100
        if "calibrate" in kwargs:
            raise TypeError(
                "calibrate is not valid when weighting='equal'."
            )
        result = uniform_background(
            predictions=predictions,
            f0=f0,
            features=features,
            size=int(size),
            **kwargs,
        )
        return Background(result=result, weighting="equal")

    if size is not None:
        raise ValueError(
            "size applies only when weighting='equal'. "
            "Use bandwidth, quantile, or candidate_size for weighted "
            "backgrounds."
        )
    if "calibrate" in kwargs:
        raise TypeError(
            "Do not pass calibrate to background(); choose weighting='kernel' "
            "or weighting='calibrated' instead."
        )

    result = kernel_weighted_background(
        predictions=predictions,
        f0=f0,
        features=features,
        calibrate=weighting == "calibrated",
        **kwargs,
    )
    return Background(result=result, weighting=weighting)

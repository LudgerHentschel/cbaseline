from __future__ import annotations

import numpy as np
import pytest

from cbaseline import kernel_weighted_background


@pytest.mark.parametrize(
    "kernel",
    ["gaussian", "epanechnikov", "uniform"],
)
def test_weighted_scalar_is_neutral(
    scalar_problem,
    kernel,
):
    X, predictions, f0 = scalar_problem

    result = kernel_weighted_background(
        predictions,
        f0,
        X,
        kernel=kernel,
        quantile=0.35,
        calibration_tolerance=1e-9,
    )

    rows, weights = result.weighted()
    assert rows.shape[0] == weights.size
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0)

    weighted_mean = np.dot(weights, result.predictions)
    assert weighted_mean == pytest.approx(f0, abs=1e-8)

    d = result.diagnostics
    assert d["calibration_success"] is True
    assert d["calibrated_neutrality_norm"] <= 1e-9
    assert d["calibrated_raw_max_abs_gap"] <= d[
        "calibration_raw_tolerance"
    ]
    assert 1.0 <= d["calibrated_ess"] <= rows.shape[0] + 1e-9


def test_weighted_multiclass_is_vector_neutral(
    centered_logit_problem,
):
    X, Z, z0 = centered_logit_problem

    result = kernel_weighted_background(
        Z,
        z0,
        X,
        kernel="gaussian",
        quantile=0.5,
        calibration_tolerance=1e-9,
    )

    weighted_mean = np.dot(
        result.calibrated_weights,
        result.predictions,
    )
    np.testing.assert_allclose(
        weighted_mean,
        z0,
        atol=1e-8,
    )

    assert result.metric.dimension == 3
    assert result.diagnostics["prediction_dimension_discarded"] == 1


def test_calibrate_false_returns_kernel_weights(scalar_problem):
    X, predictions, f0 = scalar_problem

    result = kernel_weighted_background(
        predictions,
        f0,
        X,
        quantile=0.3,
        calibrate=False,
    )

    np.testing.assert_allclose(
        result.calibrated_weights,
        result.kernel_weights,
        rtol=0.0,
        atol=0.0,
    )
    assert result.diagnostics["calibration_requested"] is False


def test_resampled_is_reproducible(scalar_problem):
    X, predictions, f0 = scalar_problem

    result = kernel_weighted_background(
        predictions,
        f0,
        X,
        quantile=0.4,
    )

    first = result.resampled(50, random_state=9)
    second = result.resampled(50, random_state=9)
    np.testing.assert_array_equal(first, second)

    with pytest.raises(ValueError, match="positive"):
        result.resampled(0)


def test_weighted_rejects_invalid_target(centered_logit_problem):
    X, Z, z0 = centered_logit_problem

    with pytest.raises(ValueError, match="affine output support"):
        kernel_weighted_background(
            Z,
            z0 + np.ones_like(z0),
            X,
        )

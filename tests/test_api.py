from __future__ import annotations

import numpy as np
import pytest

from cbaseline import Background, background
from cbaseline.uniform import UniformBackground
from cbaseline.weighted import WeightedBackground


def test_equal_background_common_interface(scalar_problem):
    X, predictions, f0 = scalar_problem

    bg = background(
        predictions,
        f0,
        X,
        weighting="equal",
        size=100,
        n_grid=101,
        refinement_rounds=2,
    )

    assert isinstance(bg, Background)
    assert isinstance(bg.result, UniformBackground)
    assert bg.weighting == "equal"
    assert bg.rows.shape == (100, X.shape[1])
    assert bg.index.shape == (100,)
    assert bg.predictions.shape == (100,)
    assert bg.weights.shape == (100,)
    assert bg.weights.sum() == pytest.approx(1.0)
    assert len(bg) == 100
    assert bg.diagnostics is bg.result.diagnostics


def test_calibrated_background_common_interface(scalar_problem):
    X, predictions, f0 = scalar_problem

    bg = background(
        predictions,
        f0,
        X,
        weighting="calibrated",
        quantile=0.4,
        calibration_tolerance=1e-9,
    )

    assert isinstance(bg.result, WeightedBackground)
    assert bg.weighting == "calibrated"
    np.testing.assert_allclose(bg.rows, bg.result.X)
    np.testing.assert_allclose(bg.weights, bg.result.calibrated_weights)
    assert np.dot(bg.weights, bg.predictions) == pytest.approx(f0, abs=1e-8)


def test_kernel_background_returns_direct_kernel_weights(scalar_problem):
    X, predictions, f0 = scalar_problem

    bg = background(
        predictions,
        f0,
        X,
        weighting="kernel",
        quantile=0.4,
    )

    assert isinstance(bg.result, WeightedBackground)
    np.testing.assert_allclose(bg.weights, bg.result.kernel_weights)
    assert bg.result.diagnostics["calibration_requested"] is False


def test_background_routes_to_existing_constructors(scalar_problem):
    X, predictions, f0 = scalar_problem

    equal = background(
        predictions,
        f0,
        X,
        weighting="equal",
        size=75,
        n_grid=101,
        refinement_rounds=2,
    )
    direct = equal.result

    from cbaseline import uniform_background

    legacy = uniform_background(
        predictions,
        f0,
        X,
        size=75,
        n_grid=101,
        refinement_rounds=2,
    )
    np.testing.assert_array_equal(
        direct.selected_index(),
        legacy.selected_index(),
    )


def test_background_validates_mode_specific_arguments(scalar_problem):
    X, predictions, f0 = scalar_problem

    default = background(
        predictions,
        f0,
        X,
        n_grid=101,
        refinement_rounds=2,
    )
    assert default.weighting == "equal"
    assert len(default) == 100

    with pytest.raises(ValueError, match="size applies only"):
        background(
            predictions,
            f0,
            X,
            weighting="calibrated",
            size=100,
        )

    with pytest.raises(ValueError, match="weighting must be"):
        background(predictions, f0, X, weighting="bad")

    with pytest.raises(TypeError, match="Do not pass calibrate"):
        background(
            predictions,
            f0,
            X,
            weighting="kernel",
            calibrate=True,
        )


def test_resampled_behavior(scalar_problem):
    X, predictions, f0 = scalar_problem

    equal = background(
        predictions,
        f0,
        X,
        weighting="equal",
        size=50,
        n_grid=51,
        refinement_rounds=1,
    )
    with pytest.raises(TypeError, match="weighted backgrounds"):
        equal.resampled(10)

    calibrated = background(
        predictions,
        f0,
        X,
        weighting="calibrated",
        quantile=0.4,
    )
    first = calibrated.resampled(20, random_state=7)
    second = calibrated.resampled(20, random_state=7)
    np.testing.assert_array_equal(first, second)

from __future__ import annotations

import numpy as np
import pytest

from cbaseline import uniform_background
from cbaseline.uniform import fixed_size_uniform_slide


def test_uniform_scalar_is_deterministic_and_exact_size(scalar_problem):
    X, predictions, f0 = scalar_problem

    first = uniform_background(
        predictions,
        f0,
        X,
        size=100,
        n_grid=151,
        refinement_rounds=2,
    )
    second = uniform_background(
        predictions,
        f0,
        X,
        size=100,
        n_grid=151,
        refinement_rounds=2,
    )

    np.testing.assert_array_equal(
        first.selected_index(),
        second.selected_index(),
    )
    assert first.selected().shape == (100, X.shape[1])
    assert first.weights().sum() == pytest.approx(1.0)
    assert np.all(first.weights() == pytest.approx(0.01))

    d = first.diagnostics
    assert d["n_selected"] == 100
    assert d["pool_certified"] is True
    assert d["selected_neutrality_norm"] <= d["initial_neutrality_norm"] + 1e-12


def test_uniform_multiclass_uses_k_minus_one_dimensions(
    centered_logit_problem,
):
    X, Z, z0 = centered_logit_problem

    result = uniform_background(
        Z,
        z0,
        X,
        size=120,
        n_grid=151,
        refinement_rounds=2,
    )

    d = result.diagnostics
    assert d["prediction_dimension_original"] == 4
    assert d["prediction_dimension_effective"] == 3
    assert d["prediction_dimension_discarded"] == 1
    assert result.selected_predictions().shape == (120, 4)
    assert np.isfinite(d["selected_neutrality_norm"])


def test_uniform_tolerance_is_reported_not_assumed(scalar_problem):
    X, predictions, f0 = scalar_problem

    result = uniform_background(
        predictions,
        f0,
        X,
        size=50,
        tolerance=0.0,
        n_grid=101,
        refinement_rounds=1,
    )

    assert result.diagnostics["tolerance"] == 0.0
    assert result.diagnostics["tolerance_met"] in {True, False}


def test_certified_pool_limit_raises():
    rng = np.random.default_rng(0)
    coordinates = rng.normal(size=(1000, 2))

    with pytest.raises(ValueError, match="certified slide pool"):
        fixed_size_uniform_slide(
            coordinates,
            size=50,
            max_shift=10.0,
            max_pool_size=60,
            n_grid=21,
            refinement_rounds=1,
        )


def test_uniform_rejects_invalid_multiclass_target(
    centered_logit_problem,
):
    X, Z, z0 = centered_logit_problem
    invalid = z0 + np.ones_like(z0)

    with pytest.raises(ValueError, match="affine output support"):
        uniform_background(
            Z,
            invalid,
            X,
            size=100,
        )

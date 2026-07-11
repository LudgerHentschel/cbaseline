from __future__ import annotations

import numpy as np
import pytest

from cbaseline import fit_prediction_metric
from cbaseline.geometry import _validate_target_affine_support


def test_scalar_metric_has_one_effective_dimension(scalar_problem):
    _, predictions, _ = scalar_problem
    metric = fit_prediction_metric(predictions)

    assert metric.original_dimension == 1
    assert metric.dimension == 1
    assert metric.discarded_dimension == 0
    assert metric.ridge >= 0

    gaps = predictions[:10] - predictions.mean()
    coordinates = metric.coordinates(gaps)
    assert coordinates.shape == (10, 1)
    assert np.all(np.isfinite(coordinates))


def test_centered_logits_drop_common_direction(centered_logit_problem):
    _, Z, z0 = centered_logit_problem
    metric = fit_prediction_metric(Z)

    assert metric.original_dimension == 4
    assert metric.dimension == 3
    assert metric.discarded_dimension == 1

    residual = metric.affine_support_residual(z0)
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_invalid_multiclass_target_rejected(centered_logit_problem):
    _, Z, z0 = centered_logit_problem
    metric = fit_prediction_metric(Z)

    invalid = z0 + np.ones_like(z0)
    with pytest.raises(ValueError, match="affine output support"):
        _validate_target_affine_support(
            metric,
            invalid,
            atol=1e-12,
            rtol=1e-12,
        )


def test_metric_rejects_nonfinite_and_bad_covariance():
    with pytest.raises(ValueError, match="finite"):
        fit_prediction_metric(np.array([0.0, np.nan, 1.0]))

    with pytest.raises(ValueError, match="covariance must have shape"):
        fit_prediction_metric(
            np.arange(10.0).reshape(5, 2),
            covariance=np.eye(3),
        )

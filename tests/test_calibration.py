from __future__ import annotations

import numpy as np
import pytest

from cbaseline.calibration import exponential_calibration


def test_exponential_calibration_balances_coordinates():
    coordinates = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [2.0, 0.0],
        ]
    )
    q = np.array([0.1, 0.2, 0.3, 0.4])

    result = exponential_calibration(
        coordinates,
        q,
        tolerance=1e-11,
    )

    assert result.success
    assert result.weights.sum() == pytest.approx(1.0)
    assert np.all(result.weights >= 0)
    np.testing.assert_allclose(
        result.weights @ coordinates,
        np.zeros(2),
        atol=1e-10,
    )


def test_calibration_reports_too_few_positive_rows():
    coordinates = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    q = np.array([0.5, 0.5, 0.0])

    result = exponential_calibration(
        coordinates,
        q,
        tolerance=1e-10,
    )

    assert not result.success
    assert "Too few positive-kernel observations" in result.message


def test_calibration_input_validation():
    with pytest.raises(ValueError, match="align"):
        exponential_calibration(
            np.zeros((3, 1)),
            np.ones(2),
        )

    with pytest.raises(ValueError, match="nonnegative"):
        exponential_calibration(
            np.zeros((3, 1)),
            np.array([0.5, -0.1, 0.6]),
        )

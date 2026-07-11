from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def scalar_problem():
    rng = np.random.default_rng(12345)
    X = rng.normal(size=(1200, 5))
    predictions = (
        0.7 * X[:, 0]
        - 0.4 * X[:, 1]
        + 0.25 * X[:, 2] ** 2
        + 0.1 * X[:, 3]
    )
    f0 = float(predictions.mean())
    return X, predictions, f0


@pytest.fixture
def centered_logit_problem():
    rng = np.random.default_rng(54321)
    X = rng.normal(size=(1500, 7))
    raw = np.column_stack(
        [
            0.9 * X[:, 0] + 0.2 * X[:, 1],
            -0.6 * X[:, 0] + 0.4 * X[:, 2],
            0.5 * X[:, 1] - 0.3 * X[:, 3],
            -0.2 * X[:, 2] + 0.6 * X[:, 4],
        ]
    )
    Z = raw - raw.mean(axis=1, keepdims=True)
    z0 = Z.mean(axis=0)
    return X, Z, z0

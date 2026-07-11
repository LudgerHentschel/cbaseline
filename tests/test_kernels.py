from __future__ import annotations

import numpy as np
import pytest

from cbaseline.kernels import (
    _active_support_count,
    _bandwidth_from_quantile,
    _bandwidth_silverman_rate,
    _kernel_values,
    _normalized_kernel_weights,
)


def test_kernel_values():
    d = np.array([0.0, 0.5, 1.0, 2.0])

    g = _kernel_values(d, "gaussian")
    assert g[0] == pytest.approx(1.0)
    assert np.all(g > 0)

    e = _kernel_values(d, "epanechnikov")
    np.testing.assert_allclose(e, [1.0, 0.75, 0.0, 0.0])

    u = _kernel_values(d, "uniform")
    np.testing.assert_allclose(u, [1.0, 1.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="Unknown kernel"):
        _kernel_values(d, "bad")


def test_normalized_weights_sum_to_one():
    raw = np.array([0.0, 0.25, 0.75, 1.25])
    weights, scaled, scores = _normalized_kernel_weights(
        raw,
        bandwidth=1.0,
        kernel="epanechnikov",
    )

    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0)
    assert weights[-1] == 0.0
    np.testing.assert_allclose(scaled, raw)
    assert np.all(scores >= 0)


def test_bandwidth_rules():
    raw = np.linspace(0.0, 10.0, 1001)

    q = _bandwidth_from_quantile(raw, 0.2)
    assert q == pytest.approx(np.quantile(raw, 0.2))

    bandwidth, count = _bandwidth_silverman_rate(
        raw,
        dimension=3,
        count_constant=0.5,
    )
    assert bandwidth > 0
    assert 5 <= count <= raw.size

    with pytest.raises(ValueError, match="quantile"):
        _bandwidth_from_quantile(raw, 0.0)


def test_active_support_boundaries():
    distances = np.array([0.0, 0.5, 1.0, 1.0, 1.5])

    assert _active_support_count(
        distances, 1.0, "uniform", 5.0
    ) == 4
    assert _active_support_count(
        distances, 1.0, "epanechnikov", 5.0
    ) == 2
    assert _active_support_count(
        distances, 0.2, "gaussian", 5.0
    ) == 4

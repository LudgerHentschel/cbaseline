from __future__ import annotations


def test_package_root_exports():
    from cbaseline import (
        Background,
        background,
        PredictionMetric,
        UniformBackground,
        WeightedBackground,
        equal_weight_background,
        fit_prediction_metric,
        fixed_size_background,
        kernel_weighted_background,
        uniform_background,
        weighted_background,
    )

    assert Background is not None
    assert callable(background)
    assert PredictionMetric is not None
    assert UniformBackground is not None
    assert WeightedBackground is not None
    assert callable(fit_prediction_metric)
    assert callable(kernel_weighted_background)
    assert callable(uniform_background)
    assert weighted_background is kernel_weighted_background
    assert equal_weight_background is uniform_background
    assert fixed_size_background is uniform_background


def test_version_is_exposed():
    import cbaseline

    assert isinstance(cbaseline.__version__, str)
    assert cbaseline.__version__

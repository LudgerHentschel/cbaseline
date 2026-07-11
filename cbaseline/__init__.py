"""Prediction-neutral empirical backgrounds for feature attribution."""

from importlib.metadata import PackageNotFoundError, version

from .api import Background, Weighting, background
from .geometry import PredictionMetric, fit_prediction_metric

# Advanced constructors remain importable from the package root, but the
# documented public workflow uses background().
from .uniform import (
    UniformBackground,
    equal_weight_background,
    fixed_size_background,
    uniform_background,
)
from .weighted import (
    WeightedBackground,
    kernel_weighted_background,
    weighted_background,
)

try:
    __version__ = version("cbaseline")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Background",
    "Weighting",
    "background",
    "PredictionMetric",
    "fit_prediction_metric",
]

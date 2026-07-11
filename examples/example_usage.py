"""Minimal examples for the public CBaseline API."""

import numpy as np

from cbaseline import background

rng = np.random.default_rng(0)
n = 5000
X = rng.standard_normal((n, 4))
predictions = (X[:, 0] - 1.0) ** 2 + X[:, 1]
f0 = float(predictions.mean())

# Equal-weight background for SHAP and other weight-blind consumers.
equal_bg = background(
    predictions,
    f0,
    X,
)
print(equal_bg)
print("equal rows:", equal_bg.rows.shape)
print("equal mean prediction:", equal_bg.predictions.mean())
print("equal neutrality:", equal_bg.diagnostics["selected_neutrality_norm"])

# Calibrated weighted background for consumers that natively accept weights.
weighted_bg = background(
    predictions,
    f0,
    X,
    weighting="calibrated",
)
print(weighted_bg)
print("weighted rows:", weighted_bg.rows.shape)
print("weighted mean prediction:", np.dot(weighted_bg.weights, weighted_bg.predictions))
print("calibrated ESS:", weighted_bg.diagnostics["calibrated_ess"])

# Direct localization weights without finite-sample calibration.
kernel_bg = background(
    predictions,
    f0,
    X,
    weighting="kernel",
)
print("kernel mean prediction:", np.dot(kernel_bg.weights, kernel_bg.predictions))

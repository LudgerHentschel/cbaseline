# CBaseline

[![PyPI version](https://img.shields.io/pypi/v/cbaseline.svg)](https://pypi.org/project/cbaseline/)
[![Documentation](https://img.shields.io/badge/docs-user%20guide-blue)](https://ludgerhentschel.github.io/cbaseline/)

**[Read the documentation](https://ludgerhentschel.github.io/cbaseline/)** · [Getting started](https://ludgerhentschel.github.io/cbaseline/getting-started.html)

CBaseline constructs **prediction-neutral background distributions** for
Integrated Gradients and SHAP. It supplies observed baseline rows and weights
for the **UnifiedIG / TreeIG stack**, and compact equal-weight backgrounds for
**SHAP** and other attribution software that accepts an unweighted matrix.

Given a fitted model, observed reference data, and a reference prediction
$f_0$, CBaseline localizes observations in prediction space and constructs a
background whose mean model output matches $f_0$, exactly to numerical
tolerance with calibrated weights or approximately with equal weights.
Attributions then explain the intended contrast $f(x)-f_0$, subject to the
reported background residual and the attribution method's numerical error.

![Observed inputs near a prediction-neutral manifold](docs/Figure_NeutralManifold.svg)

The blue points are observed inputs; the red curve is the neutral manifold,
$\mathcal{M}_0=\{x:f(x)=f_0\}$. CBaseline builds its background from observed
cases near this curve in prediction space, illustrated by the shaded band.
The background therefore represents realistic reference inputs concentrated
around the prediction being used for comparison.

The methodology is developed in two technical papers:

- [**Canonical Integrated Gradients: Expectations over Neutral Prediction Baselines** — Hentschel (2026a)](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf).
- [**A Canonical Background Distribution for Shapley Attribution** — Hentschel (2026b)](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26i.pdf).

## Installation

Install from conda-forge with conda:

```bash
conda install -c conda-forge cbaseline
```

Or install from PyPI with pip:

```bash
pip install cbaseline
```

Requires Python 3.10 or newer, NumPy, SciPy, and Numba. Attribution libraries
are installed separately. For the examples below:

```bash
pip install treeig scikit-learn shap
```

## Quickstart: Integrated Gradients

This complete regression example passes a calibrated background directly to
[TreeIG](https://github.com/LudgerHentschel/treeig):

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from cbaseline import background
from treeig import TreeIG

rng = np.random.default_rng(7)
X_train = rng.normal(size=(500, 3))
y_train = X_train[:, 0] ** 2 + X_train[:, 1] - 0.5 * X_train[:, 2]
model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X_train, y_train)
X_eval = X_train[:3]
f_train = model.predict(X_train)
f0 = float(f_train.mean())

wb = background(f_train, f0, X_train, weighting="calibrated")
ig = TreeIG(model, baseline=wb)
result = ig.explain(X_eval)

achieved = np.average(wb.predictions, axis=0, weights=wb.weights)
print("Reference:", achieved, "requested:", f0)
print("Effective sample size:", wb.diagnostics["calibrated_ess"])
np.testing.assert_allclose(
    result.values.sum(axis=1), model.predict(X_eval) - achieved, atol=1e-8
)
```

Calibrated weights are preferred when the attribution engine supports them.
For UnifiedIG and other IG engines, preserve both `wb.rows` and `wb.weights`
when averaging baseline attributions; see [integrations](https://ludgerhentschel.github.io/cbaseline/integrations.html).

## Quickstart: SHAP

Using the same fitted model and data, construct a deterministic equal-weight
background. An explicit masker preserves all selected rows:

```python
import shap

bg = background(f_train, f0, X_train, weighting="equal", size=100)
masker = shap.maskers.Independent(bg.rows, max_samples=len(bg.rows))
explainer = shap.Explainer(model.predict, masker, algorithm="permutation", seed=0)
phi = explainer(X_eval)

print("Neutrality gap:", bg.diagnostics["selected_raw_gap"])
np.testing.assert_allclose(
    phi.base_values, np.mean(model.predict(bg.rows)), atol=1e-8
)
np.testing.assert_allclose(
    phi.base_values + phi.values.sum(axis=1), model.predict(X_eval), atol=1e-8
)
```

SHAP explains the gap from the **achieved background mean**. A finite
fixed-size subset need not average exactly to `f0`; check its residual.
Standard matrix-background SHAP interfaces do not consume CBaseline's
observation weights, so use `weighting="equal"` for this workflow.

## Why prediction-neutral backgrounds?

The reference distribution determines the comparison being explained.
CBaseline uses observed inputs near the prediction-neutral set
$\mathcal{M}_0=\{x:f(x)=f_0\}$, without fitting a feature-distribution model.
Localization occurs in prediction space, so distant feature vectors may both
be relevant neutral references.

A mean feature vector need not predict `f0`. A full reference sample is neutral
at its own mean output but is not localized. CBaseline combines localization
with calibrated weights or a deterministic equal-weight selection. It changes
the reference distribution; TreeIG, UnifiedIG, or SHAP computes the attributions.

## Documentation

The [documentation guide](https://ludgerhentschel.github.io/cbaseline/) covers:

- [Getting started](https://ludgerhentschel.github.io/cbaseline/getting-started.html) and [TreeIG, UnifiedIG, and SHAP integrations](https://ludgerhentschel.github.io/cbaseline/integrations.html).
- [Prediction neutrality](https://ludgerhentschel.github.io/cbaseline/concepts.html) and [choosing `f0`](https://ludgerhentschel.github.io/cbaseline/reference-predictions.html) for regression, binary, and multiclass models.
- [Weighted and equal-weight constructions](https://ludgerhentschel.github.io/cbaseline/backgrounds.html), tuning, and [diagnostics](https://ludgerhentschel.github.io/cbaseline/diagnostics.html).
- [API reference](https://ludgerhentschel.github.io/cbaseline/api.html), [papers and citation](https://ludgerhentschel.github.io/cbaseline/references.html), and [building the Sphinx/PyData site](https://ludgerhentschel.github.io/cbaseline/building.html).

For classification, construct the background on the same score scale you
attribute. The logit of a reference probability and the mean model logit answer
different questions. Multiclass examples use one joint centered-logit vector
background across all classes.

## Citation

Please cite both supporting papers when using the combined methodology, or the
paper corresponding to your construction and attribution method:

```bibtex
@misc{hentschel2026a,
  author = {Hentschel, Ludger},
  title = {Canonical Integrated Gradients: Expectations over Neutral Prediction Baselines},
  year = {2026},
  url = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf},
}

@misc{hentschel2026b,
  author = {Hentschel, Ludger},
  title = {A Canonical Background Distribution for Shapley Attribution},
  year = {2026},
  url = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26i.pdf},
}
```

## License

CBaseline is distributed under the [BSD 3-Clause License](LICENSE).

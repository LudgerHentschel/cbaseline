# CBaseline

[![PyPI version](https://img.shields.io/pypi/v/cbaseline.svg)](https://pypi.org/project/cbaseline/)
[![Documentation](https://img.shields.io/badge/docs-user%20guide-blue)](https://ludgerhentschel.github.io/cbaseline/)

**Attribution methods explain a contrast. CBaseline constructs the reference
population that defines it.**

Integrated Gradients and Shapley attribution both decompose $f(x)-f_0$ for some
reference output $f_0$. That reference is usually inherited rather than chosen.
An all-zeros baseline generally lies off the data manifold. A mean input
$\bar{x}$ is not an observed case, and for nonlinear $f$,
$f(\bar{x}) \neq \mathbb{E}[f(X)]$, so the corresponding reference output is
not the mean prediction.

A useful reference must satisfy two separate requirements. It must be
**neutral** — its mean model output equals $f_0$ — and it must be
**localized** — assembled from cases the model already predicts near $f_0$.

CBaseline satisfies both using observed rows only. It localizes in prediction
space under a fitted metric that accounts for output scale and redundant
directions, then calibrates by exponential tilting until the weighted mean
output equals $f_0$ to numerical tolerance. Equal-weight backgrounds
approximate neutrality with a fixed-size selection and report the residual they
achieve. Attributions then explain the intended contrast $f(x)-f_0$, subject to
that residual and the attribution method's own numerical error.

The construction is method-agnostic. The same background serves Integrated
Gradients, interventional Shapley attribution — where the weighted mean is the
empty-coalition value $v(\emptyset)$ — and any implementation that accepts a
background matrix. Calibrated weights feed the **UnifiedIG / TreeIG stack**
directly; deterministic equal-weight backgrounds give **SHAP** and other
software a first-class workflow through standard background-matrix interfaces.

![Observed inputs near a prediction-neutral manifold](docs/Figure_NeutralManifold.svg)

The blue points are observed inputs; the red curve is the neutral manifold,
$\mathcal{M}_0=\{x:f(x)=f_0\}$. CBaseline builds its background from observed
cases near this curve in prediction space, illustrated by the shaded band.
The background therefore represents realistic reference inputs concentrated
around the prediction being used for comparison.

The methodology is developed in two technical papers:

- [**Canonical Integrated Gradients: Expectations over Neutral Prediction Baselines** — Hentschel (2026a)](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf).
- [**A Canonical Background Distribution for Shapley Attribution** — Hentschel (2026b)](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26i.pdf).

## Choose a background

| Attribution workflow | Mode | What neutrality means |
|---|---|---|
| An engine that accepts observation weights, such as UnifiedIG or TreeIG | `weighting="calibrated"` | The weighted mean prediction meets configured tolerances around `f0`; calibration failure raises an error. |
| A standard SHAP background-matrix interface | `weighting="equal"` | A deterministic fixed-size subset approximates `f0`; inspect the achieved mean and residual. |
| Localization without enforcing neutrality | `weighting="kernel"` | Kernel weights favor predictions near `f0`, but their mean is not constrained to equal it. |

Preserve both rows and weights for weighted attribution. Calibrated targets
must be supported by the available predictions; an arbitrary `f0` is not always
feasible. For every mode, the attribution reference is the **achieved background
mean**, and predictions must use the same output scale as the attribution engine.

## Installation

Install from conda-forge with conda:

```bash
conda install -c conda-forge cbaseline
```

Or install from PyPI with pip:

```bash
pip install cbaseline
```

Requires Python 3.10 or newer, NumPy, and SciPy. Attribution libraries
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

[Background modes](https://ludgerhentschel.github.io/cbaseline/backgrounds.html) and [diagnostics](https://ludgerhentschel.github.io/cbaseline/diagnostics.html)
explain feasibility, residuals, and failure behavior. For automated readers,
[llms.txt](https://ludgerhentschel.github.io/cbaseline/llms.txt) maps the guides,
complete examples, and rendered API reference.

The [documentation guide](https://ludgerhentschel.github.io/cbaseline/) covers:

- [Getting started](https://ludgerhentschel.github.io/cbaseline/getting-started.html) and [TreeIG, UnifiedIG, and SHAP integrations](https://ludgerhentschel.github.io/cbaseline/integrations.html).
- [Prediction neutrality](https://ludgerhentschel.github.io/cbaseline/concepts.html) and [choosing `f0`](https://ludgerhentschel.github.io/cbaseline/reference-predictions.html) for regression, binary, and multiclass models.
- [Weighted and equal-weight constructions](https://ludgerhentschel.github.io/cbaseline/backgrounds.html), tuning, and [diagnostics](https://ludgerhentschel.github.io/cbaseline/diagnostics.html).
- [API reference](https://ludgerhentschel.github.io/cbaseline/api.html), [papers and citation](https://ludgerhentschel.github.io/cbaseline/references.html), and [building the Sphinx/PyData site](https://ludgerhentschel.github.io/cbaseline/building.html).

For classification, construct the background on the same score scale you
attribute. The logit of a reference probability and the mean model logit answer
different questions. Multiclass examples use one joint centered-logit vector
background across all classes.

## Related projects

CBaseline can be used independently of the Integrated Gradients stack. Choose an
attribution engine that consumes the background in the intended form:

| Package | Role |
|---|---|
| [UnifiedIG](https://ludgerhentschel.github.io/unifiedig/) (`unifiedig`) | A common Integrated Gradients attribution interface across supported model families; accepts CBaseline backgrounds directly. |
| [TreeIG](https://ludgerhentschel.github.io/treeig/) (`treeig`) | Direct tree-path attribution for supported models; accepts CBaseline rows and weights through its background interface. |
| [skgrad](https://ludgerhentschel.github.io/skgrad/) (`skgrad`) | Analytic input gradients and Jacobians for supported scikit-learn models; a derivative component rather than an attribution engine. |

Standard SHAP matrix-background workflows use the equal-weight construction.

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

Release maintainers: see [Publishing releases](docs/publishing.md).

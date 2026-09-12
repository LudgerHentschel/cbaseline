---
myst:
  html_meta:
    description: "Use CBaseline backgrounds with TreeIG, UnifiedIG, and SHAP while preserving the appropriate rows, weights, and output scale."
---

# Integrated Gradients and SHAP integrations

CBaseline constructs the reference distribution; the attribution engine
explains predictions against it. Use calibrated weights when supported and
equal-weight selection when the interface accepts only a background matrix.

## Shared regression example

Run this setup before the TreeIG and SHAP sections. Install `treeig`,
`scikit-learn`, and `shap` in addition to CBaseline.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from cbaseline import background

rng = np.random.default_rng(7)
X_train = rng.normal(size=(500, 3))
y_train = X_train[:, 0] ** 2 + X_train[:, 1] - 0.5 * X_train[:, 2]
model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X_train, y_train)
X_eval = X_train[:3]
f_train = model.predict(X_train)
f0 = float(f_train.mean())
```

## TreeIG: calibrated baseline distributions

[TreeIG](https://github.com/LudgerHentschel/treeig) accepts a CBaseline
`Background` directly and reads its rows and weights. This is the preferred
way to pass a calibrated distribution to its CPU interface.

```python
from treeig import TreeIG

wb = background(f_train, f0, X_train, weighting="calibrated")
ig = TreeIG(model, baseline=wb)
result = ig.explain(X_eval)

achieved = np.average(wb.predictions, axis=0, weights=wb.weights)
np.testing.assert_allclose(
    result.values.sum(axis=1), ig.model_output(X_eval) - achieved, atol=1e-8
)
print("Reference gap:", achieved - f0)
print("Effective sample size:", wb.diagnostics["calibrated_ess"])
```

The equivalent explicit constructor is
`TreeIG(model, baseline=wb.rows, baseline_weights=wb.weights)`.
Pass the entire distribution through the engine to benefit from its baseline
batching. Averaging the feature rows into a single baseline generally changes
the output reference. See [TreeIG's baseline guide](https://ludgerhentschel.github.io/treeig/baselines.html).

For classification, use TreeIG's output conventions to obtain reference
scores; do not assume `model.predict` returns the scores TreeIG attributes.
Follow [choosing the reference prediction](reference-predictions.md) for
logits and class centering, and verify the output function on baseline rows.

## UnifiedIG and other Integrated Gradients engines

[UnifiedIG](https://ludgerhentschel.github.io/unifiedig/) accepts the same
CBaseline `Background` directly. Install `unifiedig` to run this section with
the shared regression setup and calibrated `wb` above:

```python
import unifiedig as uig

explanation = uig.Explainer(model, wb)(X_eval)
achieved = np.average(wb.predictions, axis=0, weights=wb.weights)
np.testing.assert_allclose(explanation.base_values, achieved, atol=1e-8)
np.testing.assert_allclose(
    explanation.base_values + explanation.values.sum(axis=1),
    model.predict(X_eval), atol=1e-8,
)
```

The background carries its aligned rows and weights through the wrapper to the
selected backend. See [UnifiedIG's baseline guide](https://ludgerhentschel.github.io/unifiedig/baselines.html)
and [model support](https://ludgerhentschel.github.io/unifiedig/supported-models.html)
when adapting this regression example to another model or output scale.

For an engine supporting only a single baseline at a time, the following
adapter makes the required averaging explicit. `attribute_one(X_eval, row)`
is a caller-supplied function returning an attribution array for one baseline;
it must explain the same output used to construct `wb`.

```python
def average_baseline_attributions(attribute_one, X_eval, wb):
    total = None
    weights = wb.weights / wb.weights.sum()
    for row, weight in zip(wb.rows, weights):
        if weight == 0:
            continue
        values = np.asarray(attribute_one(X_eval, row))
        contribution = weight * values
        total = contribution if total is None else total + contribution
    return total
```

Prefer native weighted batching when available. This adapter illustrates the
weighted-averaging contract for other attribution engines. If a backend
only supports equal-weight distributions, construct `weighting="equal"`
instead. Numerical IG needs its own integration-accuracy checks in addition
to CBaseline's mean-neutrality check.

## SHAP: deterministic equal-weight backgrounds

SHAP is a first-class use case through its unweighted background interfaces.
Construct a fixed-size subset and explicitly retain every selected row in
the masker:

```python
import shap

bg = background(f_train, f0, X_train, weighting="equal", size=100)
masker = shap.maskers.Independent(bg.rows, max_samples=len(bg.rows))
explainer = shap.Explainer(model.predict, masker, algorithm="permutation", seed=0)
phi = explainer(X_eval)

achieved = model.predict(bg.rows).mean()
np.testing.assert_allclose(phi.base_values, achieved, atol=1e-8)
np.testing.assert_allclose(
    phi.base_values + phi.values.sum(axis=1), model.predict(X_eval), atol=1e-8
)
print("Requested reference:", f0)
print("Achieved reference:", achieved)
print("Neutrality norm:", bg.diagnostics["selected_neutrality_norm"])
```

An independent masker can otherwise subsample a large background, changing
the selected reference. `max_samples=len(bg.rows)` preserves the CBaseline
construction. The fixed seed controls permutation sampling separately from
CBaseline's deterministic selection.

The total is `model.predict(X_eval) - achieved`, which differs from
`model.predict(X_eval) - f0` by the reported neutrality residual. The SHAP
base value is a useful independent check that the intended background and
output scale survived the integration.

For supported tree models, an interventional TreeExplainer can use the same
explicit masker:

```python
tree_explainer = shap.TreeExplainer(
    model, data=masker, feature_perturbation="interventional", model_output="raw"
)
tree_phi = tree_explainer(X_eval)
np.testing.assert_allclose(tree_phi.base_values, achieved, atol=1e-8)
np.testing.assert_allclose(
    tree_phi.base_values + tree_phi.values.sum(axis=1),
    model.predict(X_eval), atol=1e-6,
)
```

This example uses a regression tree. For classifiers, `raw` means the
model-specific raw output; it is not universally a logit. Use a score callable
with the general explainer when you need explicit centered-logit control.
Tree-path-dependent SHAP uses a different reference mechanism and should not
be assumed to implement this empirical background distribution.

Passing `wb.rows` alone discards calibrated weights. Standard matrix-based
SHAP workflows do not use `wb.weights`, so use the equal-weight construction
unless an explicitly weighted interface is available. Resampling the weighted
background is an optional approximation, discussed in
[background constructions](backgrounds.md).

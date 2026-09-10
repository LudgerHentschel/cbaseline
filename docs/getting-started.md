# Getting started

## Installation

Install from conda-forge with conda:

```bash
conda install -c conda-forge cbaseline
```

Or install from PyPI with pip:

```bash
pip install cbaseline
```

Python 3.10 or newer is required. NumPy and SciPy are the only runtime
dependencies; CBaseline itself does not require an attribution library.
Install `treeig`, `scikit-learn`, and `shap` to run the integration examples.

## Construct and inspect a background

This standalone example uses a known prediction function so no model training
is needed. With a fitted model, replace `predict` with the output function you
intend to explain.

```python
import numpy as np
from cbaseline import background

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 3))

def predict(X):
    return X[:, 0] ** 2 + X[:, 1] - 0.5 * X[:, 2]

predictions = predict(X)
f0 = float(predictions.mean())
wb = background(predictions, f0, X, weighting="calibrated")
bg = background(predictions, f0, X, weighting="equal", size=100)

for reference in (wb, bg):
    achieved = np.average(reference.predictions, axis=0, weights=reference.weights)
    print(reference.weighting, len(reference), achieved, achieved - f0)
    np.testing.assert_allclose(reference.rows, X[reference.index])
```

`rows`, `weights`, `index`, and `predictions` are aligned. `features` must be a
finite numeric two-dimensional array with one reference case per row;
`predictions` has shape `(n,)` or `(n, k)`. Feature columns and preprocessing
must match the fitted model. CBaseline receives outputs, not the model itself,
so the caller is responsible for this alignment.

For image inputs, flatten observed images for background construction and use
`index` to recover their original tensor shape for the attribution engine.
This does not change the prediction-space distances.

Use `wb.rows` **with** `wb.weights` in a weighted attribution engine. Use
`bg.rows` in an unweighted interface. Do not replace a distribution by its
mean feature vector: this generally changes the reference prediction.

## Next steps

Choose [the reference prediction](reference-predictions.md), compare
[construction options](backgrounds.md), and check [diagnostics](diagnostics.md).
Then follow the complete [TreeIG and SHAP examples](integrations.md).

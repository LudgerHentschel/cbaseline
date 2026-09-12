---
myst:
  html_meta:
    description: "Look up CBaseline background constructors, Background objects, prediction metrics, parameters, and diagnostics."
---

# API reference

Use `background` for normal workflows. It returns a `Background` with aligned
rows, weights, reference indices, stored predictions, and diagnostics.
`weighting="equal"` defaults to 100 rows; weighted modes reject `size`.
Additional keyword arguments go to the corresponding advanced constructor.

```{eval-rst}
.. autofunction:: cbaseline.background

.. autoclass:: cbaseline.Background
   :members: rows, weights, index, predictions, f0, metric, diagnostics, resampled

.. autofunction:: cbaseline.fit_prediction_metric

.. autoclass:: cbaseline.PredictionMetric
   :members:
```

## Advanced construction options

`uniform_background` is the constructor behind equal-weight mode.
`kernel_weighted_background` is the constructor behind kernel and calibrated
modes. Their method-specific result objects remain available through
`Background.result`. Prefer the common interface for integration code.

```{eval-rst}
.. autofunction:: cbaseline.uniform_background

.. autofunction:: cbaseline.kernel_weighted_background
```

See [background constructions](backgrounds.md) for mode selection and tuning,
and [diagnostics](diagnostics.md) for success and failure semantics.

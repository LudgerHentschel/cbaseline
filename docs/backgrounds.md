# Background constructions and tuning

All modes use observed rows and the prediction metric. Choose a construction
according to the attribution engine's support for observation weights.

| Mode | What it returns | Neutrality | Main tuning parameter |
| --- | --- | --- | --- |
| `calibrated` | Localized rows with calibrated weights | Enforced to configured numerical tolerances; failure raises | `bandwidth`, optionally `quantile` |
| `equal` | A deterministic fixed-size subset with uniform weights | Approximate; residual reported | `size` (default 100) |
| `kernel` | Localized rows with uncalibrated kernel weights | Not enforced | `bandwidth`, optionally `quantile` |

## Calibrated weights

Assume `predictions`, `f0`, and `X_reference` are aligned as in
[getting started](getting-started.md):

```python
wb = background(predictions, f0, X_reference, weighting="calibrated")
X_bg, weights = wb.rows, wb.weights
```

Kernel weights first favor outputs near `f0`. Exponential calibration adjusts
those weights under a relative-entropy criterion while enforcing their mean
output. The method is developed in [Hentschel (2026a), Section 3](references.md).
Calibration is limited by the convex hull of predictions with positive kernel
weight. A target in its relative interior is the usual feasible case; a
boundary target can require concentrated or limiting weights.

Use the public aligned `rows` and `weights` for attribution. The underlying
result retains `wb.result.kernel_weights` and
`wb.result.calibrated_weights` for comparison.

### Bandwidth

The default uses a shrinking Silverman-type rule. To choose a bandwidth in
the fitted prediction metric's units, pass `bandwidth=0.25`. Alternatively,
`quantile=0.10` initializes it from a distance quantile. A fixed quantile is a
finite-sample choice, not the shrinking-bandwidth regime in the paper.
Choose one initialization option; an explicit bandwidth takes precedence.

Supported kernels are `"gaussian"`, `"epanechnikov"`, and `"uniform"`.
Gaussian localization uses a finite cutoff in this implementation. The
`candidate_size` cap controls the weighted candidate pool, not an exact
number of output rows. Adaptive widening can expand the neighborhood to
reach calibration tolerances; inspect `final_bandwidth`, `widening_steps`,
`candidate_cap_binding`, and `kernel_mass_retained`.

Direct `weighting="kernel"` skips calibration. Its mean can differ from `f0`,
even if a diagnostic field named `calibration_success` is true; check
`calibration_requested` and the actual kernel residual.

## Equal-weight selection

```python
bg = background(
    predictions, f0, X_reference,
    weighting="equal", size=100, tolerance=1e-3,
)
```

The fixed-size construction searches a shifted prediction-space neighborhood
to improve its mean neutrality. It is not simply the 100 nearest predictions
to the original target, nor a guarantee of the globally best subset.
It returns exactly `size` distinct observed rows; the size must not exceed
the reference sample. No kernel bandwidth is used in this mode.

`tolerance` checks the achieved whitened mean-gap norm. Failure to meet it
does **not** raise: the best searched set is returned with
`tolerance_met=False`. Without a tolerance, that field is `None`.

Try several sizes to assess residual, localization, and downstream cost.
More rows can help but do not guarantee a monotonically smaller residual.
`max_pool_size` is a safety limit: if the certified candidate pool exceeds it,
the implementation raises rather than silently approximating the search.

## Resampling and computation

`wb.resampled(500, random_state=0)` is available when an unweighted Monte Carlo
approximation to the weighted distribution is specifically needed. It may
repeat rows and only matches the weighted distribution in expectation. Prefer
`weighting="equal"` for the standard SHAP workflow.

Construction scans reference predictions before working on candidate pools.
Cost depends on output dimension, support, and search settings; equal-weight
pools are not universally bounded. Downstream attribution cost also grows with
the retained baseline count. Reuse a metric when comparing constructions on
the same prediction sample:

```python
from cbaseline import fit_prediction_metric

metric = fit_prediction_metric(predictions)
for size in (50, 100, 200):
    bg = background(
        predictions, f0, X_reference,
        weighting="equal", size=size, metric=metric,
    )
    print(size, bg.diagnostics["selected_neutrality_norm"])
```

See the [API reference](api.md) for advanced search and calibration options.

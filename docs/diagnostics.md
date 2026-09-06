# Diagnostics and numerical checks

Inspect both the construction and the downstream explanation. A small
neutrality residual does not establish that the reference population is the
right one, or that an attribution approximation is accurate.

## Check the mean in output units

For any returned background `bg`:

```python
import numpy as np

achieved = np.average(bg.predictions, axis=0, weights=bg.weights)
raw_gap = achieved - bg.f0
print("Achieved reference:", achieved)
print("Gap from requested reference:", raw_gap)
```

`*_neutrality_norm` fields measure gaps in the fitted, whitened prediction
coordinates. Raw gaps remain interpretable in the output's units. For
multiclass output inspect the full vector, not only a scalar summary.

## Equal-weight backgrounds

| Diagnostic | Interpretation |
| --- | --- |
| `selected_mean_prediction` | Achieved reference output |
| `selected_raw_gap` | Achieved mean minus requested target |
| `selected_neutrality_norm` | Whitened norm of that gap |
| `n_selected` | Number of selected rows |
| `slide_shift` | Shift chosen in the neighborhood search |
| `tolerance_met` | `True` or `False` if requested, otherwise `None` |

An unmet tolerance returns a result rather than raising. Compare sizes and
inspect the reference sample's support. Do not silently treat the requested
target as the achieved baseline.

## Weighted backgrounds

| Diagnostic | Interpretation |
| --- | --- |
| `calibration_requested`, `calibration_success` | Whether calibration was requested and met its criteria |
| `calibrated_neutrality_norm` | Whitened calibrated mean-gap norm |
| `calibrated_raw_max_abs_gap` | Largest absolute output-coordinate gap |
| `kernel_ess`, `calibrated_ess` | Effective sample sizes before and after calibration |
| `calibration_degenerate` | Concentrated weights detected by implementation thresholds |
| `widening_steps`, `final_bandwidth` | Expansion needed for local support |
| `candidate_cap_binding`, `kernel_mass_retained` | Whether the candidate cap restricts localization |

For normalized weights, $\mathrm{ESS}=1/\sum_i w_i^2$. Many retained rows may
still represent little effective support. A successful but degenerate
calibration deserves inspection of weights and reference coverage; consider
a broader sample or bandwidth while keeping the intended question explicit.

If calibrated mode cannot meet either coordinate or raw-output tolerances,
it raises `ValueError` with residuals and suggested remedies. It does not
silently return an uncalibrated background. Depending on the message, widen
`bandwidth`, increase `max_bandwidth` or `max_widening_steps`, or increase
`candidate_size` when the cap binds. No amount of widening can put an
out-of-hull target inside the available predictions' convex hull.

Targets outside the affine output support are also rejected. In multiclass
problems check that target centering and class order match the predictions.

For `weighting="kernel"`, inspect `kernel_mean_prediction`,
`kernel_raw_max_abs_gap`, and `kernel_neutrality_norm` instead of reading
calibration status as a neutrality guarantee.

## Check the attribution separately

Re-evaluate the exact output function on `bg.rows`. Its weighted mean should
match `achieved`; if not, check preprocessing, output scale, and model version.
Then check that summed feature attributions plus the achieved base value
recover the output being explained, allowing for the attribution engine's
numerical tolerance. See [integrations](integrations.md) for executable checks.

Record the reference population, `f0`, output definition, mode, tuning settings,
achieved mean, residual, effective sample size where applicable, and package
versions. Keep the same background when comparing observations under one
reference question.

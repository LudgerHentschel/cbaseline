# Prediction-neutral backgrounds

## The prediction contrast

A baseline distribution is part of the attribution question. For Integrated
Gradients averaged over baselines $b_i$ with normalized weights $w_i$,

$$A_j(x)=\sum_i w_i\,\mathrm{IG}_j(x;b_i),\qquad
\sum_j A_j(x)=f(x)-\sum_i w_i f(b_i).$$

For interventional Shapley attribution, the same reference mean is the
empty-coalition value $v(\emptyset)=\mathbb{E}_Q[f(X)]$. Both methods
therefore decompose $f(x)-f_0$ when the background mean equals $f_0$.
They need not assign that total to features in the same way.

If the achieved mean is $f_0+r$, the attribution total is
$f(x)-f_0-r$, before any attribution numerical error. Report the residual
instead of describing an approximate background as exactly neutral.

## Localization and neutrality are different requirements

The neutral set is $\mathcal{M}_0=\{x:f(x)=f_0\}$. The canonical IG paper
formulates the reference as the data distribution conditional on this set.
Finite samples generally require a neighborhood around it rather than
observations exactly on it. See [Hentschel (2026a), Sections 2–3](references.md).

CBaseline localizes **in prediction space**, using a fitted metric that
accounts for scale and redundant output directions. It uses observed feature
rows without synthesizing new reference cases. High-dimensional features do
not enter the localization metric, though prediction evaluation, storage, and
attribution still depend on feature dimension. Many independent outputs can
also make localization harder.

Calibration enforces a mean constraint; it does not make every retained row
predict exactly `f0`. Two neutral distributions may explain the same total
while allocating it differently. Localization identifies which reference
population, among those with that mean, is relevant to the question.

## What remains the attribution engine's responsibility

CBaseline does not compute attributions or change the fitted model. It
provides the distribution for averaging IG paths or defining a SHAP
background. Observed baseline endpoints do not imply that every point along
an IG path, or every SHAP hybrid feature vector, is an observed input.
Nor does a background choice turn a model explanation into a causal effect.

Keep one background fixed when comparing observations under the same question.
Changing `f0`, the reference sample, or the background construction changes
the reference population. Determinism is conditional on the fitted model,
reference sample including row order, and construction settings; it does not
remove randomness in model fitting or downstream approximation.

## Comparison with common references

| Reference | Mean prediction | Observed rows | Localized near `f0` |
| --- | --- | --- | --- |
| Mean input | Generally differs from the mean output | Not necessarily | No |
| Full reference sample | Neutral at its own mean output | Yes | No |
| Random subsample | Matches the population mean only in expectation | Yes | Not generally |
| CBaseline calibrated | Matches feasible `f0` to numerical tolerance | Yes | Yes, subject to widening |
| CBaseline equal-weight | Approximate; inspect residual | Yes | Yes, with a neutrality-directed slide |

The [two papers](references.md) develop the reference-distribution viewpoint
for Integrated Gradients and Shapley attribution respectively.

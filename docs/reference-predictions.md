# Choosing the reference prediction

`f0` is required. It expresses the question: why this output rather than the
reference output? Specify the reference population and the output scale
before constructing a background.

## Regression

A practical model-relative benchmark is the mean fitted prediction on a
representative reference sample:

```python
predictions = model.predict(X_reference)
f0 = float(predictions.mean())
```

The canonical IG paper motivates the uninformed benchmark as the loss-optimal
constant prediction: for squared-error regression, the unconditional mean
outcome. The mean model prediction estimates a model-relative benchmark;
it need not equal the mean outcome in an uncalibrated model. An externally
chosen target is also allowed, subject to support and feasibility.

## Binary classification

Use the score being attributed, preferably the logit for logistic models.
Do not substitute hard class labels. A library's `decision_function` is not
universally a logit: check the model's output convention.

For a reference probability `p0` strictly between zero and one, the
corresponding logistic score is:

```python
import numpy as np

p0 = 0.30  # an explicitly chosen reference class probability
f0 = float(np.log(p0) - np.log1p(-p0))
```

The canonical probability benchmark can be estimated from class frequencies
in the chosen population, or from mean predicted probabilities for a
model-relative comparison. State which estimate you use.

For a comparison against the **mean score** instead, use:

```python
scores = model.decision_function(X_reference)
f0 = float(scores.mean())
```

These choices differ: `logit(mean probability)` generally does not equal
`mean logit`. Matching the mean logit also does not force the mean background
probability to equal `p0`.

For a decision-boundary question, use the score corresponding to the actual
probability threshold. A threshold of 0.5 corresponds to `f0=0.0` for a
logistic score; a different threshold gives a different reference.

## Multiclass classification

Use a full matrix of logits `L` with shape `(n, K)`, ordered consistently with
the model's classes. Center each row to remove the arbitrary common offset:

```python
Z = L - L.mean(axis=1, keepdims=True)
```

For a reference class-probability vector `p0`, all entries must be positive
and sum to one. Its centered-logit representative is:

```python
p0 = np.array([0.2, 0.3, 0.5])
z_star = np.log(p0) - np.log(p0).mean()
```

This is the probability-neutral benchmark used in the canonical IG paper.
Zero-frequency classes require an explicit population or smoothing decision;
there is no finite logit for a zero probability. Do not silently clip it.

For the distinct comparison against mean centered model scores, use
`z_star = Z.mean(axis=0)`. In general, this is not the centered logarithm of
the mean predicted probability vector. See [the IG paper, Section 2.1](references.md).

Construct **one joint background**, retaining the class coordinates:

```python
from cbaseline import background

bg = background(Z, z_star, X_reference, weighting="equal", size=200)
```

Use at least 200 reference rows for this example. A calibrated alternative
uses `weighting="calibrated"` without `size`, provided the target is feasible.
The metric removes the redundant common direction, giving at most `K-1`
independent dimensions; other rank deficiencies can reduce this further.
Separate scalar backgrounds for each class would change the reference
population across outputs. Reuse the joint background for all classes.

The attribution engine must explain the **same centered scores**. For an
engine that returns all raw-logit attributions, centering across its class
axis can give centered-logit attributions by linearity, but center base values
and output predictions as well and check the engine's array conventions.
Do not assume `predict` or `predict_proba` returns centered logits.

## Feasibility and contextual questions

A mean prediction from the reference sample lies in its convex hull. An
external target or a transformed probability benchmark need not. Localized
support may be narrower still; calibration may need to widen or may fail.
See [diagnostics](diagnostics.md).

A subgroup comparison should use that subgroup as the reference sample and
an appropriate subgroup target. This is a contextual question incorporating
additional information. Record it as such rather than silently changing the
population behind the unconditional benchmark.

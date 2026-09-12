---
myst:
  html_meta:
    description: "CBaseline constructs empirical reference backgrounds for Integrated Gradients and SHAP, with calibrated weights or deterministic equal-weight selection."
---

# CBaseline documentation

CBaseline is a Python package that constructs empirical reference background
distributions for Integrated Gradients, SHAP, and other compatible attribution
methods. Install and import it as `cbaseline`. Given aligned model predictions,
a reference output `f0`, and observed feature rows, `background(...)` returns
selected rows, weights, and diagnostics. Your attribution engine then computes
the feature contributions.

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
With a background distribution $Q$, complete attributions explain
$f(x)-\mathbb{E}_Q[f(X)]$, so neutrality is the condition
$\mathbb{E}_Q[f(X)]=f_0$.

CBaseline satisfies both using observed rows only. It localizes in prediction
space under a fitted metric that accounts for output scale and redundant
directions, then calibrates by exponential tilting until the weighted mean
output equals $f_0$ to numerical tolerance. Equal-weight backgrounds
approximate neutrality with a fixed-size selection and report the residual they
achieve.

![Observed inputs and the prediction-neutral manifold](Figure_NeutralManifold.svg)

The blue points are observed inputs; the red curve is the neutral manifold,
$\mathcal{M}_0=\{x:f(x)=f_0\}$. CBaseline builds its background from observed
cases near this curve in prediction space, illustrated by the shaded band.
The background therefore represents realistic reference inputs concentrated
around the prediction being used for comparison.

The construction is method-agnostic. The same background serves Integrated
Gradients, interventional Shapley attribution — where the weighted mean is the
empty-coalition value $v(\emptyset)$ — and any implementation that accepts a
background matrix. Calibrated weighted backgrounds fit naturally into the
**UnifiedIG / TreeIG stack**. Deterministic equal-weight backgrounds give
**SHAP** a first-class workflow through its standard background-matrix
interfaces.

See [prediction-neutral backgrounds](concepts.md) for why both requirements are
needed and how they constrain one another.

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

## The supporting papers

- Hentschel (2026a), [**Canonical Integrated Gradients: Expectations over Neutral Prediction Baselines**](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26h.pdf): the reference-distribution formulation and calibrated estimation.
- Hentschel (2026b), [**A Canonical Background Distribution for Shapley Attribution**](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26i.pdf): the Shapley formulation and equal-weight construction.

The guide develops the practical ideas alongside the papers; see
[papers and citation](references.md) for BibTeX and further reading.

## Where to start

[Background modes](backgrounds.md) and [diagnostics](diagnostics.md)
explain feasibility, residuals, and failure behavior. For automated readers,
[llms.txt](https://ludgerhentschel.github.io/cbaseline/llms.txt) maps the guides,
complete examples, and rendered API reference.

Use [getting started](getting-started.md) for a runnable construction,
[integrations](integrations.md) for attribution examples, and
[choosing the reference prediction](reference-predictions.md) before adapting
an example to classification.

## Related projects

CBaseline can be used independently of the Integrated Gradients stack. Choose an
attribution engine that consumes the background in the intended form:

| Package | Role |
|---|---|
| [UnifiedIG](https://ludgerhentschel.github.io/unifiedig/) (`unifiedig`) | A common Integrated Gradients attribution interface across supported model families; accepts CBaseline backgrounds directly. |
| [TreeIG](https://ludgerhentschel.github.io/treeig/) (`treeig`) | Direct tree-path attribution for supported models; accepts CBaseline rows and weights through its background interface. |
| [skgrad](https://ludgerhentschel.github.io/skgrad/) (`skgrad`) | Analytic input gradients and Jacobians for supported scikit-learn models; a derivative component rather than an attribution engine. |

Standard SHAP matrix-background workflows use the equal-weight construction.

See [the Integrated Gradients stack](ig-stack.md) for how the packages compose.

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
concepts
reference-predictions
backgrounds
integrations
diagnostics
ig-stack
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
references
building
publishing
```

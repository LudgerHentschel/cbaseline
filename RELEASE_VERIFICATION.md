# Release verification

This release candidate was checked on 2026-07-11.

## Changes in this pass

- `background()` now defaults to a deterministic equal-weight background of 100 rows.
- Weighted modes remain explicit through `weighting="kernel"` and `weighting="calibrated"`.
- The long implementation constructors remain importable but are not documented as the normal API.
- Package metadata, Python version, dependencies, citation metadata, and license wording were corrected.
- `numba`, `scikit-learn`, `xgboost`, and `lightgbm` were removed from the default development/runtime dependency sets.
- `cbaseline.__version__` was added.
- Build/check and publishing were separated into `release.sh` and `publish.sh`.
- `.gitignore` and `MANIFEST.in` were added.
- README claims now distinguish exact calibrated neutrality from finite-sample equal-weight approximation.
- Tests were made warning-clean.

## Verification results

- `python -m pytest -q`: 31 passed, 0 warnings.
- `python -m build`: wheel and source distribution built successfully.
- `python -m twine check dist/*`: both distributions passed.
- Installed-wheel smoke test: passed for default equal weighting and calibrated weighting.
- Example script: passed.
- `CITATION.cff`: parsed successfully.

The built wheel and source distribution are supplied separately from this source archive.

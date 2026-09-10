# Changelog

Notable changes to CBaseline. Releases through 0.1.2 predate this file; see the
Git history and [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md) for those.

## Unreleased

No changes to `cbaseline/` since 0.1.2. Backgrounds constructed with this
version are identical to those from 0.1.2.

### Packaging

- Remove the unused `numba` runtime dependency. CBaseline never imported it, so
  this is a metadata correction rather than a behavioral change. NumPy and
  SciPy are now the only runtime requirements, and installing CBaseline no
  longer pulls a JIT compiler.
- Correct the project URLs to the `LudgerHentschel` organization and add the
  documentation URL.
- Add a `docs` extra, `scripts/check_release_version.py`, and a release-version
  test so a Git tag and the package version cannot disagree.
- Standardize tag-driven PyPI and GitHub releases.

### Documentation

- Publish a Sphinx documentation site covering concepts, background
  construction, reference predictions, diagnostics, integrations, and the API.
- Lead the README and documentation landing page with the two requirements a
  reference must satisfy: neutrality, meaning its mean model output equals
  `f0`, and localization, meaning it is assembled from cases the model already
  predicts near `f0`.
- Show in `concepts.md` why neutrality alone is insufficient: a reweighting
  that pairs high- and low-prediction cases averages to `f0` while containing
  no case the model predicts anywhere near it.
- Record that NumPy and SciPy are the only runtime dependencies.

# Testing CBaseline

Run the complete test suite from the repository root with:

```bash
python -m pytest -q
```

The tests are organized by statistical responsibility:

- `test_api.py`: single-constructor routing and common result interface;
- `test_geometry.py`: whitening, centered-logit rank, affine support;
- `test_kernels.py`: kernel values, normalization, bandwidth rules;
- `test_calibration.py`: exponential calibration and failure reporting;
- `test_uniform.py`: deterministic equal-weight selection and diagnostics;
- `test_weighted.py`: scalar and vector weighted neutrality;
- `test_public_api.py`: package-root and compatibility imports.

The old script-style tests are not used because they exercise removed
cube-sampling and trimming implementations rather than the current
translated-neighborhood and exponential-calibration algorithms.


## Release checks

Run `./release.sh` to execute tests, build the distributions, and run
`twine check`. Publishing is deliberately separate in `./publish.sh`.

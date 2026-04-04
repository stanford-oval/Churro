# Contributing

Use Pixi for local development.

## Setup

Install Pixi, then create the default environment from the repo root:

```bash
pixi install
```

## Common Commands

Run these from the repo root:

```bash
pixi run format
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
pixi run docs-build
pixi run docs-serve
pixi run package-check
```

What they do:

- `format`: run Ruff formatting on `src/` and `tests/`
- `lint`: run Ruff lint checks on `src/` and `tests/`
- `typecheck`: run `ty` on `src/` and `tests/`
- `test`: run the unit and offline test suite
- `coverage`: run the full coverage report used for release audits
- `docs-build`: build the static documentation site into `docs/_build/html`
- `docs-serve`: start a live-reload documentation preview server on an automatically selected free port
- `package-check`: build and audit the wheel and sdist that would be published to PyPI

## Live Integration Tests

Live provider integration tests are skipped by default.

- `tests/test_page_detection_integration.py` uses `CHURRO_RUN_LIVE_VERTEX_TESTS`
- `tests/test_hf_ocr_integration.py` uses `CHURRO_RUN_LIVE_HF_TESTS`

These tests may require credentials, external services, and billable APIs. Do not enable them in routine local runs unless you intend to use those services.

## Package Check

`pixi run package-check` runs the repo-local publish gate defined in `scripts/package_check.py`. It validates the built artifacts, not the editable checkout.

The current package check does all of the following:

- removes stale `build/`, `dist/`, and generated egg-info directories
- builds a fresh wheel and sdist
- runs `twine check` on both artifacts
- verifies the wheel metadata, including project URLs, extras, and the `churro-ocr` console entry point
- verifies that repo-only content such as `tests/`, `tooling/`, `scripts/`, and release audit notes do not ship inside the artifacts
- smoke-installs the base wheel and base sdist in clean virtual environments and checks `import churro_ocr`, `import churro_ocr.providers`, and `python -m churro_ocr --help`
- smoke-installs the lightweight `local` and `pdf` extras from the built wheel
- audits direct dependency licenses for incompatible or unknown licenses

If `package-check` fails, treat that as a release blocker until the artifact or documentation contract is fixed.

For benchmark runs and evaluation outputs from a repo checkout, see [Benchmarking](benchmarking.md).

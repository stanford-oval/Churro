# Contributing to Churro

Use Pixi for local development. Treat this checkout as the repo root.

## Setup

Install Pixi, then create the default environment from the checkout root:

```bash
pixi install
```

## Common Commands

Run these from the checkout root:

```bash
pixi run format
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
pixi run package-check
```

What they do:
- `format`: run Ruff formatting on `src/` and `tests/`
- `lint`: run Ruff lint checks on `src/` and `tests/`
- `typecheck`: run `ty` on `src/` and `tests/`
- `test`: run the unit and offline test suite
- `coverage`: run the full coverage report used for release audits
- `package-check`: build and audit the wheel and sdist that would be published to PyPI

## Benchmarking and Release Checks

Repo-only benchmarking, evaluation outputs, and the package audit are documented in `REPO_WORKFLOWS.md`.

## Live Integration Tests

Live provider integration tests are skipped by default.

- `tests/test_page_detection_integration.py` uses `CHURRO_RUN_LIVE_VERTEX_TESTS`
- `tests/test_hf_ocr_integration.py` uses `CHURRO_RUN_LIVE_HF_TESTS`

These tests may require credentials, external services, and billable APIs. Do not enable them in routine local runs unless you intend to use those services.

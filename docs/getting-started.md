# Getting Started

`churro-ocr` is the Python package and CLI for OCR on one-page images, photographed spreads, and PDFs.
This page takes the shortest path to one successful local transcription before branching into task-specific guides.

## Prerequisites

- Python 3.12 or newer
- `uv` available on `PATH`

## Install the CLI

For the CLI-first workflow used in this guide, install Churro with UV as a tool.

```bash
uv tool install churro-ocr
```

If you are adding `churro-ocr` to a project instead, use `uv add churro-ocr` and prefix the CLI commands below with `uv run`.

## Install the First Runtime

The canonical getting-started path uses the local Hugging Face backend and the `stanford-oval/churro-3B` model.

```bash
churro-ocr install hf
```

Use `--torch-backend` with `hf` when you need a specific PyTorch build:

```bash
churro-ocr install hf --torch-backend cu126
```

For hosted providers, self-hosted OpenAI-compatible servers, Azure, Mistral, or PDF support, continue with [Providers And Configuration](guides/providers.md).

## First Successful Run

```bash
churro-ocr transcribe \
  --image scan.png \
  --backend hf \
  --model stanford-oval/churro-3B
```

This prints the OCR text to stdout.
Add `--output output.txt` when you want the CLI to write the text to a file instead.

## If You're Writing Python Next

| Goal | Start with |
| --- | --- |
| OCR one page or one image | `OCRClient` |
| Detect page crops only | `DocumentPageDetector` |
| Run an end-to-end image or PDF OCR workflow | `DocumentOCRPipeline` |
| Tune provider options directly | `build_ocr_backend(...)` + `OCRBackendSpec` |

For the page-and-pipeline mental model behind those types, read [Core Concepts](core-concepts.md).

## Where To Go Next

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} CLI
:link: cli
:link-type: doc

Stay in the shell for OCR checks, page extraction, and runtime installs.
:::

:::{grid-item-card} OCR Workflows
:link: guides/ocr-workflows
:link-type: doc

Use the Python API for single-page OCR, PDFs, photographed spreads, and async flows.
:::

:::{grid-item-card} Page Detection
:link: guides/page-detection
:link-type: doc

Extract page crops without OCR, or choose a detector backend for boundary discovery.
:::

:::{grid-item-card} Providers And Configuration
:link: guides/providers
:link-type: doc

Choose another backend, install its runtime, and see minimal provider setup examples.
:::

:::{grid-item-card} Core Concepts
:link: core-concepts
:link-type: doc

Learn the `DocumentPage` and pipeline model that ties the APIs together.
:::
::::

## Working From the Source Code

If you are developing from a clone instead of installing from PyPI, use the contributor instructions in [Contributing](contributing.md).

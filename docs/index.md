# Churro OCR Documentation

Churro OCR is a Python 3.12+ OCR toolkit for historical document transcription.
The docs start with a CLI-first local workflow, then branch into task guides and deeper reference material.

## Quick Start

```bash
uv tool install churro-ocr
churro-ocr install hf
churro-ocr transcribe \
  --image scan.png \
  --backend hf \
  --model stanford-oval/churro-3B
```

Use [Getting Started](getting-started.md) for the full setup flow and first-run guidance.

## Start Here

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Install the CLI, install the first runtime, and verify one successful OCR run.
:::
::::

## Common Tasks

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} CLI
:link: cli
:link-type: doc

Run `transcribe`, `extract-pages`, and runtime installs from the shell.
:::

:::{grid-item-card} OCR Workflows
:link: guides/ocr-workflows
:link-type: doc

Choose the right Python workflow for single-page images, PDFs, and photographed spreads.
:::

:::{grid-item-card} Page Detection
:link: guides/page-detection
:link-type: doc

Extract page crops without OCR, or choose a detector backend for layout discovery.
:::

:::{grid-item-card} Providers And Configuration
:link: guides/providers
:link-type: doc

Choose a backend, install its runtime, and see minimal provider setup examples.
:::
::::

## Learn More

- [Benchmark Snapshot](leaderboard.md)
- [Benchmarking Guide](benchmarking.md)
- [Paper](https://arxiv.org/abs/2509.19768)
- [Dataset](https://huggingface.co/datasets/stanford-oval/churro-dataset)
- [Model](https://huggingface.co/stanford-oval/churro-3B)
- [GitHub Repository](https://github.com/stanford-oval/Churro)

```{toctree}
:hidden:
:maxdepth: 1
:caption: Start Here

Overview <self>
Getting Started <getting-started>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Common Tasks

cli
guides/ocr-workflows
guides/page-detection
guides/providers
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Advanced

core-concepts
guides/advanced-customization
guides/historical-document-xml
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Benchmark

leaderboard
benchmarking
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reference

api/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

contributing
```

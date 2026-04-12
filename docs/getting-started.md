# Getting Started

`churro-ocr` is the Python package and CLI for running CHURRO-style OCR workflows on one image, photographed spreads, and PDFs. The PyPI package name is `churro-ocr`, and the Python import package is `churro_ocr`.

## Which API Should You Use?

| Goal | API |
| --- | --- |
| OCR one page or one image | `OCRClient` |
| Detect page crops only | `DocumentPageDetector` |
| Run an end-to-end image or PDF OCR workflow | `DocumentOCRPipeline` |
| Tune provider options directly | `build_ocr_backend(...)` + `OCRBackendSpec` |

## Install

For the CLI-first workflow used in this guide, install Churro with UV as a tool.
Python 3.12 or newer is required.

```bash
uv tool install churro-ocr
```

If you are adding `churro-ocr` to a project instead, use `uv add churro-ocr` and prefix the CLI commands below with `uv run`.

Then install the runtime for the backend you plan to use:

```bash
churro-ocr install llm
churro-ocr install hf
churro-ocr install local
churro-ocr install pdf
churro-ocr install all
```

Common runtime targets:

- `llm`: hosted multimodal OCR through LiteLLM-backed providers
- `hf`: local Hugging Face OCR plus a PyTorch runtime
- `local`: clients for OpenAI-compatible local or self-hosted servers
- `pdf`: PDF rasterization support for `process_pdf_*` and `extract-pages --pdf`
- `all`: every optional runtime in one command

Use `--torch-backend` with `hf` or `all` when you need a specific PyTorch build:

```bash
churro-ocr install hf --torch-backend cu126
```

For the full provider/runtime matrix, use [Providers And Configuration](guides/providers.md).

## First OCR Example

Use `OCRClient` when your input is already one page per image.
This example uses `provider="litellm"`, so install the `llm` runtime first.

```python
from churro_ocr.ocr import OCRClient
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="litellm",
        model="vertex_ai/gemini-2.5-flash",
    )
)

page = OCRClient(backend).ocr_image(image_path="scan.png")

print(page.text)
print(page.provider_name)
print(page.model_name)
```

When an API accepts both `image` and `image_path`, pass exactly one of them.

## Quick CLI Sanity Check

Use the CLI when you want to confirm a model or backend before writing Python code.

```bash
uv tool install churro-ocr
churro-ocr install hf
churro-ocr transcribe \
  --image scan.png \
  --backend hf \
  --model stanford-oval/churro-3B
```

## Working From A Repo Checkout

If you are developing from a clone instead of installing from PyPI, use the contributor instructions in [Contributing](contributing.md).

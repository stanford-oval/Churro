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

Use UV as the supported install path.

```bash
uv tool install churro-ocr
# or, in a project:
uv add churro-ocr
```

Then install the runtime for the backend you plan to use:

```bash
uv run churro-ocr install llm
uv run churro-ocr install hf
uv run churro-ocr install local
```

If you installed the CLI with `uv tool install churro-ocr`, drop the `uv run` prefix.
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

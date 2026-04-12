# churro-ocr

`churro-ocr` is a Python toolkit for OCR and page detection on historical documents.

Full documentation and project overview live at https://stanford-oval.github.io/Churro/.

## Install

For the CLI-first workflow used in the docs, install Churro with UV as a tool.

```bash
uv tool install churro-ocr
```

If you are adding `churro-ocr` to a project instead, use `uv add churro-ocr` and prefix CLI commands with `uv run`.

Runtime setup and provider-specific install commands are in
[Getting Started](https://stanford-oval.github.io/Churro/getting-started.html)
and
[Providers And Configuration](https://stanford-oval.github.io/Churro/guides/providers.html).

## Which API Should You Use?

| Goal | API |
| --- | --- |
| OCR one page or one image | `OCRClient` |
| Detect page crops only | `DocumentPageDetector` |
| Run an end-to-end image or PDF OCR workflow | `DocumentOCRPipeline` |
| Tune backend/provider options directly | `build_ocr_backend(...)` + `OCRBackendSpec` |

## Quick Start

This example assumes you already installed the runtime for the provider you want to use.

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
```

## More

- Overview: https://stanford-oval.github.io/Churro/
- Getting started: https://stanford-oval.github.io/Churro/getting-started.html
- Benchmark leaderboard: https://stanford-oval.github.io/Churro/leaderboard.html
- Provider setup: https://stanford-oval.github.io/Churro/guides/providers.html
- CLI docs: https://stanford-oval.github.io/Churro/cli.html
- API reference: https://stanford-oval.github.io/Churro/api/index.html

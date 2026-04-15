# OCR Workflows

Use this page after [Getting Started](../getting-started.md) when you want Python recipes instead of shell commands.
For page crops without OCR, use [Page Detection](page-detection.md).

## Choose A Workflow

| Input shape | Start with | Why |
| --- | --- | --- |
| One image already equals one page | `OCRClient` | simplest OCR path with no page detection |
| One image may contain multiple pages | `DocumentOCRPipeline` with a detection backend | page detection and OCR stay in one workflow |
| A PDF | `DocumentOCRPipeline` | rasterization and page OCR stay in one pipeline |
| You only want page crops, not text yet | `DocumentPageDetector` | detection-only workflow |

## OCR One Image

Use `OCRClient` when each input image already represents one page.
Install `hf` first if you have not already.

```python
from churro_ocr.ocr import OCRClient
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="hf",
        model="stanford-oval/churro-3B",
    )
)

page = OCRClient(backend).ocr_image(image_path="scan.png")

print(page.text)
print(page.provider_name)
print(page.model_name)
```

When an API accepts both `image` and `image_path`, pass exactly one of them.

## OCR A PDF

Install the `pdf` runtime first.
If you also want local Hugging Face OCR for PDFs, install `all` or install both `hf` and `pdf`.

```bash
churro-ocr install pdf
```

Then `DocumentOCRPipeline` can rasterize a PDF and OCR each page.

```python
from churro_ocr import DocumentOCRPipeline
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

pipeline = DocumentOCRPipeline(
    build_ocr_backend(
        OCRBackendSpec(
            provider="hf",
            model="stanford-oval/churro-3B",
        )
    ),
    max_concurrency=4,
)

result = pipeline.process_pdf_sync("document.pdf", dpi=300, trim_margin=30)

for page in result.pages:
    print(page.page_index, page.text)
```

## Detect Pages And OCR A Photographed Spread

This flow is useful when one input image contains multiple pages.
Install `llm` first if you want an LLM-based detector.

```python
from pathlib import Path

from churro_ocr import DocumentOCRPipeline, PageDetectionRequest
from churro_ocr.providers import (
    LLMPageDetector,
    LiteLLMTransportConfig,
    OCRBackendSpec,
    build_ocr_backend,
)

INPUT_IMAGE = Path("spread.jpg")
OUTPUT_DIR = Path("output")
MODEL = "vertex_ai/gemini-2.5-flash"

transport = LiteLLMTransportConfig()

ocr_backend = build_ocr_backend(
    OCRBackendSpec(
        provider="litellm",
        model=MODEL,
        transport=transport,
    )
)

pipeline = DocumentOCRPipeline(
    ocr_backend,
    detection_backend=LLMPageDetector(
        model=MODEL,
        transport=transport,
    ),
    max_concurrency=4,
)

result = pipeline.process_image_sync(
    PageDetectionRequest(
        image_path=INPUT_IMAGE,
        trim_margin=20,
    )
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for page in result.pages:
    image_path = OUTPUT_DIR / f"page_{page.page_index:04d}.png"
    text_path = OUTPUT_DIR / f"page_{page.page_index:04d}.txt"

    page.image.save(image_path)
    text_path.write_text(page.text or "", encoding="utf-8")
```

If your input is already one page per image, skip the `detection_backend` and use `OCRClient`.

## Async Entry Points

Every sync helper has an async equivalent.

### Async OCR For One Page

```python
import asyncio

from churro_ocr.ocr import OCRClient
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend


async def main() -> None:
    backend = build_ocr_backend(
        OCRBackendSpec(
            provider="hf",
            model="stanford-oval/churro-3B",
        )
    )
    page = await OCRClient(backend).aocr_image(
        image_path="scan.png",
        page_index=3,
        source_index=7,
        metadata={"job_id": "demo"},
    )
    print(page.text)
    print(page.metadata)


asyncio.run(main())
```

### Async Document OCR

```python
import asyncio

from churro_ocr import DocumentOCRPipeline, PageDetectionRequest
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend


async def main() -> None:
    pipeline = DocumentOCRPipeline(
        build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model="stanford-oval/churro-3B",
            )
        ),
        max_concurrency=4,
    )
    image_result = await pipeline.process_image(
        PageDetectionRequest(image_path="scan.png"),
        ocr_metadata={"job_id": "demo-image"},
    )
    print(image_result.texts())


asyncio.run(main())
```

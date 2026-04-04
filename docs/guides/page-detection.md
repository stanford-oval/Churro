# Page Detection

Use `DocumentPageDetector` when you want page crops without OCR.

## Which Detector Should You Use?

| Detector | Good default when |
| --- | --- |
| none | you want the whole image or rasterized PDF page treated as a single page |
| Azure | you want Azure Document Intelligence to find pages for you |
| LLM | you want a multimodal model to infer page boundaries from an image |

## Default Detector

When you do not provide a backend, the detector treats the whole image as one page.

```python
from churro_ocr.page_detection import DocumentPageDetector, PageDetectionRequest

result = DocumentPageDetector().detect_image_sync(
    PageDetectionRequest(image_path="scan.png")
)

for page in result.pages:
    print(page.page_index, page.image.size)
```

## Azure-backed Page Detection

Use `AzurePageDetector` when you want Azure Document Intelligence to find pages from an image or rasterized PDF page.

```python
from churro_ocr.page_detection import DocumentPageDetector, PageDetectionRequest
from churro_ocr.providers import AzurePageDetector

detector = DocumentPageDetector(
    backend=AzurePageDetector(
        endpoint="https://<resource>.cognitiveservices.azure.com/",
        api_key="<azure-doc-intelligence-key>",
    )
)

result = detector.detect_image_sync(
    PageDetectionRequest(image_path="scan.png", trim_margin=30)
)
```

## LLM-based Page Detection

Use `LLMPageDetector` when you want a multimodal model to identify page boundaries.

```python
from churro_ocr.page_detection import DocumentPageDetector, PageDetectionRequest
from churro_ocr.providers import LLMPageDetector, LiteLLMTransportConfig

detector = DocumentPageDetector(
    backend=LLMPageDetector(
        model="vertex_ai/gemini-2.5-flash",
        transport=LiteLLMTransportConfig(),
    )
)

result = detector.detect_image_sync(
    PageDetectionRequest(image_path="spread.jpg", trim_margin=20)
)
```

## Important Inputs

- `image` and `image_path` are mutually exclusive on `PageDetectionRequest`.
- `trim_margin` expands the detected crop by that many pixels and clips the result to the image bounds.
- `detect_pdf(...)` rasterizes each PDF page before detection, so `dpi` only affects PDF workflows.

Pair page detection with OCR through [DocumentOCRPipeline](ocr-workflows.md). Use the [API Reference](../api/page_detection.md) when you need exact type definitions.

# Getting Started

`churro-ocr` is the Python package and CLI for running CHURRO-style OCR workflows on one image, photographed spreads, and PDFs. The PyPI package name is `churro-ocr`, and the Python import package is `churro_ocr`.

## Which API Should You Use?

| Goal | API |
| --- | --- |
| OCR one page or one image | `OCRClient` |
| Detect page crops only | `DocumentPageDetector` |
| Run an end-to-end image or PDF OCR workflow | `DocumentOCRPipeline` |
| Tune provider options directly | `build_ocr_backend(...)` + `OCRBackendSpec` |

## Install Only What You Need

```bash
pip install churro-ocr
pip install "churro-ocr[llm]"
pip install "churro-ocr[local]"
pip install "churro-ocr[hf]"
pip install "churro-ocr[vllm]"
pip install "churro-ocr[azure]"
pip install "churro-ocr[mistral]"
pip install "churro-ocr[pdf]"
pip install "churro-ocr[all]"
```

## Provider Extras

| Extra | Use it when |
| --- | --- |
| `llm` | you want hosted multimodal OCR and LLM-based page detection through LiteLLM |
| `local` | you have a local or self-hosted OpenAI-compatible server |
| `hf` | you want local Transformers inference in-process |
| `vllm` | you want higher-throughput local serving |
| `azure` | you want Azure Document Intelligence OCR or layout detection |
| `mistral` | you want Mistral OCR |
| `pdf` | you want PDF rasterization through `pypdfium2` |
| `all` | you want every supported backend and utility extra |

## First OCR Example

Use `OCRClient` when your input is already one page per image.

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
churro-ocr transcribe \
  --image scan.png \
  --backend hf \
  --model stanford-oval/churro-3B
```

## Working From A Repo Checkout

If you are developing from a clone instead of installing from PyPI, use the contributor instructions in [Contributing](contributing.md).

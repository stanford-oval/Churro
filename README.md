# churro-ocr

`churro-ocr` is a Python toolkit for OCR and page detection on historical documents.

It gives you one consistent interface whether you are using:
- hosted multimodal models through LiteLLM
- a local OpenAI-compatible server
- local Hugging Face or vLLM models
- Azure Document Intelligence
- Mistral OCR

If you are new to the library, the shortest path is:
1. Install the extra for the backend you want.
2. Create an `OCRBackendSpec`.
3. Build it with `build_ocr_backend(...)`.
4. Use `OCRClient` for one page or `DocumentOCRPipeline` for a full document.

The PyPI package name is `churro-ocr`. The Python import package is `churro_ocr`.

## Setup

Install only the pieces you need:

```bash
pip install churro-ocr
pip install "churro-ocr[llm]"
pip install "churro-ocr[local]"
pip install "churro-ocr[hf]"
pip install "churro-ocr[huggingface]"
pip install "churro-ocr[vllm]"
pip install "churro-ocr[azure]"
pip install "churro-ocr[mistral]"
pip install "churro-ocr[pdf]"
pip install "churro-ocr[all]"
```

What each extra is for:
- `llm`: hosted multimodal OCR and LLM-based page detection through LiteLLM
- `local`: OpenAI-compatible OCR servers
- `hf` and `huggingface`: equivalent extras for local Hugging Face Transformers OCR
- `vllm`: local vLLM OCR
- `azure`: Azure Document Intelligence OCR and page detection
- `mistral`: Mistral OCR
- `pdf`: PDF rasterization via pypdfium2
- `all`: everything above

Credential setup depends on the provider you choose:
- LiteLLM-backed models use LiteLLM's normal authentication flow.
- OpenAI-compatible servers usually need `api_base` and `api_key`.
- Azure Document Intelligence needs an endpoint and API key.
- Mistral needs an API key.

When you need to pass connection details directly, use `LiteLLMTransportConfig` or the provider-specific options dataclasses shown below.

If you are working from a repo checkout instead of installing from PyPI, use Pixi from the checkout root. The public workflow is `pixi run lint`, `pixi run test`, and `pixi run package-check`, not the internal Nx file.

## Core Idea

`churro-ocr` has a small set of concepts:

- `DocumentPage`: one page image plus optional OCR text and metadata
- `OCRBackendSpec`: a declarative description of which OCR backend to build
- `build_ocr_backend(...)`: the factory that turns a spec into a runnable OCR backend
- `OCRClient`: OCR for one image or one `DocumentPage`
- `DocumentPageDetector`: page detection only
- `DocumentOCRPipeline`: page detection plus OCR for a complete image or PDF flow, with bounded OCR concurrency

Most users do not need to think about prompts or preprocessing directly. `churro-ocr` ships built-in OCR model profiles and automatically applies the right prompt and output cleanup for:
- generic OCR models
- `stanford-oval/churro-3B`
- `kristaller486/dots.ocr-1.5`

## Which API Should I Use?

| Goal | API |
| --- | --- |
| OCR one page or one image | `OCRClient` |
| Detect page crops only | `DocumentPageDetector` |
| Run an end-to-end image/PDF OCR workflow | `DocumentOCRPipeline` |
| Try things from the shell | `churro-ocr` CLI |
| Tune backend/provider options directly | `build_ocr_backend(...)` + `OCRBackendSpec` |

## Which OCR Backend Should I Use?

| Provider | Install extra | Good default when |
| --- | --- | --- |
| `litellm` | `llm` | you want to use a hosted multimodal model through LiteLLM |
| `openai-compatible` | `local` | you have a local or self-hosted OpenAI-style server |
| `hf` | `hf` or `huggingface` | you want local Transformers inference in-process |
| `vllm` | `vllm` | you want higher-throughput local serving |
| `azure` | `azure` | you want Azure Document Intelligence OCR |
| `mistral` | `mistral` | you want Mistral's OCR API |

All of these backends use the same builder:

```python
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="litellm",
        model="vertex_ai/gemini-2.5-flash",
    )
)
```

## Quick Start: OCR One Image

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

## Async Entry Points

Every public sync helper has an async equivalent. Use the async forms when you are already inside an event loop or want to coordinate OCR with your own concurrency controls.

Async OCR for a single page or image:

```python
import asyncio

from churro_ocr.ocr import OCRClient
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend


async def main() -> None:
    backend = build_ocr_backend(
        OCRBackendSpec(
            provider="litellm",
            model="vertex_ai/gemini-2.5-flash",
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

Async page detection for images or PDFs:

```python
import asyncio

from churro_ocr.page_detection import DocumentPageDetector, PageDetectionRequest


async def main() -> None:
    detector = DocumentPageDetector()
    image_result = await detector.detect_image(
        PageDetectionRequest(image_path="spread.jpg", trim_margin=20)
    )
    pdf_result = await detector.detect_pdf("document.pdf", dpi=300, trim_margin=20)
    print(image_result.source_type, len(image_result.pages))
    print(pdf_result.source_type, len(pdf_result.pages))


asyncio.run(main())
```

Async end-to-end document OCR:

```python
import asyncio

from churro_ocr import DocumentOCRPipeline, PageDetectionRequest
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend


async def main() -> None:
    pipeline = DocumentOCRPipeline(
        build_ocr_backend(
            OCRBackendSpec(
                provider="litellm",
                model="vertex_ai/gemini-2.5-flash",
            )
        ),
        max_concurrency=4,
    )
    image_result = await pipeline.process_image(
        PageDetectionRequest(image_path="spread.jpg", trim_margin=20),
        ocr_metadata={"job_id": "demo-image"},
    )
    pdf_result = await pipeline.process_pdf(
        "document.pdf",
        dpi=300,
        trim_margin=20,
        ocr_metadata={"job_id": "demo-pdf"},
    )
    print(image_result.texts())
    print(pdf_result.as_ocr_results()[0].metadata)


asyncio.run(main())
```

## Full Example: Detect Pages and OCR a Photographed Spread

This example shows the complete flow for a single image that may contain multiple pages.
It uses an LLM page detector to find page crops, then OCRs each crop with the same model family.

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

    print(
        f"page={page.page_index} "
        f"provider={page.provider_name} "
        f"model={page.model_name} "
        f"text_file={text_path}"
    )
```

If your input is already one page per image, skip the `detection_backend` and use `OCRClient` instead.

## Backend Recipes

### LiteLLM

Use this for hosted multimodal models routed through LiteLLM.

```python
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="litellm",
        model="vertex_ai/gemini-2.5-flash",
    )
)
```

If you need to override connection details or completion settings:

```python
from churro_ocr.providers import LiteLLMTransportConfig, OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="litellm",
        model="gpt-4.1-mini",
        transport=LiteLLMTransportConfig(
            api_base="https://example.invalid/v1",
            api_key="secret",
            api_version="2025-01-01-preview",
            completion_kwargs={"temperature": 0},
        ),
    )
)
```

### OpenAI-Compatible Servers

Use this for local or self-hosted servers that expose an OpenAI-style API.

```python
from churro_ocr.providers import (
    LiteLLMTransportConfig,
    OCRBackendSpec,
    build_ocr_backend,
)

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="openai-compatible",
        model="local-model",
        transport=LiteLLMTransportConfig(
            api_base="http://127.0.0.1:8000/v1",
            api_key="dummy",
        ),
    )
)
```

### Hugging Face

Use this for local Transformers inference inside the current Python process.

```python
from churro_ocr.providers import HuggingFaceOptions, OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="hf",
        model="stanford-oval/churro-3B",
        options=HuggingFaceOptions(
            model_kwargs={"device_map": "auto", "torch_dtype": "auto"},
        ),
    )
)
```

### vLLM

Use this for higher-throughput local inference.

```python
from churro_ocr.providers import OCRBackendSpec, VLLMOptions, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="vllm",
        model="stanford-oval/churro-3B",
        options=VLLMOptions(),
    )
)
```

### Azure Document Intelligence

Use this when you want Azure's OCR stack instead of a general multimodal model.

```python
from churro_ocr.providers import (
    AzureDocumentIntelligenceOptions,
    OCRBackendSpec,
    build_ocr_backend,
)

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="azure",
        options=AzureDocumentIntelligenceOptions(
            endpoint="https://<resource>.cognitiveservices.azure.com/",
            api_key="<azure-doc-intelligence-key>",
        ),
    )
)
```

### Mistral OCR

```python
from churro_ocr.providers import MistralOptions, OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="mistral",
        model="mistral-ocr-latest",
        options=MistralOptions(api_key="<mistral-api-key>"),
    )
)
```

## Backend Spec Reference

`OCRBackendSpec` is the builder input shared across all OCR providers.

| Field | Meaning |
| --- | --- |
| `provider` | One of `litellm`, `openai-compatible`, `azure`, `mistral`, `hf`, or `vllm`. |
| `model` | Required for `litellm`, `openai-compatible`, `hf`, and `vllm`. Optional for `azure`. Defaults to `mistral-ocr-latest` when omitted for `mistral`. |
| `profile` | `None`, a built-in profile name, or a custom `OCRModelProfile`. If omitted, `resolve_ocr_profile(...)` picks the built-in profile for the model id when available, otherwise the generic default profile. |
| `transport` | Shared request transport config for LiteLLM-based providers. Use this for `litellm`, `openai-compatible`, or LLM page detection. |
| `options` | Provider-specific dataclass. Use the options type that matches `provider`. |

Provider option dataclasses:

| Type | Used by | Required fields | Defaults and notes |
| --- | --- | --- | --- |
| `LiteLLMTransportConfig` | `litellm`, `openai-compatible`, `LLMPageDetector` | None at the dataclass level. `openai-compatible` usually needs `api_base` and `api_key`. | `completion_kwargs={}`, `cache_dir=None`, `image_detail=None`, `api_version=None`. |
| `OpenAICompatibleOptions` | `openai-compatible` | None | `model_prefix=None`. Use it when your local server expects a provider prefix before the model id. |
| `HuggingFaceOptions` | `hf` | None | `trust_remote_code=None`, `processor_kwargs={}`, `model_kwargs={}`, `generation_kwargs={}`, `vision_input_builder=None`, `backend_variant=None`. |
| `VLLMOptions` | `vllm` | None | `trust_remote_code=None`, `processor_kwargs={}`, `llm_kwargs={}`, `sampling_kwargs={}`, `limit_mm_per_prompt={}`. |
| `AzureDocumentIntelligenceOptions` | `azure` | `endpoint`, `api_key` | `model` is optional for Azure OCR in `OCRBackendSpec`. |
| `MistralOptions` | `mistral` | `api_key` | If `OCRBackendSpec.model` is omitted, the backend defaults to `mistral-ocr-latest`. |

The library also exports `DEFAULT_OCR_MAX_TOKENS` from `churro_ocr.providers` for profile and backend integrations that need a shared OCR token budget.

## Page Detection Only

Use `DocumentPageDetector` when you want page crops without OCR.

The default detector treats the whole image as one page:

```python
from churro_ocr.page_detection import DocumentPageDetector, PageDetectionRequest

result = DocumentPageDetector().detect_image_sync(
    PageDetectionRequest(image_path="scan.png")
)

for page in result.pages:
    print(page.page_index, page.image.size)
```

For Azure-backed page detection:

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

For LLM-based page detection:

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

## PDF OCR

If you install the `pdf` extra, `DocumentOCRPipeline` can rasterize PDFs and OCR each page.

```python
from churro_ocr import DocumentOCRPipeline
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

pipeline = DocumentOCRPipeline(
    build_ocr_backend(
        OCRBackendSpec(
            provider="litellm",
            model="vertex_ai/gemini-2.5-flash",
        )
    ),
    max_concurrency=4,
)

result = pipeline.process_pdf_sync("document.pdf", dpi=300, trim_margin=30)

for page in result.pages:
    print(page.page_index, page.text)
```

## Result Objects

The main result types are stable public interfaces:

| Type | Returned by | Important fields |
| --- | --- | --- |
| `DocumentPage` | `OCRClient`, `DocumentPageDetector`, `DocumentOCRPipeline` | `image`, `text`, `provider_name`, `model_name`, `metadata`, `ocr_metadata`, `page_index`, `source_index`, `bbox`, `polygon` |
| `OCRResult` | low-level OCR backend calls and `DocumentOCRResult.as_ocr_results()` | `text`, `provider_name`, `model_name`, `metadata` |
| `PageDetectionResult` | `DocumentPageDetector.detect_*` | `pages`, `source_type`, `metadata` |
| `DocumentOCRResult` | `DocumentOCRPipeline.process_*` | `pages`, `source_type`, `metadata`, `texts()`, `as_ocr_results()` |

Field meanings:
- `metadata` is caller-side or detector-side metadata attached to the page or result object.
- `ocr_metadata` is provider-returned OCR metadata for one page. `DocumentOCRResult.as_ocr_results()` copies it into each `OCRResult.metadata`.
- `page_index` is the page position within the current detection or OCR result.
- `source_index` is the index in the original input source. For single-image flows it is usually `0`. For `detect_pdf(...)` and `process_pdf(...)`, it is the rasterized PDF page index the crop came from.
- `source_type` is `"image"` or `"pdf"` on `PageDetectionResult` and `DocumentOCRResult`.

## Advanced: Custom Prompt Template for a Hugging Face Model

Most users should rely on the built-in model profiles. If you need to override prompt rendering for a custom model, create a custom `OCRModelProfile`.

```python
from churro_ocr import HFChatTemplate
from churro_ocr.providers import (
    HuggingFaceOptions,
    OCRBackendSpec,
    OCRModelProfile,
    build_ocr_backend,
)

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="hf",
        model="your-org/your-vlm",
        profile=OCRModelProfile(
            profile_name="custom",
            template=HFChatTemplate(
                system_message="Transcribe the page exactly.",
                user_prompt=None,
            ),
        ),
        options=HuggingFaceOptions(model_kwargs={"device_map": "auto"}),
    )
)
```

## Prompt and Template Exports

Most users should start with the built-in profiles and templates rather than building prompts from scratch.

Built-in OCR profiles resolved by `resolve_ocr_profile(...)`:
- the generic default profile for unknown models
- `stanford-oval/churro-3B`, which uses `CHURRO_3B_XML_TEMPLATE`
- `kristaller486/dots.ocr-1.5`, which uses `DOTS_OCR_1_5_OCR_TEMPLATE`

Useful public template exports:

| Export | Module | Use case |
| --- | --- | --- |
| `HFChatTemplate` | `churro_ocr.templates` | Build a Hugging Face chat-style multimodal prompt with optional system text, user prompt text, and image inclusion. |
| `DEFAULT_OCR_TEMPLATE` | `churro_ocr.templates` | Generic OCR prompt template used by the default model profile. |
| `CHURRO_3B_XML_TEMPLATE` | `churro_ocr.templates` | Built-in template for `stanford-oval/churro-3B`. |
| `DOTS_OCR_1_5_OCR_TEMPLATE` | `churro_ocr.templates` | Built-in template for `kristaller486/dots.ocr-1.5`. |
| `OCRPromptTemplate` | `churro_ocr.templates` | Base template protocol/type for custom profile integration. |

Useful public prompt exports:

| Export | Module | Use case |
| --- | --- | --- |
| `DEFAULT_OCR_SYSTEM_PROMPT` | `churro_ocr.prompts` | Default system instruction for generic OCR prompting. |
| `DEFAULT_OCR_USER_PROMPT` | `churro_ocr.prompts` | Default user prompt for plain OCR output. |
| `DEFAULT_MARKDOWN_OCR_USER_PROMPT` | `churro_ocr.prompts` | Default user prompt when markdown-style OCR output is preferred. |
| `DEFAULT_OCR_OUTPUT_TAG` | `churro_ocr.prompts` | Shared tag name used by the default OCR postprocessor. |
| `DEFAULT_BOUNDARY_DETECTION_PROMPT` | `churro_ocr.prompts` | Default prompt used by LLM-based page and text-block boundary detection helpers. |
| `strip_ocr_output_tag(...)` | `churro_ocr.prompts` | Remove the default OCR wrapper tag from model output before downstream evaluation or postprocessing. |

If you need prompt overrides without replacing the entire backend, pass a custom `OCRModelProfile` through `OCRBackendSpec(profile=...)` and keep the provider-specific options unchanged.

## CLI

Use the CLI when you want a quick sanity check before writing Python code.

Use `churro-ocr --help` or `python -m churro_ocr --help` to see the top-level commands.

`churro-ocr extract-pages` writes one PNG file per detected page into `--output-dir`.
Files are named sequentially like `page_0000.png`, `page_0001.png`, and so on.
The command also prints each written file path to stdout.

OCR one image:

```bash
churro-ocr transcribe \
  --image scan.png \
  --backend litellm \
  --model vertex_ai/gemini-2.5-flash
```

Extract pages from an image:

```bash
churro-ocr extract-pages \
  --image spread.jpg \
  --output-dir pages/
```

This creates files like:

```text
pages/page_0000.png
pages/page_0001.png
```

Extract pages with Azure page detection:

```bash
churro-ocr extract-pages \
  --image spread.jpg \
  --output-dir pages/ \
  --page-detector azure \
  --endpoint https://<resource>.cognitiveservices.azure.com/ \
  --api-key <azure-doc-intelligence-key>
```

OCR with a local OpenAI-compatible server:

```bash
churro-ocr transcribe \
  --image scan.png \
  --backend openai-compatible \
  --model local-model \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key dummy
```

Extract pages from a PDF:

```bash
churro-ocr extract-pages \
  --pdf document.pdf \
  --output-dir pages/ \
  --dpi 300 \
  --trim-margin 30
```

CLI contract:

`transcribe` backends:

| `--backend` value | Required flags | Notes |
| --- | --- | --- |
| `litellm` | `--model` | Uses LiteLLM credentials and routing. `--base-url`, `--api-key`, and `--api-version` are optional transport overrides. |
| `openai-compatible` | `--model`, `--base-url`, `--api-key` | For local or self-hosted OpenAI-style servers. |
| `azure` | `--endpoint`, `--api-key` | `--model` is optional. |
| `mistral` | `--api-key` | `--model` defaults to `mistral-ocr-latest`. |
| `hf` | `--model` | Local Transformers OCR. |
| `vllm` | `--model` | Local vLLM OCR. |

`extract-pages` page detectors:

| `--page-detector` value | Required flags | Notes |
| --- | --- | --- |
| `none` | none | Default behavior. Treats the whole image or rasterized PDF page as one page crop. |
| `llm` | `--model` | Uses `LLMPageDetector`. `--base-url`, `--api-key`, and `--api-version` are optional transport overrides. |
| `azure` | `--endpoint`, `--api-key` | Uses Azure Document Intelligence layout detection. |

Additional CLI rules:
- `transcribe` requires exactly one `--image`.
- `extract-pages` requires exactly one of `--image` or `--pdf`.
- `--dpi` only affects the `--pdf` path because PDFs are rasterized before page detection.
- `--trim-margin` expands each detected bounding box or polygon crop by that many pixels, clipped to the image bounds. With `--page-detector none`, that usually leaves the full image unchanged.

## Public Modules

The main public modules are:
- `churro_ocr`
- `churro_ocr.document`
- `churro_ocr.ocr`
- `churro_ocr.page_detection`
- `churro_ocr.providers`
- `churro_ocr.templates`
- `churro_ocr.prompts`

## Repo-only Workflows

Benchmarking, contributor commands, and the pre-publish package audit live in the repository workflow guide at `REPO_WORKFLOWS.md`.
Contributor setup lives in `CONTRIBUTING.md`.
These commands require a repo checkout and are not part of the published `churro-ocr` package.

## License

Apache-2.0

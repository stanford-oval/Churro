# Providers And Configuration

Use this page to choose a backend and install the matching runtime.
For custom profiles, prompt templates, and response helpers, continue with [Advanced Customization](advanced-customization.md).

All Churro OCR backends use the same builder entry point:

```python
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="hf",
        model="stanford-oval/churro-3B",
    )
)
```

## Runtime Install Matrix

Install the base package first as shown in [Getting Started](../getting-started.md).
Commands on this page assume the CLI is installed and available as `churro-ocr`.

| Provider or feature | Install command | Good default when |
| --- | --- | --- |
| `litellm` | `churro-ocr install llm` | you want hosted multimodal OCR routed through LiteLLM |
| `openai-compatible` | `churro-ocr install local` | you have a local or self-hosted OpenAI-style server |
| `hf` | `churro-ocr install hf` | you want local Transformers inference in-process |
| `azure` | `churro-ocr install azure` | you want Azure Document Intelligence OCR or page detection |
| `mistral` | `churro-ocr install mistral` | you want Mistral OCR |
| `pdf` | `churro-ocr install pdf` | you want `process_pdf_*` or `extract-pages --pdf` |
| `all` | `churro-ocr install all` | you want every optional runtime in one environment |

`hf` and `all` also install a PyTorch runtime.
Pass `--torch-backend <name>` when you need a specific build, for example `churro-ocr install hf --torch-backend cu126`.

## Recommended Starting Points

| Situation | Good default | Why |
| --- | --- | --- |
| local OCR with no API account | `hf` + `stanford-oval/churro-3B` | matches the quickest credential-free onboarding path |
| hosted OCR | `litellm` + `vertex_ai/gemini-2.5-flash` | easiest hosted path with the standard builder interface |
| layout-heavy local OCR | `hf` + `datalab-to/chandra-ocr-2` | built-in profile matches Chandra's layout-oriented defaults |
| higher-throughput local serving | `openai-compatible` + your own OpenAI-style server | good when you already run a served local backend such as vLLM or llama.cpp |
| managed OCR APIs | `azure` or `mistral` | provider-managed OCR without local model weights |

## Minimal Provider Examples

### Hugging Face

```python
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="hf",
        model="stanford-oval/churro-3B",
    )
)
```

Built-in model-specific profiles are resolved automatically for known models such as `stanford-oval/churro-3B`, `datalab-to/chandra-ocr-2`, `deepseek-ai/DeepSeek-OCR-2`, `kristaller486/dots.ocr-1.5`, `rednote-hilab/dots.mocr`, `opendatalab/MinerU2.5-2509-1.2B`, `PaddlePaddle/PaddleOCR-VL-1.5`, `LiquidAI/LFM2.5-VL-1.6B`, and the supported `olmOCR` checkpoints.

For `opendatalab/MinerU2.5-2509-1.2B`, the built-in `hf` and `openai-compatible` backends both run the model's two-step layout-plus-block pipeline and return markdown with embedded HTML tables when needed. Repo-local benchmark evaluation normalizes that markdown or HTML back to plain text before metrics are computed.

Use `provider="openai-compatible"` when you want to point Churro at a served OpenAI-style endpoint such as vLLM. The generic `litellm` provider is intentionally not supported for this model because MinerU2.5 needs multiple prompt stages instead of a single OCR call.

### LiteLLM

```python
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="litellm",
        model="vertex_ai/gemini-2.5-flash",
    )
)
```

### OpenAI-compatible

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
        ),
    )
)
```

If you want to use vLLM or llama.cpp, serve it separately and point this backend at that server's OpenAI-compatible endpoint.
See the [official vLLM serving docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html) or the [official llama.cpp serving docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

### Azure Document Intelligence

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
        model="mistral-ocr-2512",
        options=MistralOptions(api_key="<mistral-api-key>"),
    )
)
```

## Next Steps

- Use [OCR Workflows](ocr-workflows.md) for Python recipes built on these backends.
- Use [CLI](../cli.md) for shell commands, quick checks, and page extraction.
- Use [Advanced Customization](advanced-customization.md) for custom `OCRModelProfile` work, prompt/template exports, and response helpers.
- Use the [Provider APIs](../api/providers.md), [templates API](../api/templates.md), and [prompts API](../api/prompts.md) when you need exact type definitions and signatures.

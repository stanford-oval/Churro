# Providers And Configuration

All Churro OCR backends use the same builder entry point:

```python
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="litellm",
        model="vertex_ai/gemini-2.5-flash",
    )
)
```

## Which OCR Backend Should You Use?

Install the base package first as shown in [Getting Started](../getting-started.md).
This page is the source of truth for matching providers to runtime targets.

| Provider | Install command | Good default when |
| --- | --- | --- |
| `litellm` | `churro-ocr install llm` | you want hosted multimodal models routed through LiteLLM |
| `openai-compatible` | `churro-ocr install local` | you have a local or self-hosted OpenAI-style server |
| `hf` | `churro-ocr install hf` | you want local Transformers inference in-process |
| `azure` | `churro-ocr install azure` | you want Azure Document Intelligence OCR |
| `mistral` | `churro-ocr install mistral` | you want Mistral OCR |

## Recommended Starting Points

| Situation | Good default | Why |
| --- | --- | --- |
| hosted OCR | `litellm` + `vertex_ai/gemini-2.5-flash` | easiest hosted path with the standard builder interface |
| local OCR | `hf` + `stanford-oval/churro-3B` | first-party local model support in-process |
| layout-heavy local OCR | `hf` + `datalab-to/chandra-ocr-2` | built-in profile matches Chandra's layout-oriented prompt, scaling, and generation defaults |
| higher-throughput local serving | `openai-compatible` + your own OpenAI-style server | good when you already run a served local backend such as vLLM |

## Hosted Providers

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

Override transport or completion settings when you need to:

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

## Local And Self-Hosted Providers

Before using a local or self-hosted provider, install the matching runtime from the table above.

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

If you want to use vLLM, serve it separately and point this backend at that server's OpenAI-compatible endpoint. See the [official vLLM serving docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html).

### Hugging Face

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

Built-in model-specific profiles are resolved automatically for known models such as `stanford-oval/churro-3B`, `datalab-to/chandra-ocr-2`, `deepseek-ai/DeepSeek-OCR-2`, `kristaller486/dots.ocr-1.5`, `rednote-hilab/dots.mocr`, and the supported `olmOCR` checkpoints.

## `OCRBackendSpec` Reference

| Field | Meaning |
| --- | --- |
| `provider` | One of `litellm`, `openai-compatible`, `azure`, `mistral`, or `hf`. |
| `model` | Required for `litellm`, `openai-compatible`, `mistral`, and `hf`. Optional for `azure`. For `mistral`, use one of `mistral-ocr-2505` or `mistral-ocr-2512`. |
| `profile` | `None`, a built-in profile name, or a custom `OCRModelProfile`. |
| `transport` | Shared request transport config for LiteLLM-based providers. |
| `options` | Provider-specific dataclass matching `provider`. |

### Provider Option Dataclasses

| Type | Used by | Required fields | Notes |
| --- | --- | --- | --- |
| `LiteLLMTransportConfig` | `litellm`, `openai-compatible`, `LLMPageDetector` | None at the dataclass level | Use this for transport, credentials, and completion settings. `api_base` is required for `openai-compatible`; `api_key` is optional. |
| `OpenAICompatibleOptions` | `openai-compatible` | None | Use `model_prefix` when your local server expects a provider prefix. |
| `HuggingFaceOptions` | `hf` | None | Carries runtime, processor, generation, and template options. |
| `AzureDocumentIntelligenceOptions` | `azure` | `endpoint`, `api_key` | `model` is optional for Azure OCR in `OCRBackendSpec`. |
| `MistralOptions` | `mistral` | `api_key` | `model` is required and must be `mistral-ocr-2505` or `mistral-ocr-2512`. |

## Advanced Customization

### Custom Profiles And Templates

Most users should rely on the built-in model profiles. If you need to override prompt rendering for a custom Hugging Face model, pass a custom `OCRModelProfile`.

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

### Prompt And Template Exports

Useful public template exports:

| Export | Module | Use case |
| --- | --- | --- |
| `HFChatTemplate` | `churro_ocr.templates` | Build a Hugging Face chat-style multimodal prompt. |
| `DEFAULT_OCR_TEMPLATE` | `churro_ocr.templates` | Generic OCR prompt template used by the default model profile. |
| `CHURRO_3B_XML_TEMPLATE` | `churro_ocr.templates` | Built-in template for `stanford-oval/churro-3B`. |
| `CHANDRA_OCR_2_OCR_TEMPLATE` | `churro_ocr.templates` | Built-in template for `datalab-to/chandra-ocr-2`. |
| `DEEPSEEK_OCR_2_OCR_TEMPLATE` | `churro_ocr.templates` | Built-in template for `deepseek-ai/DeepSeek-OCR-2`. |
| `DOTS_OCR_1_5_OCR_TEMPLATE` | `churro_ocr.templates` | Built-in template for `kristaller486/dots.ocr-1.5`. |
| `DOTS_MOCR_OCR_TEMPLATE` | `churro_ocr.templates` | Built-in template for `rednote-hilab/dots.mocr`. |
| `OCRPromptTemplate` | `churro_ocr.templates` | Base protocol for custom profile integration. |

Useful public prompt exports:

| Export | Module | Use case |
| --- | --- | --- |
| `DEFAULT_OCR_SYSTEM_PROMPT` | `churro_ocr.prompts` | Default system instruction for generic OCR prompting. |
| `DEFAULT_OCR_USER_PROMPT` | `churro_ocr.prompts` | Default user prompt for plain OCR output. |
| `DEFAULT_MARKDOWN_OCR_USER_PROMPT` | `churro_ocr.prompts` | Default user prompt when markdown-style OCR output is preferred. |
| `CHANDRA_OCR_LAYOUT_PROMPT` | `churro_ocr.prompts` | Upstream Chandra OCR 2 layout-block HTML prompt. |
| `DEFAULT_OCR_OUTPUT_TAG` | `churro_ocr.prompts` | Shared tag name used by the default OCR postprocessor. |
| `DEFAULT_BOUNDARY_DETECTION_PROMPT` | `churro_ocr.prompts` | Default prompt used by LLM-based page and text-block boundary detection helpers. |
| `parse_chandra_response(...)` | `churro_ocr.prompts` | Convert Chandra HTML-layout output to plain text and preserve raw HTML metadata. |
| `strip_ocr_output_tag(...)` | `churro_ocr.prompts` | Remove the default OCR wrapper tag from model output. |

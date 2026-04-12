# Advanced Customization

Most users should rely on the built-in model profiles described in [Providers And Configuration](providers.md).
Use this page when you need to override prompt rendering, work directly with template helpers, or parse model-specific OCR output.

## Custom `OCRModelProfile`

If you need to override prompt rendering for a custom Hugging Face model, pass a custom `OCRModelProfile`.

```python
from churro_ocr.providers import (
    HuggingFaceOptions,
    OCRBackendSpec,
    OCRModelProfile,
    build_ocr_backend,
)
from churro_ocr.templates import HFChatTemplate

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

## Template Exports And Helpers

Useful public template exports live in `churro_ocr.templates`.
Use the [templates API](../api/templates.md) for exact signatures.

| Export | Use case |
| --- | --- |
| `HFChatTemplate` | Build a Hugging Face chat-style multimodal prompt. |
| `build_ocr_conversation(...)` | Render a template or template callable into the conversation payload passed to OCR backends. |
| `DEFAULT_OCR_TEMPLATE` | Generic OCR prompt template used by the default model profile. |
| `CHURRO_3B_XML_TEMPLATE` | Built-in template for `stanford-oval/churro-3B`. |
| `CHANDRA_OCR_2_OCR_TEMPLATE` | Built-in template for `datalab-to/chandra-ocr-2`. |
| `DEEPSEEK_OCR_2_OCR_TEMPLATE` | Built-in template for `deepseek-ai/DeepSeek-OCR-2`. |
| `DOTS_OCR_1_5_OCR_TEMPLATE` | Built-in template for `kristaller486/dots.ocr-1.5`. |
| `DOTS_MOCR_OCR_TEMPLATE` | Built-in template for `rednote-hilab/dots.mocr`. |
| `PADDLEOCR_VL_1_5_OCR_TEMPLATE` | Built-in template for `PaddlePaddle/PaddleOCR-VL-1.5`. |
| `OLMOCR_2_7B_1025_OCR_TEMPLATE` | Built-in template for the supported `olmOCR-2-7B-1025` checkpoints. |
| `LFM2_5_VL_1_6B_OCR_TEMPLATE` | Built-in template for `LiquidAI/LFM2.5-VL-1.6B`. |
| `OCRConversation` | Type alias for the rendered multimodal conversation payload. |
| `OCRPromptTemplate` | Base protocol for custom profile integration. |
| `OCRPromptTemplateCallable` | Callable form for dynamic prompt rendering from a `DocumentPage`. |
| `OCRPromptTemplateLike` | Union accepted by helper APIs that can take either a protocol instance or callable template. |

## Prompt Exports And Response Helpers

Useful public prompt exports and response helpers live in `churro_ocr.prompts`.
Use the [prompts API](../api/prompts.md) for exact signatures.

| Export | Use case |
| --- | --- |
| `DEFAULT_OCR_SYSTEM_PROMPT` | Default system instruction for generic OCR prompting. |
| `DEFAULT_OCR_USER_PROMPT` | Default user prompt for plain OCR output. |
| `DEFAULT_MARKDOWN_OCR_USER_PROMPT` | Default user prompt when markdown-style OCR output is preferred. |
| `CHANDRA_OCR_LAYOUT_PROMPT` | Upstream Chandra OCR 2 layout-block HTML prompt. |
| `DEFAULT_OCR_OUTPUT_TAG` | Shared tag name used by the default OCR postprocessor. |
| `DEFAULT_BOUNDARY_DETECTION_PROMPT` | Default prompt used by LLM-based page and text-block boundary detection helpers. |
| `OLMOCR_V4_YAML_PROMPT` | Upstream olmOCR YAML-front-matter prompt used by the built-in olmOCR templates. |
| `parse_chandra_response(...)` | Convert Chandra HTML-layout output to plain text and preserve raw HTML metadata. |
| `parse_olmocr_response(...)` | Convert olmOCR YAML or markdown output into plain text plus parsed metadata. |
| `strip_ocr_output_tag(...)` | Remove the default OCR wrapper tag from model output. |

## Exact Reference

- Use the [Provider APIs](../api/providers.md) for `OCRBackendSpec`, `OCRModelProfile`, and provider option dataclasses.
- Use the [templates API](../api/templates.md) for template protocols, conversations, and built-in templates.
- Use the [prompts API](../api/prompts.md) for prompt constants and response-parsing helpers.

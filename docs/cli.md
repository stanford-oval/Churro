# CLI

Use the CLI when you want a quick sanity check before writing Python code.

Use `churro-ocr --help` or `python -m churro_ocr --help` to inspect the top-level commands.

## Command Summary

| Command | Use it when |
| --- | --- |
| `transcribe` | you want OCR text for one image |
| `extract-pages` | you want page crops from an image or PDF |

## `transcribe` Examples

### OCR One Image

```bash
churro-ocr transcribe \
  --image scan.png \
  --backend litellm \
  --model vertex_ai/gemini-2.5-flash
```

### OCR With A Local OpenAI-compatible Server

```bash
churro-ocr transcribe \
  --image scan.png \
  --backend openai-compatible \
  --model local-model \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key dummy
```

## `extract-pages` Examples

### Extract Pages From An Image

```bash
churro-ocr extract-pages \
  --image spread.jpg \
  --output-dir pages/
```

This writes sequential PNG files such as `page_0000.png`, `page_0001.png`, and so on, and prints each written path to stdout.

### Extract Pages With Azure Page Detection

```bash
churro-ocr extract-pages \
  --image spread.jpg \
  --output-dir pages/ \
  --page-detector azure \
  --endpoint https://<resource>.cognitiveservices.azure.com/ \
  --api-key <azure-doc-intelligence-key>
```

### Extract Pages From A PDF

```bash
churro-ocr extract-pages \
  --pdf document.pdf \
  --output-dir pages/ \
  --dpi 300 \
  --trim-margin 30
```

## Command Contracts

### `transcribe` Backends

| `--backend` value | Required flags | Notes |
| --- | --- | --- |
| `litellm` | `--model` | Uses LiteLLM credentials and routing. `--base-url`, `--api-key`, and `--api-version` are optional transport overrides. |
| `openai-compatible` | `--model`, `--base-url`, `--api-key` | For local or self-hosted OpenAI-style servers. |
| `azure` | `--endpoint`, `--api-key` | `--model` is optional. |
| `mistral` | `--api-key` | `--model` defaults to `mistral-ocr-latest`. |
| `hf` | `--model` | Local Transformers OCR. |
| `vllm` | `--model` | Local vLLM OCR. |

### `extract-pages` Detectors

| `--page-detector` value | Required flags | Notes |
| --- | --- | --- |
| `none` | none | Default behavior. Treats the whole image or rasterized PDF page as one crop. |
| `llm` | `--model` | Uses `LLMPageDetector`. `--base-url`, `--api-key`, and `--api-version` are optional transport overrides. |
| `azure` | `--endpoint`, `--api-key` | Uses Azure Document Intelligence layout detection. |

## Additional Rules

- `transcribe` requires exactly one `--image`.
- `extract-pages` requires exactly one of `--image` or `--pdf`.
- `--dpi` only affects the `--pdf` path because PDFs are rasterized before page detection.
- `--trim-margin` expands each detected crop by the requested number of pixels, clipped to image bounds.

# CLI

Use the CLI when you want to validate a backend, transcribe one image, or extract page crops without writing Python.

Use `churro-ocr --help` or `python -m churro_ocr --help` to inspect the top-level commands.

## Install the CLI

Python 3.12 or newer is required.

```bash
uv tool install churro-ocr
```

If you are adding `churro-ocr` to a project instead, use `uv add churro-ocr` and prefix the commands on this page with `uv run`.

## Install a Runtime

Choose the optional runtime that matches the backend or feature you want to use:

| Target | Command | Use it when |
| --- | --- | --- |
| `hf` | `churro-ocr install hf` | you want local Transformers OCR in-process |
| `llm` | `churro-ocr install llm` | you want hosted multimodal OCR through LiteLLM-backed providers |
| `local` | `churro-ocr install local` | you have a local or self-hosted OpenAI-style server |
| `azure` | `churro-ocr install azure` | you want Azure Document Intelligence OCR or page detection |
| `mistral` | `churro-ocr install mistral` | you want Mistral OCR |
| `pdf` | `churro-ocr install pdf` | you want `extract-pages --pdf` or PDF workflows in Python |
| `all` | `churro-ocr install all` | you want every optional runtime in one environment |

Use `--torch-backend` with `hf` or `all` when you need a specific PyTorch build:

```bash
churro-ocr install hf --torch-backend cu126
```

The examples below use the local `hf` path first.
For backend choice and Python setup, continue with [Providers And Configuration](guides/providers.md).

## First Successful Transcription

```bash
churro-ocr transcribe \
  --image scan.png \
  --backend hf \
  --model stanford-oval/churro-3B
```

## `transcribe` Examples

### Write OCR Text To A File

```bash
churro-ocr transcribe \
  --image scan.png \
  --backend hf \
  --model stanford-oval/churro-3B \
  --output output.txt
```

This writes the OCR text to `output.txt` and prints that written path to stdout.

### OCR With LiteLLM

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
  --base-url http://127.0.0.1:8000/v1
```

For vLLM, serve the model separately with its OpenAI-compatible server and then use this same `openai-compatible` route.
See the [official vLLM serving docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html).

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

Install `pdf` first if you have not already:

```bash
churro-ocr install pdf
```

Then extract rasterized PDF pages as PNG files:

```bash
churro-ocr extract-pages \
  --pdf document.pdf \
  --output-dir pages/ \
  --dpi 300 \
  --trim-margin 30
```

Use [Page Detection](guides/page-detection.md) when you want the Python API for detection only.
Use [OCR Workflows](guides/ocr-workflows.md) when you want page detection and OCR together in Python.

## Command Contracts

### `transcribe` Backends

| `--backend` value | Required flags | Notes |
| --- | --- | --- |
| `litellm` | `--model` | Uses LiteLLM credentials and routing. `--base-url`, `--api-key`, and `--api-version` are optional transport overrides. |
| `openai-compatible` | `--model`, `--base-url` | For local or self-hosted OpenAI-style servers. `--api-key` is optional. |
| `azure` | `--endpoint`, `--api-key` | `--model` is optional. |
| `mistral` | `--api-key`, `--model` | `--model` must be either `mistral-ocr-2505` or `mistral-ocr-2512`. |
| `hf` | `--model` | Local Transformers OCR. |

### `extract-pages` Detectors

| `--page-detector` value | Required flags | Notes |
| --- | --- | --- |
| `none` | none | Default behavior. Treats the whole image or rasterized PDF page as one crop. |
| `llm` | `--model` | Uses `LLMPageDetector`. `--base-url`, `--api-key`, and `--api-version` are optional transport overrides. |
| `azure` | `--endpoint`, `--api-key` | Uses Azure Document Intelligence layout detection. |

## Additional Rules

- `transcribe` requires exactly one `--image`.
- `--output` writes OCR text to a file and prints the written path.
- `extract-pages` requires exactly one of `--image` or `--pdf`.
- `--dpi` only affects the `--pdf` path because PDFs are rasterized before page detection.
- `--trim-margin` expands each detected crop by the requested number of pixels, clipped to image bounds.

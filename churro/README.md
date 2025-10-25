<p align="center">
	<img src="static/churro.png" width="70px" alt="CHURRO Logo" style="display:block;margin:0 auto;" />
	<p align="center">CHURRO: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition</p>
	<p align="center">
		<a href="https://huggingface.co/stanford-oval/churro-3B" target="_blank"><img src="https://img.shields.io/badge/Model-CHURRO%203B-8A4FFF" alt="Model" /></a>
		<a href="https://huggingface.co/datasets/stanford-oval/churro-dataset" target="_blank"><img src="https://img.shields.io/badge/Dataset-CHURRO--DS-0A7BBB" alt="Dataset" /></a>
		<a href="https://arxiv.org/abs/2509.19768" target="_blank"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B" alt="Paper" /></a>
		<a href="https://github.com/stanford-oval/churro/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/stanford-oval/churro?style=social" alt="GitHub Stars" /></a>
	</p>
</p>

<p align="center">
	<sub><i>Handwritten and printed text recognition across 22 centuries and 46 language clusters, including historical and dead languages.</i></sub>
</p>

<p align="center">
		<img src="static/performance_cost.png" alt="Cost vs Performance comparison showing CHURRO's accuracy advantage at significantly lower cost" width="75%" />
		<br/>
		<sub><i>Cost vs. accuracy: CHURRO (3B) achieves higher accuracy than much larger commercial and open-weight VLMs while being substantially cheaper.</i></sub>
</p>

---
## Table of Contents
1. [Overview](#overview)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Configure Providers](#configure-providers)
6. [CLI Workflows](#cli-workflows)
	- [Quick Start: Inference](#quick-start-inference)
	- [Preprocess PDFs and Images](#preprocess-pdfs-and-images)
	- [Benchmark on CHURRO-DS](#benchmark-on-churro-ds)
	- [Local vLLM Container Notes](#local-vllm-container-notes)
7. [Adding a New OCR System](#adding-a-new-ocr-system)
8. [HistoricalDocument XML](#historicaldocument-xml)
9. [Citation](#citation)
10. [License](#license)

---

## Overview
**CHURRO** is a 3B-parameter open-weight vision-language model (VLM) for historical document transcription. It is trained on **CHURRO-DS**, a curated dataset of ~100K pages from 155 historical collections spanning 22 centuries and 46 language clusters.

On the CHURRO-DS test set, CHURRO delivers **15.5× lower cost than Gemini 2.5 Pro while exceeding its accuracy**.

## Prerequisites
### System Packages (Ubuntu example)
```bash
sudo apt-get update && sudo apt-get install -y \
	libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev libfreetype6-dev \
	liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk libharfbuzz-dev \
	libfribidi-dev libxcb1-dev
```

### Docker (recommended for local models)
- Install Docker: https://docs.docker.com/engine/install/
- GPU users: add the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- CPU-only machines can still run local models, but expect significantly slower throughput.

## Environment Setup
We use [Pixi](https://pixi.sh/) to manage Python environments and dependencies. If you are familiar with [Conda](https://docs.conda.io/), you can think of Pixi as a much faster alternative. The following commands set up a Pixi shell with all required packages. Make sure the environment is active before running any Python code.
```bash
git clone https://github.com/stanford-oval/churro.git
cd churro
curl -fsSL https://pixi.sh/install.sh | bash
pixi shell  # create and enter the managed environment

## Configure Providers
```
Copy the example environment file:
```bash
cp .example.env .env
```
Populate only the variables you need in `.env`.
All environment variables live in `.env` and are autoloaded via `python-dotenv`. Use the table below as a quick reference to decide which credentials you must supply.

| Workflow | Required providers | Key variables |
|----------|-------------------|---------------|
| Azure Document Intelligence OCR (`--system azure`) or if using `docs-to-images` command without `--no-trim` | Azure Document Intelligence | `AZURE_DI_ENDPOINT`, `AZURE_DOC_KEY` |
| LLM-based OCR against Vertex AI deployments | Google Vertex AI | `VERTEX_AI_LOCATION` |
| LLM-based OCR against Azure/OpenAI deployments | Azure OpenAI or OpenAI | `AZURE_API_BASE`, `AZURE_OPENAI_API_KEY` (or `OPENAI_API_KEY`), `AZURE_API_VERSION` |
| Mistral OCR (`--system mistral_ocr`) | Mistral | `MISTRAL_API_KEY` |
| Local vLLM models (`--system finetuned` or `llm` with engines backed by `vllm/`) | Docker + Hugging Face | `LOCAL_VLLM_PORT`, `HF_TOKEN` (only if using private models) |

When a workflow does not need a provider, leave the corresponding variables blank. See `.example.env` for full documentation of each field.
For Vertex AI usage, additionally ensure that the Google Cloud SDK is installed and authenticated: https://cloud.google.com/sdk/docs/install

Note that for all API LLM calls, the outputs are cached in `.litellm_cache/`. So subsequent runs with the same inputs will be much faster and free.

## CLI Workflows
The unified Typer CLI lives under `churro/cli`. All examples below assume you are inside a `pixi shell` or prefix commands with `pixi run`.

### Quick Start: Inference
Single image (local CHURRO model hosted via vLLM):
```bash
pixi run python -m churro.cli infer \
	--system finetuned \
	--engine churro \
	--image tests/churro_dataset_sample_1.jpeg
```

`finetuned` system returns HistoricalDocument XML by default; add `--strip-xml` to output plain text instead. See [HistoricalDocument XML](#historicaldocument-xml) for schema details and parsing tips.

Batch directory with filtered suffixes and output files:
```bash
pixi run python -m churro.cli infer \
	--system finetuned \
	--engine churro \
	--image-dir path/to/images \
	--suffix png --suffix jpeg \
	--recursive \
	--output-dir workdir/texts/ \
	--skip-existing \
	--max-concurrency 8
```

Use `pixi run python -m churro.cli infer --help` to see every option, including how to use other LLMs via `--system llm --engine <engine>` arguments.

### Preprocess PDFs and Images
If you have raw PDF scans or image directories, first use the `docs-to-images` command to convert them into page-aligned PNGs ready for OCR.
`docs-to-images` normalizes PDF scans and image directories into page-aligned PNGs. The default engine `gemini-2.5-pro-low` calls a Vertex AI model to detect double-page spreads, then calls Azure Document Intelligence to detect page boundaries and trim margins.

Single PDF:
```bash
pixi run python -m churro.cli docs-to-images \
	--input-file path/to/file.pdf \
	--output-dir workdir/images/
```

Mixed directory with custom suffix filters and dry run:
```bash
pixi run python -m churro.cli docs-to-images \
	--input-dir path/to/scans \
	--suffix pdf --suffix tif --suffix png \
	--recursive \
	--output-dir workdir/images/ \
	--dry-run
```

Here is how this pipeline works:
- An LLM estimates whether a rasterized page contains a two-page spread. Provide `--engine <MODEL_MAP key>` to swap to a different splitter if you do not have Vertex AI access.
- Margin trimming is enabled by default via Azure Document Intelligence. Use `--no-trim` to disable this stage.
- `--batch-pages`, `--queue-maxsize`, `--raster-workers`, `--page-workers`, and `--llm-concurrency-limit` balance CPU-bound rasterization and LLM throughput.
- Pages are written as `<source_base>_page_XXXX.png`, even when spreads split into multiple images.

### Benchmark on CHURRO-DS
Run end-to-end evaluation against the CHURRO dataset. The command automatically initializes any required local vLLM server before processing.
```bash
pixi run python -m churro.cli benchmark \
	--system finetuned \
	--engine churro \
	--dataset-split test \
	--input-size 0 \
	--max-concurrency 32
```

Important options:
- `--system {azure,mistral_ocr,llm,finetuned}` determines which OCR backend to use.
- `--engine <key>` is required for `llm` and `finetuned` systems; see `churro/utils/llm/models.py` for the full `MODEL_MAP` of logical keys (GPT-4/5, Claude, Gemini, Qwen 2.5, MiniCPM, CHURRO, and more).
- `--tensor-parallel-size` / `--data-parallel-size` tune vLLM scaling for local engines.
- `--resize <pixels>` optionally resizes large images before inference.

Outputs land under `workdir/results/<split>/<system>_<engine>/` (the engine suffix is omitted for `azure` and `mistral_ocr`).


### LLM Improver
The Churro CLI supports optional post-processing with the `LLMImprover`, enabled via `--use-improver`. Pair it with `--improver-engine`. Improver can help fix OCR errors, and improve the formatting of complex documents' Markdown.

### Backup Engines
You can supply a backup engine for LLM-based OCR systems using `--backup-engine`, and for LLM improvers using `--improver-backup-engine`.
Both backup options allow the pipeline to retry with a secondary model if the first call fails. For example, when a provider's content filter incorrectly flags historical material or when a transient outage interrupts inference.


### Local vLLM Container Notes
When you run `infer` or `benchmark` with an `llm` or `finetuned` system whose engine has an `hf_repo` entry, the CLI will:
- Read `LOCAL_VLLM_PORT` and `HF_TOKEN` from your environment (`churro/utils/docker/vllm.py`).
- Pull the corresponding Hugging Face repository on first launch. Expect multi-gigabyte downloads.
- Start a Docker container exposing an OpenAI-compatible API at `http://localhost:<LOCAL_VLLM_PORT>/v1`.
- Stop the container automatically when the command exits or crashes.

Make sure the chosen port is free and that Docker is running. GPU acceleration is optional but dramatically improves throughput.

## Adding a New OCR System
Pull requests for new VLMs and OCR backends are welcome.

If adding a new LLM, simply add it to `utils/llm/models.py` (`MODEL_MAP`). Include an `hf_repo` for vLLM-served models.

For entirely new OCR systems, follow all steps:
1. Register the system in `churro/systems/ocr_factory.py` so the CLI can instantiate it.
2. Implement `process_image` and `get_system_name` in a subclass of `BaseOCR`.
3. Use `--system <system_name>` with the CLI or import the factory in your own scripts.

## HistoricalDocument XML
`HistoricalDocument` is the XML schema we use in the CHURRO dataset and model for rich transcriptions. It is specifically designed to capture complex layouts, scribal edits, and missing text, which are all common in historical documents, while preserving reading order.

Each response contains a root `<HistoricalDocument>` element with optional `<Metadata>` details (languages, scripts, writing direction, notes) followed by one or more `<Page>` blocks. A page combines optional `<Header>` and `<Footer>` regions with a required `<Body>` that nests structural tags such as `<Paragraph>`, `<MarginalNote>`, `<Figure>`, and `<List>`. Inline markup like `<Addition>`, `<Deletion>`, `<Gap/>`, and `<InterlinearNote>` captures scribal edits or missing text while preserving reading order.

```xml
<HistoricalDocument xmlns="http://example.com/historicaldocument">
	<Metadata>
		<Language>lat</Language>
		<Script>Latn</Script>
	</Metadata>
	<Page>
		<Header/>
		<Body>
			<Paragraph>
				<Line>In nomine domini amen.</Line>
				<Line><Gap reason="illegible"/> nos notarii subscripsimus.</Line>
			</Paragraph>
		</Body>
	</Page>
</HistoricalDocument>
```

The complete definition lives in `churro/evaluation/historical_doc.xsd`. The inference CLI's `--strip-xml` flag and the evaluation helpers call `churro.evaluation.xml_utils.extract_actual_text_from_xml()` to remove all XML tags and flatten the content into plain text when you do not need the markup.
---

## Citation
If you use CHURRO or CHURRO-DS, please cite:

```bibtex
@inproceedings{semnani2025churro,
	title        = {{CHURRO}: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition},
	author       = {Semnani, Sina J. and Zhang, Han and He, Xinyan and Tekg{"u}rler, Merve and Lam, Monica S.},
	booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)},
	year         = {2025}
}
```

---

## License
- Model Weights: Qwen research license (see HF model card)
- Dataset: Due to licensing restrictions on the original datasets used in Churro, use is permitted for research purposes only.
- Code: Apache 2.0

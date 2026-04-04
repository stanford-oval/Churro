# Churro Repo Workflows

This guide is for repository checkouts of Churro. Run commands from the repo root.

## Benchmarking on CHURRO-DS

The benchmark runner lives in this repo at `tooling.benchmarking.benchmark`. Run it from the checkout root.

The smallest useful benchmark command looks like this:

```bash
pixi run python -m tooling.benchmarking.benchmark \
  --backend litellm \
  --dataset-split test \
  --model vertex_ai/gemini-2.5-pro
```

Useful flags:
- `--dataset-split dev|test`: choose the CHURRO-DS split
- `--input-size N`: benchmark only the first `N` selected pages
- `--offset N`: skip the first `N` selected pages
- `--language` and `--document-type`: filter the benchmark subset before slicing
- `--output-dir PATH`: override the default results directory
- `--max-concurrency N`: cap the number of in-flight OCR requests
- `--vllm-gpu-memory-utilization` and `--vllm-cpu-offload-gb`: pass through selected vLLM runtime knobs

By default, results are written under `workdir/results/<split>/`.

Example commands for different models:

| Model | Model ID | Backend | Full command |
| --- | --- | --- | --- |
| Gemini 2.5 Pro | `vertex_ai/gemini-2.5-pro` | `litellm` | `pixi run python -m tooling.benchmarking.benchmark --backend litellm --dataset-split test --model vertex_ai/gemini-2.5-pro --output-dir workdir/results/test/litellm_vertex_ai_gemini-2.5-pro` |
| Qwen 3.5-0.8B | `Qwen/Qwen3.5-0.8B` | `vllm` | `pixi run python -m tooling.benchmarking.benchmark --backend vllm --dataset-split test --model Qwen/Qwen3.5-0.8B --output-dir workdir/results/test/vllm_Qwen_Qwen3.5-0.8B` |

## Evaluation Outputs

Each benchmark run writes one result directory. The important files are:
- `outputs.json`: one row per evaluated page with the raw predicted text, gold text, and page-level metrics.
- `all_metrics.json`: aggregate metrics grouped across the full run, by main language, by document type, and by the language/type combination.

`outputs.json` stores page-level metric values as raw fractions. `all_metrics.json` converts aggregate values to percentages and rounds them to one decimal place.

The current aggregate metrics include:
- `normalized_levenshtein_similarity`: character-level similarity after OCR cleanup and text normalization.
- `bleu`: BLEU score for non-empty predictions.
- `repetition`: whether the normalized prediction shows long repeated suffix patterns.
- `is_empty`: whether the normalized prediction is empty after cleanup.
- `llm_cost ($)`, `azure_cost ($)`, and `elapsed_time (s)`: run-level summary fields added after aggregation.

Normalization in evaluation strips the default OCR wrapper tag, extracts text from supported XML-like OCR output, normalizes whitespace and punctuation, and applies additional Arabic normalization for Arabic and Persian examples.

## Filtering and Slicing

Subset filters are applied before offset and limit:
- `--language` filters on `main_language`
- `--document-type` filters on `document_type`
- `--offset` skips rows after filtering
- `--input-size` limits rows after filtering and offset

That means `--language Arabic --offset 100 --input-size 50` selects rows 101 to 150 from the Arabic-only subset, not from the full split.

## Development in This Repo

The public contributor entrypoints are Pixi tasks:

```bash
pixi run format
pixi run lint
pixi run typecheck
pixi run test
pixi run package-check
```

Coverage from a repo checkout:

```bash
pixi run coverage
```

## Package Check

`pixi run package-check` runs the repo-local publish gate defined in `scripts/package_check.py`. It validates the built artifacts, not the editable checkout.

The current package check does all of the following:
- removes stale `build/`, `dist/`, and generated egg-info directories
- builds a fresh wheel and sdist
- runs `twine check` on both artifacts
- verifies the wheel metadata, including project URLs, extras, and the `churro-ocr` console entry point
- verifies that repo-only content such as `tests/`, `tooling/`, `scripts/`, `REPO_WORKFLOWS.md`, and `PYPI_AUDIT.md` do not ship inside the artifacts
- smoke-installs the base wheel and base sdist in clean virtual environments and checks `import churro_ocr`, `import churro_ocr.providers`, and `python -m churro_ocr --help`
- smoke-installs the lightweight `local` and `pdf` extras from the built wheel
- audits direct dependency licenses for incompatible or unknown licenses

If `package-check` fails, treat that as a release blocker until the artifact or documentation contract is fixed.

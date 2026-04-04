# Benchmarking

This page is for reproducing CHURRO-DS benchmark runs from a repo checkout.

Run these commands from the repo root after `pixi install`.

For the current committed benchmark snapshot, see the [Benchmark Leaderboard](leaderboard.md). Contributor setup, test commands, and package checks live in [Contributing](contributing.md).

## Smallest Useful Run

The benchmark runner lives in this repo at `tooling.benchmarking.benchmark`.

```bash
pixi run python -m tooling.benchmarking.benchmark \
  --backend litellm \
  --dataset-split test \
  --model vertex_ai/gemini-2.5-pro
```

By default, results are written under `workdir/results/<split>/`.

## Common Flags

- `--dataset-split dev|test`: choose the CHURRO-DS split
- `--input-size N`: benchmark only the first `N` selected pages
- `--offset N`: skip the first `N` selected pages
- `--language` and `--document-type`: filter the benchmark subset before slicing
- `--output-dir PATH`: override the default results directory
- `--max-concurrency N`: cap the number of in-flight OCR requests
- `--vllm-gpu-memory-utilization` and `--vllm-cpu-offload-gb`: pass through selected vLLM runtime knobs

## Output Files

Each benchmark run writes one result directory. The important files are:

- `outputs.json`: one row per evaluated page with the raw predicted text, gold text, and page-level metrics
- `all_metrics.json`: aggregate metrics grouped across the full run, by main language, by document type, and by the language/type combination

`outputs.json` stores page-level metric values as raw fractions. `all_metrics.json` converts aggregate values to percentages and rounds them to one decimal place. The evaluation pipeline strips the default OCR wrapper tag, flattens supported XML-like OCR output, normalizes whitespace and punctuation, and applies additional Arabic normalization for Arabic and Persian examples.

## Filtering And Slicing

Subset filters are applied before offset and limit:

- `--language` filters on `main_language`
- `--document-type` filters on `document_type`
- `--offset` skips rows after filtering
- `--input-size` limits rows after filtering and offset

That means `--language Arabic --offset 100 --input-size 50` selects rows 101 to 150 from the Arabic-only subset, not from the full split.

## Example Commands

| Model | Model ID | Backend | Full command |
| --- | --- | --- | --- |
| Gemini 2.5 Pro | `vertex_ai/gemini-2.5-pro` | `litellm` | `pixi run python -m tooling.benchmarking.benchmark --backend litellm --dataset-split test --model vertex_ai/gemini-2.5-pro --output-dir workdir/results/test/litellm_vertex_ai_gemini-2.5-pro` |
| Qwen 3.5-0.8B | `Qwen/Qwen3.5-0.8B` | `vllm` | `pixi run python -m tooling.benchmarking.benchmark --backend vllm --dataset-split test --model Qwen/Qwen3.5-0.8B --output-dir workdir/results/test/vllm_Qwen_Qwen3.5-0.8B` |

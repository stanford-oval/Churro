# Benchmarking

For the official leaderboard results, see the [Benchmark Leaderboard](leaderboard.md).

This page describes how to benchmark your own model on [CHURRO-DS](https://huggingface.co/datasets/stanford-oval/churro-dataset). Please open a pull request if you would like to add your model to the official leaderboard.


## Smallest Useful Run

The benchmark runner lives in this repo at `tooling.benchmarking.benchmark`.

```bash
pixi run python -m tooling.benchmarking.benchmark \
  --backend litellm \
  --dataset-split test \
  --model vertex_ai/gemini-2.5-pro
```

By default, results are written under `workdir/results/<split>/`.
The evaluation pipeline strips the default OCR wrapper tag, flattens supported XML-like OCR output, normalizes whitespace and punctuation, and applies additional Arabic normalization for languages with Arabic script (Arabic and Persian).

## Common Flags

- `--dataset-split dev|test`: choose the CHURRO-DS split
- `--input-size N`: benchmark only the first `N` selected pages
- `--offset N`: skip the first `N` selected pages
- `--language` and `--document-type`: filter the benchmark subset before slicing
- `--output-dir PATH`: override the default results directory
- `--max-concurrency N`: cap the number of in-flight OCR requests

## Output Files

Each benchmark run writes one result directory. The directory contains two JSON files:

- `outputs.json`: one row per evaluated page with the raw predicted text, gold text, and page-level metrics
- `all_metrics.json`: aggregate metrics grouped across the full run, by main language, by document type, and by the language/type combination


## Filtering And Slicing

You can run benchmarks on subsets of the data by combining `--language`, `--document-type`, `--offset`, and `--input-size`. The filters are applied in the following order:

- `--language` filters on `main_language`
- `--document-type` filters on `document_type`
- `--offset` skips rows after filtering
- `--input-size` limits rows after filtering and offset

That means for example `--language Arabic --offset 100 --input-size 50` selects rows 101 to 150 from the Arabic-only subset, not from the full split.

## Example Commands

If you want to benchmark a model using vLLM, run a vLLM server separately and point `--backend openai-compatible` at its OpenAI-compatible endpoint. See the [official vLLM serving docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html).

| Model | Model ID | Backend | Full command |
| --- | --- | --- | --- |
| Gemini 2.5 Pro | `vertex_ai/gemini-2.5-pro` | `litellm` | `pixi run python -m tooling.benchmarking.benchmark --backend litellm --dataset-split test --model vertex_ai/gemini-2.5-pro --output-dir workdir/results/test/litellm_vertex_ai_gemini-2.5-pro` |
| Qwen 3.5-0.8B | `Qwen/Qwen3.5-0.8B` | `openai-compatible` | `pixi run python -m tooling.benchmarking.benchmark --backend openai-compatible --dataset-split test --model Qwen/Qwen3.5-0.8B --base-url http://127.0.0.1:8000/v1 --output-dir workdir/results/test/openai-compatible_Qwen_Qwen3.5-0.8B` |

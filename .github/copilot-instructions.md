## AI Assistant Project Instructions

Focus on concrete, project-specific practices. Keep answers concise, typed, and actionable.

### Core Architecture
* OCR systems in `ocr/systems/` subclass `BaseOCR` (async `process()` + `get_system_name()`). They are created only via `OCRFactory` using `--system` (`azure|mistral_ocr|llm|hybrid|finetuned`). Do not instantiate concrete classes directly in new entrypoints.
* End-to-end run: `ocr/end_to_end.py` -> parse args -> optionally start local vLLM containers -> run OCR system -> evaluate via `evaluation/metrics.py` -> write `outputs.json`, `all_metrics.json`, scatter plot to `results/<split>/<system>_<engine>`.
* Model selection: `utils/llm/models.py` `MODEL_MAP` (logical key -> ordered provider candidates + `static_params` + optional `hf_repo`). Centralize model/provider changes here.
* PDF ingestion: `utils/pdf/runner.py` async pipeline (process pool rasterization + async LLM-driven page split + trimming). Public API `run_pdf_pipeline` re-exported in `utils/pdf/__init__.py`.
* Metrics: `evaluation/metrics.py` handles language/type aggregation, cost injection (`llm_cost ($)`, `azure_cost ($)`), elapsed time, and token scatter plot. Preserve existing JSON keys.
* Fine-tuning: `azure_ml/train_vlm.py` streams HF dataset, builds chat messages, applies masking in custom collator. Reuse patterns before inventing new training frameworks.

### Conventions & Style
* Python 3.12. Always type hint (prefer builtins: `list[str]`, `dict[str, Any]`).
* Google-style docstrings for new public functions/classes.
* Use `utils.log_utils.logger` (loguru+rich) instead of `print`.
* Run `pixi run ruff check --fix .` after non-trivial edits (line length ≤100). Import grouping defined in `ruff.toml`.

### Patterns to Follow / Avoid
* Colocate narrow helpers with feature module; move to `utils/` only after ≥2 consumers.
* Extend `_extract_system_config` in `ocr_factory` instead of re-parsing identical args.
* Never hardcode `results/` or replicate directory overwriting logic.

If design is ambiguous (e.g., new page-splitting heuristic or cost dimension), first output a short design note before generating code.
Always run python code via `pixi run python <script>` to ensure correct environment.

End of instructions.
# Changelog

All notable changes to `churro-ocr` will be documented in this file.

## 0.3.0

### Added

- Added UV-managed runtime installers through `churro-ocr install` for `llm`, `local`, `hf`, `azure`, `mistral`, `pdf`, and `all`.
- Added built-in OCR profiles and provider integrations for Chandra OCR 2, Dots OCR 1.5, dots.mocr, DeepSeek OCR 2, FireRed OCR, GLM-OCR, Infinity-Parser 7B, Liquid LFM2.5-VL 1.6B, MinerU2.5, Nanonets OCR2, olmOCR 2 7B, PaddleOCR-VL 1.5, and Qianfan OCR.
- Added richer prompt, template, and response-processing helpers for OCR backends that emit markdown or HTML.
- Added benchmark leaderboard improvements, including expandable per-language score views and refreshed benchmark coverage across newly supported models.

### Changed

- Updated the `hf` extra to `transformers>=5,<6` and moved local PyTorch installation behind the runtime installer workflow.
- Reworked the docs around a CLI-first onboarding path with expanded provider guidance, advanced customization docs, and a more detailed PyPI/README presentation.
- Tightened typing and internal provider boundaries across OCR, page detection, evaluation, and helper modules.

### Fixed

- Improved retry handling with a total timeout budget, better transient provider-error handling, and more graceful handling for timed-out OCR pages.
- Hardened LiteLLM request cleanup, unmapped-model handling, and default OCR parsing for providers that return empty or markdown-heavy responses.
- Fixed evaluation and benchmarking stability issues around multiprocessing pools, cached LiteLLM clients, and OCR metadata retention.

### Breaking Changes

- Removed the `vllm` extra and the in-process `churro_ocr.providers.vllm` backend. Serve vLLM separately and use the `openai-compatible` backend instead.
- Stopped re-exporting template classes from the top-level `churro_ocr` package. Import `HFChatTemplate`, `OCRPromptTemplate`, and related helpers from `churro_ocr.templates`.

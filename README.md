<div align="center">

# <img src="static/churro.png" alt="CHURRO logo" width="40" /> Churro

</div>

<p align="center">
  <a href="https://huggingface.co/stanford-oval/churro-3B">🤗 Model</a> •
  <a href="https://huggingface.co/datasets/stanford-oval/churro-dataset">🗂️ Dataset</a> •
  <a href="https://arxiv.org/abs/2509.19768">📄 Paper</a>
  <br/><br/>
  <a href="https://stanford-oval.github.io/Churro/">📚 Docs</a> •
  <a href="https://stanford-oval.github.io/Churro/leaderboard.html">🏆 Leaderboard</a> •
  <a href="https://github.com/stanford-oval/churro/stargazers">
    <img src="https://img.shields.io/github/stars/stanford-oval/churro?style=social" alt="GitHub Stars badge" />
  </a>
</p>

Churro is the fastest way to turn hard-to-read historical scans into reliable text. It gives researchers, libraries, archives, and product teams a unified OCR toolkit for handwritten and printed sources, combining high accuracy, low operating cost, and a clean Python API and CLI workflow.

## Supported OCR Models and Backends

Churro includes built-in profiles, templates, and post-processing for many OCR models and integrations, including:

- Hosted vision-language models, including Gemini, GPT, Claude, and more, through LiteLLM integration
- OpenAI-compatible servers, including vLLM, Ollama, TGI, and more
- Azure Document Intelligence
- Mistral OCR
- `Chandra OCR`
- `DeepSeek OCR`
- `Dots OCR`
- `MinerU`
- `Infinity Parser`
- `PaddleOCR VL`
- `LFM VL`

## Churro Model and Dataset

We also provide first-party support for a purpose-trained model and dataset for historical OCR:

- Churro 3B VLM exceeds the accuracy of Gemini 2.5 Pro at 15.5x lower cost.
- Churro-DS dataset contains ~100K pages from 155 historical collections spanning 22 centuries and 46 language clusters.

<p align="center">
  <img src="static/performance_cost.png" alt="Cost vs Performance comparison showing Churro's accuracy advantage at significantly lower cost" width="75%" />
  <br/>
  <sub><i>Cost vs. accuracy: Churro (3B) achieves higher accuracy than much larger commercial and open-weight VLMs while being substantially cheaper.</i></sub>
</p>

## Quick Start

Python 3.12+ and `uv` are required.

```bash
uv tool install churro-ocr
churro-ocr install hf
churro-ocr transcribe --image scan.png --backend hf --model stanford-oval/churro-3B
```

For more in-depth information, see the [Getting Started](https://stanford-oval.github.io/Churro/getting-started.html) guide.

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

## License

- Code: Apache 2.0
- Model weights: Qwen research license
- Dataset: research use only because of the underlying source licenses

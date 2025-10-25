"""Helpers for conditionally starting local vLLM servers for OCR pipelines.

Central responsibilities:
    * Inspect the `MODEL_MAP` for vLLM-backed engine entries.
    * Resolve the appropriate Hugging Face repository for a hosted variant.
    * Conditionally launch a local vLLM Docker container when an OCR run
        requires a locally served model (provider model name starts with ``vllm/``).

Public entrypoints:
    - has_at_least_one_vllm(engine_key)
    - get_hf_repo_for_hosted(engine_key)
    - maybe_start_vllm_server_for_engine(args)

Functions accept lightweight, explicit parameters (or an `argparse.Namespace`-like
object) to avoid importing OCR-specific code here.
"""

from __future__ import annotations

from churro.config import ChurroSettings, get_settings as get_churro_settings
from churro.utils.llm.models import MODEL_MAP
from churro.utils.llm.types import ModelInfo
from churro.utils.log_utils import logger

from . import DockerContainer, start_vllm_server


__all__ = [
    "get_hf_repo_for_hosted",
    "has_at_least_one_vllm",
    "maybe_start_vllm_server_for_engine",
]


def get_hf_repo_for_hosted(engine_key: str) -> str | None:
    """Return the first Hugging Face repo associated with a vLLM provider model.

    Iterates over model info entries for ``engine_key`` inside ``MODEL_MAP`` and
    returns the first non-empty ``hf_repo`` whose ``provider_model`` starts with
    ``"vllm/"``.
    """
    infos: list[ModelInfo] | None = MODEL_MAP.get(engine_key)
    if not infos:
        return None
    for info in infos:
        provider_model = info["provider_model"]
        if provider_model.startswith("vllm/"):
            repo = info.get("hf_repo")  # optional
            if repo:
                return repo
    return None


def has_at_least_one_vllm(engine_key: str) -> bool:
    """Return True if any provider model variant for the engine uses vLLM."""
    infos: list[ModelInfo] | None = MODEL_MAP.get(engine_key)
    if not infos:
        return False
    return any(info["provider_model"].startswith("vllm/") for info in infos)


def _select_model_repo(engine: str) -> str:
    """Return the backing HF repo for a vLLM-hosted engine or raise."""
    model_repo = get_hf_repo_for_hosted(engine)
    if model_repo is None:
        raise ValueError(
            f"Engine '{engine}' expected to have a vLLM-backed hf_repo but none was found."
        )
    return model_repo


def maybe_start_vllm_server_for_engine(
    *,
    engine: str | None,
    system: str,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    log_prefix: str | None = None,
    install_flash_attn: bool = False,
    settings: ChurroSettings | None = None,
) -> DockerContainer | None:
    """Conditionally start a vLLM server for the provided engine.

    Side effects (container launch) only occur when all of the following are true:
      * system in {"llm", "finetuned"}
      * engine provided and corresponds to at least one vLLM provider variant

    Args:
        engine: Logical engine key (MODEL_MAP) to potentially serve.
        system: OCR system type (e.g., 'llm', 'finetuned').
        tensor_parallel_size: Tensor parallel degree for vLLM container.
        data_parallel_size: Data parallel degree for vLLM container.
        log_prefix: Optional log prefix string.
        install_flash_attn: Whether to attempt to install flash attention
            support inside the container.
        settings: Optional pre-loaded configuration snapshot. Supplying this allows
            callers (including tests) to inject custom ports or tokens without
            mutating global environment variables.

    Returns:
        DockerContainer | None: Container object if launched else None.
    """
    if not (system in {"llm", "finetuned"} and engine):
        return None

    if not has_at_least_one_vllm(engine):
        return None

    config = settings or get_churro_settings()
    host_port = config.local.vllm_port
    if host_port is None:
        raise ValueError("LOCAL_VLLM_PORT must be set in environment to start local vLLM server.")
    if not (0 < host_port < 65536):
        raise ValueError(f"LOCAL_VLLM_PORT value out of range: {host_port}")

    model_repo = _select_model_repo(engine)

    logger.info(
        f"{log_prefix or ''} Starting local vLLM server for engine '{engine}' with HF repo '{model_repo}'."
    )

    max_model_len = None
    engine_infos: list[ModelInfo] | None = MODEL_MAP.get(engine)
    if engine_infos:
        first = engine_infos[0]
        max_model_len = first["max_completion_tokens"]

    container = start_vllm_server(
        model=model_repo,
        served_model_name=engine,
        host_port=host_port,
        max_model_len=max_model_len,
        force_replace=True,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        install_flash_attn=install_flash_attn,
    )
    return container

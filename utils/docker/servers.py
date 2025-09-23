"""Higher-level helpers for launching model serving runtimes."""

from __future__ import annotations

import contextlib
from typing import Optional, Pattern, Sequence

from utils.log_utils import logger

from .container import DockerContainer
from .logging_utils import format_prefix, log_multiline
from .operations import start_container, wait_for_readiness


__all__ = [
    "start_vllm_server",
]

# Internal helpers (keep local to this module)


def _pip_install(
    container: DockerContainer,
    packages: Sequence[str],
    *,
    log_prefix: Optional[str],
) -> int:
    """Install pip packages inside the container."""
    if packages:
        # Centralize install logging so callers don't need to emit their own lines.
        joined = " ".join(packages)
        logger.info(f"{format_prefix(log_prefix)}Installing {joined}...")
    cmd: list[str] = [
        "pip",
        "install",
        "--no-input",
        "--root-user-action=ignore",
        *packages,
    ]
    code, out = container.exec(cmd)
    if out:
        log_multiline(out, log_prefix)
        logger.info("")
    if code != 0:
        joined = " ".join(packages) or "<no packages>"
        logger.warning(
            f"{format_prefix(log_prefix)}Warning: failed to install {joined} (continuing)"
        )
    return code


def _get_package_version(
    container: DockerContainer,
    package: str,
    *,
    log_prefix: Optional[str],
) -> str:
    """Return package version inside container or 'not installed'."""
    py_code: str = (
        "import importlib,importlib.metadata as m,sys;"
        f"print(m.version('{package}') if importlib.util.find_spec('{package}') else 'not installed')"
    )
    version_line: Optional[str] = None
    for interp in (["python3", "-c", py_code], ["python", "-c", py_code]):
        code, out = container.exec(interp)
        if code == 0 and out:
            version_line = out.strip().splitlines()[-1]
            break
    if version_line is None:
        code, out = container.exec(["pip", "show", package])
        if code == 0 and out:
            for ln in out.splitlines():
                if ln.lower().startswith("version:"):
                    version_line = ln.split(":", 1)[1].strip()
                    break
    version_line = version_line or "not installed"
    log_multiline(version_line, log_prefix)
    return version_line


def start_vllm_server(
    *,
    model: str,
    served_model_name: Optional[str] = None,
    host_port: int = 9000,
    container_port: int = 8000,
    gpus: str = "all",
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    huggingface_cache: str = "~/.cache/huggingface",
    image: str = "vllm/vllm-openai:v0.10.2",
    model_args: Optional[Sequence[str]] = None,
    ready_timeout: float = 180.0,
    ready_pattern: (str | Pattern[str]) = r"Application startup complete|Uvicorn running on http",
    log_prefix: Optional[str] = "[vLLM] ",
    force_replace: bool = False,
    install_flash_attn: bool = False,
) -> DockerContainer:
    """Start a vLLM OpenAI-compatible API container and wait for readiness."""
    volumes: dict[str, str] = {huggingface_cache: "/root/.cache/huggingface"}
    ports: dict[int, int] = {host_port: container_port}

    logger.info(f"{format_prefix(log_prefix)}Data parallel size set to {data_parallel_size}")
    logger.info(f"{format_prefix(log_prefix)}Tensor parallel size set to {tensor_parallel_size}")

    cmd: list[str] = [
        "--model",
        model,
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--data-parallel-size",
        str(data_parallel_size),
        "--trust_remote_code",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if model_args:
        cmd += list(model_args)

    base_image_part: str = image.rsplit("/", 1)[-1]
    base_no_tag: str = base_image_part.split(":", 1)[0]
    sanitized_name: str = base_no_tag.replace("/", "-").replace(":", "-") or "vllm"

    logger.info(f"Starting vLLM container '{sanitized_name}'. This may take several minutes...")

    container = start_container(
        image=image,
        name=sanitized_name,
        gpus=gpus,
        volumes=volumes,
        ports=ports,
        ipc="host",
        cmd=cmd,
        force_replace=force_replace,
        pull=True,
    )

    if "mistral" in model.lower():
        # Mistral models need this package
        _pip_install(
            container,
            ["mistral-common==1.8.4"],
            log_prefix=log_prefix,
        )
        logger.info(f"{format_prefix(log_prefix)}mistral_common version:")
        _get_package_version(container, "mistral_common", log_prefix=log_prefix)

    _pip_install(container, ["open_clip_torch"], log_prefix=log_prefix)
    logger.info(f"{format_prefix(log_prefix)}open_clip_torch version:")
    _get_package_version(container, "open_clip_torch", log_prefix=log_prefix)

    if install_flash_attn:
        # install flash-attn
        _pip_install(container, ["flash-attn"], log_prefix=log_prefix)
        logger.info(f"{format_prefix(log_prefix)}flash-attn version:")
        _get_package_version(container, "flash-attn", log_prefix=log_prefix)

    try:
        wait_for_readiness(
            container,
            ready_pattern=ready_pattern,
            ready_timeout=ready_timeout,
            check_interval=1.0,
            log_prefix=log_prefix,
        )
    except Exception:
        with contextlib.suppress(Exception):
            container.stop()
        raise
    return container

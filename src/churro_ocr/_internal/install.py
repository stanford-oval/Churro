"""UV-backed runtime installation helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Final

from churro_ocr.errors import ConfigurationError

PROJECT_DISTRIBUTION_NAME: Final[str] = "churro-ocr"
VLLM_RUNTIME_DIR_ENV: Final[str] = "CHURRO_OCR_VLLM_RUNTIME_DIR"
DEFAULT_VLLM_RUNTIME_DIR: Final[Path] = Path.home() / ".cache" / "churro-ocr" / "runtimes" / "vllm"
INSTALL_TARGETS: Final[tuple[str, ...]] = (
    "llm",
    "local",
    "hf",
    "vllm",
    "azure",
    "mistral",
    "pdf",
    "all",
)
_CURRENT_ENV_TARGET_EXTRAS: Final[dict[str, tuple[str, ...]]] = {
    "llm": ("llm",),
    "local": ("local",),
    "hf": ("hf",),
    "vllm": ("local",),
    "azure": ("azure",),
    "mistral": ("mistral",),
    "pdf": ("pdf",),
    "all": ("llm", "local", "hf", "azure", "mistral", "pdf"),
}
_PYTORCH_TARGETS: Final[frozenset[str]] = frozenset({"hf", "all"})
_PYTORCH_PACKAGES: Final[tuple[str, ...]] = ("torch", "torchvision")


@dataclass(frozen=True, slots=True)
class RuntimeInstallResult:
    """Summary of a completed runtime installation."""

    target: str
    executed_commands: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...] = ()
    vllm_runtime_dir: Path | None = None


def install_command_hint(target: str) -> str:
    """Return a short user-facing install hint for a runtime target."""
    return f"Run `churro-ocr install {target}`."


def recommended_vllm_runtime_hint() -> str:
    """Return the short user-facing hint for the dedicated vLLM runtime."""
    return "Run `churro-ocr install vllm`."


def resolve_vllm_runtime_dir(runtime_dir: Path | None = None) -> Path:
    """Resolve the dedicated vLLM runtime directory."""
    if runtime_dir is not None:
        return runtime_dir.expanduser().resolve()
    override = os.environ.get(VLLM_RUNTIME_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_VLLM_RUNTIME_DIR


def install_runtime_dependencies(
    *,
    target: str,
    torch_backend: str = "auto",
    vllm_runtime_dir: Path | None = None,
) -> RuntimeInstallResult:
    """Install one of the supported optional runtime targets with UV."""
    normalized_target = target.strip().lower()
    if normalized_target not in INSTALL_TARGETS:
        supported = ", ".join(INSTALL_TARGETS)
        raise ConfigurationError(f"Unknown install target '{target}'. Choose one of: {supported}.")

    uv_executable = _require_uv_executable()
    executed_commands: list[tuple[str, ...]] = []
    notes: list[str] = []

    current_env_extras = _CURRENT_ENV_TARGET_EXTRAS.get(normalized_target, ())
    requirements = _requirements_for_extra(current_env_extras)
    if requirements:
        executed_commands.append(
            _run_command(
                [
                    uv_executable,
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--upgrade",
                    *requirements,
                ]
            )
        )

    if normalized_target in _PYTORCH_TARGETS:
        executed_commands.append(
            _run_command(
                [
                    uv_executable,
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--upgrade",
                    f"--torch-backend={torch_backend}",
                    *_PYTORCH_PACKAGES,
                ]
            )
        )

    resolved_runtime_dir: Path | None = None
    if normalized_target in {"vllm", "all"}:
        resolved_runtime_dir = resolve_vllm_runtime_dir(vllm_runtime_dir)
        runtime_python = _ensure_runtime_python(resolved_runtime_dir)
        executed_commands.append(
            _run_command(
                [
                    uv_executable,
                    "pip",
                    "install",
                    "--python",
                    str(runtime_python),
                    "--upgrade",
                    f"--torch-backend={torch_backend}",
                    _requirement_for_extra("vllm", package_name="vllm"),
                ]
            )
        )
        notes.append(
            "The dedicated vLLM runtime is ready. Start it with `churro-ocr serve-vllm --model <model-id>`."
        )
        notes.append(
            "Use `--backend openai-compatible --base-url http://127.0.0.1:8000/v1` "
            "to connect to the served runtime."
        )

    return RuntimeInstallResult(
        target=normalized_target,
        executed_commands=tuple(executed_commands),
        notes=tuple(notes),
        vllm_runtime_dir=resolved_runtime_dir,
    )


def serve_vllm_runtime(
    *,
    model: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    runtime_dir: Path | None = None,
    extra_args: tuple[str, ...] = (),
) -> int:
    """Run `python -m vllm serve` from the dedicated Churro vLLM runtime."""
    resolved_runtime_dir = resolve_vllm_runtime_dir(runtime_dir)
    runtime_python = _venv_python(resolved_runtime_dir)
    if not runtime_python.exists():
        raise ConfigurationError(
            f"No dedicated vLLM runtime exists at {resolved_runtime_dir}. {recommended_vllm_runtime_hint()}"
        )

    command = [
        str(runtime_python),
        "-m",
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        *extra_args,
    ]
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:  # pragma: no cover - depends on host process state
        raise ConfigurationError(
            f"Failed to launch the dedicated vLLM runtime at {resolved_runtime_dir}: {exc}"
        ) from exc
    return int(completed.returncode)


def _require_uv_executable() -> str:
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise ConfigurationError(
            "`churro-ocr install` requires `uv` on PATH. Install uv and rerun the command."
        )
    return uv_executable


def _distribution_requirements() -> list[str]:
    try:
        distribution = metadata.distribution(PROJECT_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - depends on install mode
        raise ConfigurationError(
            "The Churro installer must run from an installed `churro-ocr` environment."
        ) from exc
    return list(distribution.requires or [])


def _requirements_for_extra(extras: tuple[str, ...]) -> list[str]:
    requirements = _distribution_requirements()
    selected: list[str] = []
    for extra in extras:
        for requirement_text in requirements:
            requirement, _, marker = requirement_text.partition(";")
            if f'extra == "{extra}"' not in marker:
                continue
            normalized_requirement = requirement.strip()
            if normalized_requirement and normalized_requirement not in selected:
                selected.append(normalized_requirement)
    return selected


def _requirement_for_extra(extra: str, *, package_name: str) -> str:
    normalized_package_name = package_name.replace("_", "-").lower()
    for requirement in _requirements_for_extra((extra,)):
        dependency_name = _requirement_name(requirement)
        if dependency_name.replace("_", "-").lower() == normalized_package_name:
            return requirement
    raise ConfigurationError(
        f"Could not resolve the `{package_name}` dependency from the `{extra}` extra metadata."
    )


def _ensure_runtime_python(runtime_dir: Path) -> Path:
    runtime_python = _venv_python(runtime_dir)
    if runtime_python.exists():
        return runtime_python

    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(runtime_dir)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(
            f"Failed to create the dedicated vLLM runtime at {runtime_dir}: {exc}"
        ) from exc

    if not runtime_python.exists():
        raise ConfigurationError(
            f"Created the dedicated vLLM runtime at {runtime_dir}, but no Python interpreter was found."
        )
    return runtime_python


def _run_command(command: list[str]) -> tuple[str, ...]:
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        rendered_command = " ".join(command)
        raise ConfigurationError(f"Command failed: {rendered_command}") from exc
    return tuple(command)


def _requirement_name(requirement: str) -> str:
    name_chars: list[str] = []
    for character in requirement:
        if character.isalnum() or character in {"-", "_", "."}:
            name_chars.append(character)
            continue
        if character == "[":
            break
        break
    return "".join(name_chars)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"

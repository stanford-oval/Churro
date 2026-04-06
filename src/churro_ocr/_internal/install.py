"""UV-backed runtime installation helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Final

from churro_ocr.errors import ConfigurationError

PROJECT_DISTRIBUTION_NAME: Final[str] = "churro-ocr"
INSTALL_TARGETS: Final[tuple[str, ...]] = (
    "llm",
    "local",
    "hf",
    "azure",
    "mistral",
    "pdf",
    "all",
)
_CURRENT_ENV_TARGET_EXTRAS: Final[dict[str, tuple[str, ...]]] = {
    "llm": ("llm",),
    "local": ("local",),
    "hf": ("hf",),
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


def install_command_hint(target: str) -> str:
    """Return a short user-facing install hint for a runtime target."""
    return f"Run `churro-ocr install {target}`."


def install_runtime_dependencies(
    *,
    target: str,
    torch_backend: str = "auto",
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

    return RuntimeInstallResult(
        target=normalized_target,
        executed_commands=tuple(executed_commands),
        notes=tuple(notes),
    )


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


def _run_command(command: list[str]) -> tuple[str, ...]:
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        rendered_command = " ".join(command)
        raise ConfigurationError(f"Command failed: {rendered_command}") from exc
    return tuple(command)

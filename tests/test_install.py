from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import churro_ocr._internal.install as install_module
from churro_ocr.errors import ConfigurationError


class _FakeDistribution:
    def __init__(self, *, requires: list[str] | None) -> None:
        self.requires = requires


def test_install_runtime_dependencies_installs_hf_and_torch_with_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(
        install_module.metadata,
        "distribution",
        lambda _: _FakeDistribution(
            requires=[
                'qwen-vl-utils; extra == "hf"',
                'transformers>=4.57.0,<5; extra == "hf"',
            ]
        ),
    )
    monkeypatch.setattr(install_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        install_module.subprocess,
        "run",
        lambda command, check=True: commands.append(list(command)) or SimpleNamespace(returncode=0),
    )

    result = install_module.install_runtime_dependencies(
        target="hf",
        torch_backend="cu126",
    )

    assert result.target == "hf"
    assert result.vllm_runtime_dir is None
    assert commands == [
        [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            "qwen-vl-utils",
            "transformers>=4.57.0,<5",
        ],
        [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            "--torch-backend=cu126",
            "torch",
            "torchvision",
        ],
    ]


def test_install_runtime_dependencies_installs_local_client_and_dedicated_vllm_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    runtime_dir = tmp_path / "vllm-runtime"
    runtime_python = runtime_dir / "bin" / "python"

    monkeypatch.setattr(
        install_module.metadata,
        "distribution",
        lambda _: _FakeDistribution(
            requires=[
                'litellm[caching]==1.82.3; extra == "local"',
                'vllm>=0.18,<1; extra == "vllm"',
            ]
        ),
    )
    monkeypatch.setattr(install_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    def _fake_ensure_runtime_python(path: Path) -> Path:
        assert path == runtime_dir.resolve()
        return runtime_python

    monkeypatch.setattr(
        install_module,
        "_ensure_runtime_python",
        _fake_ensure_runtime_python,
    )
    monkeypatch.setattr(
        install_module.subprocess,
        "run",
        lambda command, check=True: commands.append(list(command)) or SimpleNamespace(returncode=0),
    )

    result = install_module.install_runtime_dependencies(
        target="vllm",
        vllm_runtime_dir=runtime_dir,
    )

    assert result.target == "vllm"
    assert result.vllm_runtime_dir == runtime_dir.resolve()
    assert commands == [
        [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            "litellm[caching]==1.82.3",
        ],
        [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            str(runtime_python),
            "--upgrade",
            "--torch-backend=auto",
            "vllm>=0.18,<1",
        ],
    ]
    assert result.notes


def test_install_runtime_dependencies_requires_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_module.shutil, "which", lambda _: None)

    with pytest.raises(ConfigurationError, match="requires `uv` on PATH"):
        install_module.install_runtime_dependencies(target="hf")


def test_resolve_vllm_runtime_dir_prefers_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(install_module.VLLM_RUNTIME_DIR_ENV, str(tmp_path / "from-env"))

    resolved = install_module.resolve_vllm_runtime_dir(tmp_path / "from-arg")

    assert resolved == (tmp_path / "from-arg").resolve()


def test_resolve_vllm_runtime_dir_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(install_module.VLLM_RUNTIME_DIR_ENV, str(tmp_path / "from-env"))

    resolved = install_module.resolve_vllm_runtime_dir()

    assert resolved == (tmp_path / "from-env").resolve()


def test_serve_vllm_runtime_runs_module_from_dedicated_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_python = runtime_dir / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("")

    commands: list[list[str]] = []
    monkeypatch.setattr(
        install_module.subprocess,
        "run",
        lambda command, check=False: commands.append(list(command)) or SimpleNamespace(returncode=0),
    )

    exit_code = install_module.serve_vllm_runtime(
        model="stanford-oval/churro-3B",
        host="0.0.0.0",
        port=9000,
        runtime_dir=runtime_dir,
        extra_args=("--tensor-parallel-size", "2"),
    )

    assert exit_code == 0
    assert commands == [
        [
            str(runtime_python),
            "-m",
            "vllm",
            "serve",
            "stanford-oval/churro-3B",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--tensor-parallel-size",
            "2",
        ]
    ]


def test_serve_vllm_runtime_requires_installed_runtime(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="install vllm"):
        install_module.serve_vllm_runtime(
            model="stanford-oval/churro-3B",
            runtime_dir=tmp_path / "missing-runtime",
        )

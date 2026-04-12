from __future__ import annotations

import sys
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
                'transformers>=5,<6; extra == "hf"',
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
    assert commands == [
        [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            "qwen-vl-utils",
            "transformers>=5,<6",
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


def test_install_runtime_dependencies_installs_local_client_with_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(
        install_module.metadata,
        "distribution",
        lambda _: _FakeDistribution(requires=['litellm[caching]==1.82.3; extra == "local"']),
    )
    monkeypatch.setattr(install_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        install_module.subprocess,
        "run",
        lambda command, check=True: commands.append(list(command)) or SimpleNamespace(returncode=0),
    )

    result = install_module.install_runtime_dependencies(target="local")

    assert result.target == "local"
    assert commands == [
        [
            "/usr/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            "litellm[caching]==1.82.3",
        ]
    ]
    assert result.notes == ()


def test_install_runtime_dependencies_requires_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_module.shutil, "which", lambda _: None)

    with pytest.raises(ConfigurationError, match="requires `uv` on PATH"):
        install_module.install_runtime_dependencies(target="hf")

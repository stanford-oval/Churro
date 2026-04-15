from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image
from typer.testing import CliRunner

if TYPE_CHECKING:
    from tests._types import ImageColor, ImportFailurePatcher, WriteImageFile

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_REPO_SRC_PATH = Path(__file__).resolve().parents[1] / "src"
_REPO_SRC_PATH_STR = str(_REPO_SRC_PATH)

if _REPO_SRC_PATH_STR in sys.path:
    sys.path.remove(_REPO_SRC_PATH_STR)
sys.path.insert(0, _REPO_SRC_PATH_STR)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def minimal_pdf_path() -> Path:
    return _TESTS_DIR / "assets" / "minimal-document.pdf"


@pytest.fixture
def test_artifact_dir_path() -> Path:
    return _REPO_ROOT / "workdir" / "test-artifacts"


@pytest.fixture
def write_image_file(tmp_path: Path) -> WriteImageFile:
    def _write_image_file(
        *,
        size: tuple[int, int] = (10, 10),
        filename: str = "sample.png",
        mode: str = "RGB",
        color: ImageColor = "white",
    ) -> Path:
        image_path = tmp_path / filename
        Image.new(mode, size, color=color).save(image_path)
        return image_path

    return _write_image_file


@pytest.fixture
def patch_import_failure(monkeypatch: pytest.MonkeyPatch) -> ImportFailurePatcher:
    real_import = builtins.__import__

    def _patch_import_failure(
        *,
        failing_name: str,
        exception_type: type[ImportError] = ImportError,
    ) -> None:
        def _fake_import(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == failing_name:
                message = f"missing {failing_name}"
                raise exception_type(message)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

    return _patch_import_failure

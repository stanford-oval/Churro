from __future__ import annotations

from argparse import Namespace
import builtins
from pathlib import Path

import pytest

from churro import args


@pytest.fixture
def fake_module_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Force churro.args to treat tmp_path as its module root."""

    def fake_resolve(self: Path) -> Path:  # pragma: no cover - exercised via create_output_prefix
        return tmp_path / "module.py"

    monkeypatch.setattr(args.Path, "resolve", fake_resolve, raising=False)
    return tmp_path


def test_validate_args_requires_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(args, "MODEL_MAP", {"valid-engine": object()})
    with pytest.raises(AssertionError, match="LLM engine must be specified"):
        args._validate_args(Namespace(system="llm", engine=None))


def test_validate_args_rejects_unknown_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(args, "MODEL_MAP", {"valid-engine": object()})
    with pytest.raises(AssertionError, match="Invalid engine"):
        args._validate_args(Namespace(system="llm", engine="unknown"))


def test_create_output_prefix_creates_directory(
    fake_module_root: Path,
) -> None:
    target = Namespace(system="azure", engine=None, dataset_split="dev")
    output = Path(args.create_output_prefix(target))
    expected = fake_module_root / "workdir" / "results" / "dev" / "azure"
    assert output == expected
    assert output.exists()


def test_create_output_prefix_aborts_non_interactive(
    fake_module_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = fake_module_root / "workdir" / "results" / "dev" / "azure"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "existing.txt").write_text("keep-me")

    class DummyStdIn:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(args.sys, "stdin", DummyStdIn())

    target = Namespace(system="azure", engine=None, dataset_split="dev")
    with pytest.raises(SystemExit):
        args.create_output_prefix(target)

    assert (output_dir / "existing.txt").exists()


def test_create_output_prefix_allows_interactive_overwrite(
    fake_module_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = fake_module_root / "workdir" / "results" / "dev" / "azure"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_file = output_dir / "existing.txt"
    existing_file.write_text("old")

    class TtyStdIn:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(args.sys, "stdin", TtyStdIn())
    monkeypatch.setattr(builtins, "input", lambda _: "y")

    target = Namespace(system="azure", engine=None, dataset_split="dev")
    result = Path(args.create_output_prefix(target))

    assert result == output_dir
    assert output_dir.exists()
    assert not existing_file.exists()

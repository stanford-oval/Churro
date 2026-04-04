"""Prefer the repo-local `src/` package tree over unrelated site-packages installs."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_SRC = Path(__file__).resolve().parent / "src"
_REPO_SRC_STR = str(_REPO_SRC)

if _REPO_SRC_STR in sys.path:
    sys.path.remove(_REPO_SRC_STR)
sys.path.insert(0, _REPO_SRC_STR)

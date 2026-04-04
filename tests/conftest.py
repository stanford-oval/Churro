from __future__ import annotations

import sys
from pathlib import Path

_REPO_SRC_PATH = Path(__file__).resolve().parents[1] / "src"
_REPO_SRC_PATH_STR = str(_REPO_SRC_PATH)

if _REPO_SRC_PATH_STR in sys.path:
    sys.path.remove(_REPO_SRC_PATH_STR)
sys.path.insert(0, _REPO_SRC_PATH_STR)

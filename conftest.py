# Test configuration utilities.
# Ensures the repository root is on sys.path so that 'utils', 'ocr', etc. can be imported
# when running pytest without installing the package.
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

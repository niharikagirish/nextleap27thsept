"""Make the project root importable so ``import app`` works from any test cwd.

pytest prepends the *test file's* directory to sys.path, not the project root, so
without this the suite fails to import ``app`` unless it is run from a particular
working directory. The project root is the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

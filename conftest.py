"""Make the repository root importable however pytest is invoked.

`evals/tier1` imports `tests.fakes` -- the eval gate runs the same in-memory
store and embedder the unit tests use, so the fixtures it loads behave exactly
as they do everywhere else.

Without this, `pytest` over the whole project works (rootdir lands on the path
via `tests/`) but `pytest evals/tier1` alone fails with ModuleNotFoundError --
which is precisely what someone debugging a failing gate would run first.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

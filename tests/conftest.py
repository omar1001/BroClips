"""Test setup: every test gets its own empty BroClips data folder — never the user's real one.

The data folder is set BEFORE any broclips import (broclips.util opens a log file in it when imported), and the
"data folder pointer" file next to the app is redirected too, so a pointer the user saved (e.g. E:\\BroClips) can
never be read or changed by a test."""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SESSION = Path(tempfile.mkdtemp(prefix="broclips-tests-"))
os.environ["BROCLIPS_DATA"] = str(_SESSION / "data")
atexit.register(shutil.rmtree, _SESSION, ignore_errors=True)

from broclips import config  # noqa: E402

config.POINTER = _SESSION / "data_location.txt"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """A fresh data folder per test, and a fresh (stopped) work line."""
    from broclips import jobs
    d = tmp_path / "data"
    monkeypatch.setenv("BROCLIPS_DATA", str(d))
    monkeypatch.setattr(config, "POINTER", tmp_path / "data_location.txt")
    old = jobs.LINE
    jobs.LINE = jobs.WorkLine()
    yield d
    jobs.LINE.stop(2)
    jobs.LINE = old

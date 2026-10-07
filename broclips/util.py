"""Small shared helpers: logging, running programs without a console window, safe JSON files, time formatting."""
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import config

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # never flash a console window (SPEC §2.4)

log = logging.getLogger("broclips")
if not log.handlers:
    log.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    try:
        _fh = logging.FileHandler(config.logs_dir() / f"{datetime.now():%Y-%m-%d}.log", encoding="utf-8")
        _fh.setFormatter(_fmt)
        log.addHandler(_fh)
    except OSError:
        pass
    if sys.stdout is not None:                            # pythonw has no console
        _sh = logging.StreamHandler(sys.stdout)
        _sh.setFormatter(_fmt)
        log.addHandler(_sh)


def run(cmd, cwd=None, check=True, timeout=None):
    """Run a program without a window; raise with the end of its error output when it fails."""
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", creationflags=NO_WINDOW, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed ({p.returncode}): {p.stderr[-1500:]}")
    return p


def run_bytes(cmd, timeout=None):
    """Run a program without a window and return its raw stdout (frames, audio)."""
    return subprocess.run([str(c) for c in cmd], capture_output=True, creationflags=NO_WINDOW,
                          timeout=timeout).stdout


def part_path(out):
    """Temporary name for an output; renamed to the real name only after success (no half-written files)."""
    out = Path(out)
    return out.with_name(out.stem + ".part" + out.suffix)


def finish(tmp, final, tries=60):
    """Rename tmp -> final, retrying: a virus scanner can lock a fresh file for a moment on Windows."""
    for i in range(tries):
        try:
            Path(tmp).replace(final)
            return final
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(1)


def save_json(obj, path):
    """Write JSON via a temp file + rename, retrying (another thread may be reading it)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=1)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    for i in range(40):
        try:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, p)
            return
        except PermissionError:
            if i == 39:
                raise
            time.sleep(0.25)


def load_json(path, default=None):
    """Read JSON; a missing, half-written or locked file gives the default."""
    p = Path(path)
    for _ in range(3):
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (PermissionError, ValueError):
            time.sleep(0.1)
    return default


def mmss(sec):
    sec = max(0, float(sec or 0))
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def programs_running(names):
    """True if any of these programs (exe names, case-insensitive) is running — used to pause heavy work."""
    if not names:
        return False
    try:
        import psutil
    except ImportError:
        return False
    want = {n.lower() for n in names}
    for p in psutil.process_iter(["name"]):
        if (p.info["name"] or "").lower() in want:
            return True
    return False


def lower_priority():
    """Below-normal priority so games and the desktop stay smooth (children inherit it)."""
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if config.IS_WINDOWS else 10)
    except Exception:
        pass

"""Paths and settings. Everything the user can change lives in <data>/settings.json (defaults below); API keys live
only in <data>/secrets.json (SPEC §2.3, §5). The data folder itself is remembered in a tiny pointer file next to the
app, so it can be moved to another disk."""
import copy
import json
import os
import sys
from pathlib import Path

APP_NAME = "BroClips"
VERSION = "0.1.0"
APP_DIR = Path(__file__).resolve().parent              # .../BroClips/broclips
REPO_DIR = APP_DIR.parent
WEB_DIR = APP_DIR / "web"
PORT = 8770
POINTER = REPO_DIR / "data_location.txt"               # one line: the data folder (absent = the default)


def default_data_dir():
    return Path(os.environ.get("BROCLIPS_DATA") or (Path.home() / APP_NAME))


def data_dir():
    """The data folder (settings, keys, projects, outputs, downloads). Created on first use."""
    d = default_data_dir()
    try:
        p = POINTER.read_text(encoding="utf-8").strip()
        if p:
            d = Path(p)
    except OSError:
        pass
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_data_dir(path):
    """Move the data folder pointer (the files are NOT moved: the user picks where new things go)."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    POINTER.write_text(str(p), encoding="utf-8")
    return p


DEFAULTS = {
    "thinking": {                   # the LLM role (SPEC §4.1)
        "provider": "ollama",       # ollama | gemini | openai | anthropic | fake
        "ollama": {"url": "http://127.0.0.1:11434", "model": "gemma3:12b", "num_ctx": 16384,
                   "manage": {"enabled": False, "exe": "", "models_dir": "", "port": 11435}},
        "gemini": {"model": ""},    # "" = the provider's default
        "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_name": "openai"},
        "anthropic": {"model": ""},
    },
    "listening": {                  # the speech-to-text role (SPEC §4.2)
        "provider": "local",        # local | openai | gemini | fake
        "local": {"model": "auto", "device": "auto", "models_dir": ""},     # auto = large-v3 with CUDA, else small
        "openai": {"base_url": "https://api.openai.com/v1", "model": "whisper-1", "key_name": "openai"},
        "gemini": {"model": ""},
    },
    "allow_images_to_cloud": False,  # SPEC §4.3
    "language": "auto",             # default content language for new projects
    "output_dir": "",               # "" = <data>/projects
    "encoder": "auto",              # auto | nvenc | x264
    "pause_while_running": [],      # exe names, e.g. ["League of Legends.exe"]
    "blur_boxes": [],               # [[x, y, w, h], ...] in 1920x1080 coordinates, blurred in every output
    "cutout_model": "",             # path to the BiRefNet ONNX file ("" = <data>/models/BiRefNet-general-epoch_244.onnx)
    "max_shorts": 8,
}

PROJECT_DEFAULTS = {
    "name": "My videos",
    "prompt": "",                   # what the videos are, audience, platforms, language, tone
    "video_type": "gameplay",       # gameplay | camera | screen | podcast | other
    "platforms": ["youtube", "shorts", "tiktok"],
    "language": "auto",
    "marathon": False,
    "jump_cuts": True,
    "silence_s": 1.2,
    "layout": "auto",               # auto (from video_type) | follow | zoom | fit
    "max_shorts": 8,
}


def _merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def settings():
    """The current settings, defaults filled in for anything missing."""
    return _merge(DEFAULTS, _read(data_dir() / "settings.json", {}))


def save_settings(new):
    from .util import save_json
    save_json(_merge(DEFAULTS, new), data_dir() / "settings.json")


def secrets():
    return _read(data_dir() / "secrets.json", {}) or {}


def secret(name):
    return str(secrets().get(name) or "").strip()


def set_secret(name, value):
    """Store an API key (or remove it with an empty value). Only ever written to secrets.json."""
    from .util import save_json
    s = secrets()
    if value:
        s[name] = str(value).strip()
    else:
        s.pop(name, None)
    save_json(s, data_dir() / "secrets.json")


def masked_secrets():
    """{"gemini": "…abcd", ...} — what the UI may show."""
    return {k: ("…" + v[-4:] if len(v) > 4 else "set") for k, v in secrets().items() if v}


def projects_dir():
    out = settings().get("output_dir")
    d = Path(out) if out else data_dir() / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def assets_dir():
    d = data_dir() / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir():
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir():
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir():
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


IS_WINDOWS = sys.platform.startswith("win")

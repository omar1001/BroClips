"""Projects and videos on disk (SPEC §3 data layout):

  <projects>/<pid>/project.json                    the project's settings (config.PROJECT_DEFAULTS + the user's)
  <projects>/<pid>/videos/<vid>/video.json         {"src": "D:\\rec\\lesson1.mp4", "name", "dur", "added", "status"}
  <projects>/<pid>/videos/<vid>/...                everything BroClips makes for that video

The user's video files are only REFERENCED (video.json "src" = the absolute path): never copied, moved, changed or
deleted. Deleting a project / removing a video moves only BroClips' own folder to the Recycle Bin.
IDs are short slugs + 4 random characters ("python-lessons-k3x9"), so they are safe in paths and URLs."""
import importlib
import random
import re
import shutil
import string
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from . import config
from .util import load_json, log, save_json

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mts", ".m2ts", ".mpg",
              ".mpeg", ".3gp")
VIDEO_TYPES = ("gameplay", "camera", "screen", "podcast", "other")
PLATFORMS = ("youtube", "shorts", "tiktok", "reels")
LAYOUTS = ("auto", "follow", "zoom", "fit")
ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
_lock = threading.RLock()             # read-modify-write of project/video files (the server has many threads)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def valid_id(x):
    return isinstance(x, str) and bool(ID_RE.fullmatch(x))


def _check(*ids):
    for x in ids:
        if not valid_id(x):
            raise ValueError(f"Unknown id: {x!r}")


def new_id(name, fallback):
    """'Python Lessons #1' -> 'python-lessons-1-k3x9' (non-Latin names fall back to e.g. 'project-k3x9')."""
    ascii_name = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:24].strip("-") or fallback
    return f"{slug}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=4))}"


def project_dir(pid):
    _check(pid)
    return config.projects_dir() / pid


def video_dir(pid, vid):
    _check(pid, vid)
    return config.projects_dir() / pid / "videos" / vid


def _merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _clean_name(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()[:80]


def _bool(x):
    return x.strip().lower() in ("1", "true", "yes", "on") if isinstance(x, str) else bool(x)


def clean_fields(fields):
    """Only known project fields, with sane types and ranges; anything else is ignored."""
    f, out = fields or {}, {}
    if "name" in f and _clean_name(f["name"]):
        out["name"] = _clean_name(f["name"])
    if "prompt" in f:
        out["prompt"] = str(f["prompt"] or "")[:5000]
    if f.get("video_type") in VIDEO_TYPES:
        out["video_type"] = f["video_type"]
    if isinstance(f.get("platforms"), list):
        out["platforms"] = [p for p in PLATFORMS if p in f["platforms"]]
    if "language" in f:
        lang = str(f["language"] or "").strip()
        if re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?|auto", lang):
            out["language"] = lang
    for b in ("marathon", "jump_cuts"):
        if b in f:
            out[b] = _bool(f[b])
    if "silence_s" in f:
        try:
            out["silence_s"] = round(max(0.3, min(10.0, float(f["silence_s"]))), 2)
        except (TypeError, ValueError):
            pass
    if f.get("layout") in LAYOUTS:
        out["layout"] = f["layout"]
    if "max_shorts" in f:
        try:
            out["max_shorts"] = max(1, min(30, int(float(f["max_shorts"]))))
        except (TypeError, ValueError):
            pass
    return out


# ---------- projects ----------
def get(pid):
    """One project (defaults filled in) + its id and video count, or None."""
    if not valid_id(pid):
        return None
    d = config.projects_dir() / pid
    raw = load_json(d / "project.json")
    if not isinstance(raw, dict):
        return None
    p = _merge(config.PROJECT_DEFAULTS, raw)
    p["id"] = pid
    p["videos"] = sum(1 for v in (d / "videos").glob("*/video.json")) if (d / "videos").is_dir() else 0
    return p


def list_projects():
    """Every project, newest first."""
    root = config.projects_dir()
    out = [get(d.name) for d in root.iterdir() if d.is_dir() and valid_id(d.name) and (d / "project.json").is_file()]
    out = [p for p in out if p]
    out.sort(key=lambda p: (p.get("created") or "", p["id"]), reverse=True)
    return out


def create(name):
    s = config.settings()
    name = _clean_name(name) or config.PROJECT_DEFAULTS["name"]
    with _lock:
        root = config.projects_dir()
        pid = new_id(name, "project")
        while (root / pid).exists():
            pid = new_id(name, "project")
        p = _merge(config.PROJECT_DEFAULTS, {"name": name, "language": s.get("language") or "auto",
                                            "max_shorts": s.get("max_shorts") or 8})
        p.update(id=pid, created=_now())
        save_json(p, root / pid / "project.json")
    log.info("project %s created", pid)
    return get(pid)


def update(pid, fields):
    """Change a project's settings (merged with config.PROJECT_DEFAULTS). Returns the project, or None if unknown."""
    with _lock:
        cur = get(pid)
        if cur is None:
            return None
        raw = load_json(project_dir(pid) / "project.json", {}) or {}
        new = _merge(_merge(config.PROJECT_DEFAULTS, raw), clean_fields(fields))
        new.update(id=pid, updated=_now())
        new.pop("videos", None)
        save_json(new, project_dir(pid) / "project.json")
    return get(pid)


def delete(pid):
    """Move the project's folder (only BroClips' own files) to the Recycle Bin. Source videos are never touched:
    they are only referenced, and add_videos() refuses files that live inside a project folder."""
    d = project_dir(pid)
    if not d.exists():
        return True
    root = config.projects_dir().resolve()
    if d.resolve().parent != root:
        raise ValueError("Refusing to delete a folder outside BroClips' projects folder.")
    ok = to_recycle_bin([d])
    log.info("project %s deleted (Recycle Bin): %s", pid, ok)
    return ok


# ---------- videos ----------
def _inside_a_project(path):
    """True if a file lives inside one of BroClips' own project folders (its outputs, not a recording)."""
    root = config.projects_dir().resolve()
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return False
    return len(rel.parts) >= 2 and (root / rel.parts[0] / "project.json").is_file()


def _probe(path):
    """media.probe() info, or {} while media.py does not exist yet. Its errors go to the caller."""
    try:
        media = importlib.import_module(f"{__package__}.media")
    except ModuleNotFoundError as ex:
        if ex.name == f"{__package__}.media":
            return {}
        raise
    probe = getattr(media, "probe", None)
    info = probe(str(path)) if probe else {}
    return info if isinstance(info, dict) else {}


def _duration(info):
    for k in ("dur", "duration"):
        v = info.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return round(float(v), 2)
    fmt = info.get("format") if isinstance(info.get("format"), dict) else {}
    try:
        return round(float(fmt.get("duration")), 2)
    except (TypeError, ValueError):
        return None


def _video_files(pid):
    d = project_dir(pid) / "videos"
    return sorted(d.glob("*/video.json")) if d.is_dir() else []


def add_videos(pid, paths):
    """Add video files to a project by path (each must exist; nothing is copied or moved).
    -> {"added": [video states], "skipped": [{"path", "why"}]}"""
    if get(pid) is None:
        raise ValueError("Unknown project.")
    added, skipped = [], []
    with _lock:
        have = {}
        for f in _video_files(pid):
            src = (load_json(f, {}) or {}).get("src")
            if src:
                have[str(Path(src)).lower()] = f.parent.name
        for raw in paths or []:
            s = str(raw or "").strip().strip('"').strip("'").strip()
            if not s:
                continue
            p = Path(s).expanduser()
            if not p.is_absolute() or not p.is_file():
                skipped.append({"path": s, "why": "File not found. Check the path (it must be the full path)."})
                continue
            p = p.resolve()
            if p.suffix.lower() not in VIDEO_EXTS:
                skipped.append({"path": s, "why": "This is not a video file BroClips can read (mp4, mkv, mov, …)."})
                continue
            if str(p).lower() in have:
                skipped.append({"path": s, "why": "Already in this project."})
                continue
            if _inside_a_project(p):
                skipped.append({"path": s, "why": "This file was made by BroClips. Add your original recording."})
                continue
            try:
                info = _probe(p) or {}
            except Exception as ex:
                log.warning("probe failed for %s: %s", p, ex)
                skipped.append({"path": s, "why": f"Could not read this video ({str(ex).strip()[:200]})."})
                continue
            if info.get("has_video") is False:
                skipped.append({"path": s, "why": "This file has no picture (only sound). BroClips needs a video."})
                continue
            if info.get("n_audio") == 0:
                skipped.append({"path": s, "why": "This video has no sound. BroClips needs someone talking in it."})
                continue
            vid = new_id(p.stem, "video")
            while (project_dir(pid) / "videos" / vid).exists():
                vid = new_id(p.stem, "video")
            st = p.stat()
            v = {"src": str(p), "name": p.stem, "dur": _duration(info), "added": _now(), "status": "new",
                 "size": st.st_size, "mtime": int(st.st_mtime)}
            simple = {k: x for k, x in info.items() if isinstance(x, (str, int, float, bool)) or x is None}
            if simple:
                v["probe"] = simple
            save_json(v, video_dir(pid, vid) / "video.json")
            have[str(p).lower()] = vid
            added.append(vid)
            log.info("video %s added to %s: %s", vid, pid, p)
    return {"added": [video_state(pid, v) for v in added], "skipped": skipped}


def set_video(pid, vid, fields):
    """Merge fields into video.json (later steps record their status here). Returns the new dict or None."""
    with _lock:
        f = video_dir(pid, vid) / "video.json"
        v = load_json(f)
        if not isinstance(v, dict):
            return None
        v.update(fields or {})
        save_json(v, f)
        return v


def remove_video(pid, vid):
    """Take a video out of the project: BroClips' folder for it goes to the Recycle Bin. The file itself stays."""
    d = video_dir(pid, vid)
    if not d.exists():
        return True
    if d.resolve().parent != (project_dir(pid) / "videos").resolve():
        raise ValueError("Refusing to delete a folder outside the project.")
    ok = to_recycle_bin([d])
    log.info("video %s removed from %s (Recycle Bin): %s", vid, pid, ok)
    return ok


def video_state(pid, vid):
    """A video as the UI shows it: video.json + status (from the work line, then the last result) + outputs."""
    if not (valid_id(pid) and valid_id(vid)):
        return None
    d = video_dir(pid, vid)
    v = load_json(d / "video.json")
    if not isinstance(v, dict):
        return None
    from . import jobs
    v = dict(v, id=vid, project=pid)
    v["src_exists"] = bool(v.get("src")) and Path(v["src"]).is_file()
    v["status"] = v.get("status") or "new"
    v["msg"] = ""
    j = jobs.job_for(pid, vid)
    if j:
        v["status"], v["job"] = j["status"], j
        if j["status"] == "working":
            v["progress"] = j.get("progress") or {}
    else:
        made = jobs.result_for(pid, vid, kind="make")
        if made:
            v["status"] = made["status"]
    last = jobs.result_for(pid, vid)
    if last:
        v["msg"] = last.get("msg") or ""
    v["outputs"] = {"shorts": len([f for f in (d / "shorts").glob("short_*.mp4") if ".part" not in f.name]),
                    "long": (d / "long.mp4").is_file(),
                    "thumbs": len([f for f in (d / "thumbs").glob("thumb_*.jpg") if ".part" not in f.name]),
                    "texts": (d / "texts.json").is_file()}
    return v


def list_videos(pid):
    out = [video_state(pid, f.parent.name) for f in _video_files(pid) if valid_id(f.parent.name)]
    out = [v for v in out if v]
    out.sort(key=lambda v: (v.get("added") or "", v["id"]))
    return out


# ---------- the Recycle Bin (port of LoL util.to_recycle_bin) ----------
def to_recycle_bin(paths):
    """Move files/folders to the Recycle Bin (recoverable) — never a permanent delete. True on success.
    Outside Windows they go to <data>/trash/ instead."""
    paths = [str(Path(p).resolve()) for p in paths if Path(p).exists()]
    if not paths:
        return True
    if not config.IS_WINDOWS:
        trash = config.data_dir() / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.move(p, str(trash / f"{Path(p).name}-{int(time.time())}"))
        return True
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]
    buf = ctypes.create_unicode_buffer("\0".join(paths) + "\0\0")          # double-null-terminated list
    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
    op = SHFILEOPSTRUCTW(None, FO_DELETE, ctypes.cast(buf, wintypes.LPCWSTR), None,
                         FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI, False, None, None)
    return ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op)) == 0 and not any(Path(p).exists() for p in paths)

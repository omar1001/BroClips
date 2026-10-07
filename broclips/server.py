"""The local web app (SPEC §3, §9): a stdlib ThreadingHTTPServer on 127.0.0.1 (only this PC can reach it) serving
the pages in broclips/web/ and a small JSON API. `python -m broclips` starts it and opens an Edge/Chrome --app
window; a second start just opens the window (a newer version asks an idle older one to quit first).

Later modules add their own API with the route registry:
    from .server import route, json_response, bytes_response, file_response, ApiError
    @route("GET", "/api/thing")                    # exact path
    @route("POST", "/api/thing/", prefix=True)     # every path under it; req.rest = the part after the prefix
    def thing(req): return {"ok": True}            # dict/list -> JSON; or a Response; raise ApiError("plain words")

Safety: requests must carry Host 127.0.0.1/localhost:<port>; a browser Origin must be this app; POSTs must be JSON
(so another website open in the same browser cannot drive the app or read keys). API keys never leave
secrets.json except masked ("…abcd"). /files serves only files inside a video's own BroClips folder."""
import argparse
import importlib
import json
import logging
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import assets, config, jobs, projects          # assets registers the "assets" job kind
from .providers import LLMS, STTS
from .providers.base import ProviderError
from .util import NO_WINDOW, load_json, log, lower_priority, save_json

OPTIONAL_MODULES = ("pipeline", "results_api", "thumbs_api")   # later milestones (M2, M3): imported if present
TEST_TIMEOUT = 120                  # seconds for a provider 🧪 Test (a big local model can take a while to load)
MAX_BODY = 5_000_000
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
         ".txt": "text/plain; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".ico": "image/x-icon", ".svg": "image/svg+xml", ".webp": "image/webp", ".mp4": "video/mp4",
         ".webm": "video/webm", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".srt": "text/plain; charset=utf-8",
         ".ttf": "font/ttf"}
VIDEO_PATTERNS = " ".join("*" + e for e in projects.VIDEO_EXTS)

# OpenAI-compatible services: picking one fills its address; its key is stored under its own name, and the key name
# is always derived from the address here (so a key is only ever sent to the service it belongs to).
LLM_PRESETS = [
    {"id": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1", "key": "openai",
     "model": "gpt-4o-mini", "key_url": "https://platform.openai.com/api-keys"},
    {"id": "openrouter", "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key": "openrouter",
     "model": "", "key_url": "https://openrouter.ai/keys"},
    {"id": "groq", "name": "Groq", "base_url": "https://api.groq.com/openai/v1", "key": "groq", "model": "",
     "key_url": "https://console.groq.com/keys"},
    {"id": "lmstudio", "name": "LM Studio (on this PC)", "base_url": "http://127.0.0.1:1234/v1", "key": "lmstudio",
     "model": "", "key_url": ""},
    {"id": "custom", "name": "Another service (type its address)", "base_url": "", "key": "custom", "model": "",
     "key_url": ""},
]
STT_PRESETS = [
    {"id": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1", "key": "openai",
     "model": "whisper-1", "key_url": "https://platform.openai.com/api-keys"},
    {"id": "groq", "name": "Groq", "base_url": "https://api.groq.com/openai/v1", "key": "groq",
     "model": "whisper-large-v3", "key_url": "https://console.groq.com/keys"},
    {"id": "custom", "name": "Another service (type its address)", "base_url": "", "key": "custom", "model": "",
     "key_url": ""},
]
KEY_NAMES = ("gemini", "anthropic", "openai", "openrouter", "groq", "lmstudio", "custom")


# ---------- the route registry ----------
_EXACT = {}         # (METHOD, path) -> fn
_PREFIX = []        # [(METHOD, prefix, fn)], longest prefix first


def route(method, path, prefix=False):
    def deco(fn):
        if prefix:
            _PREFIX.append((method.upper(), path, fn))
            _PREFIX.sort(key=lambda r: -len(r[1]))
        else:
            _EXACT[(method.upper(), path)] = fn
        return fn
    return deco


def _find(method, path):
    fn = _EXACT.get((method, path))
    if fn:
        return fn, ""
    for m, p, f in _PREFIX:
        if m == method and path.startswith(p):
            return f, path[len(p):]
    return None, ""


class ApiError(Exception):
    """Raise in a route: the page shows the message as is."""

    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


class Response:
    def __init__(self, body=b"", ctype="application/json; charset=utf-8", code=200, headers=None, file=None,
                 start=0, end=-1):
        self.body, self.ctype, self.code, self.headers = body, ctype, code, dict(headers or {})
        self.file, self.start, self.end = file, start, end


def json_response(obj, code=200, headers=None):
    return Response(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", code,
                    dict({"Cache-Control": "no-store"}, **(headers or {})))


def bytes_response(data, ctype, code=200, headers=None):
    return Response(data, ctype, code, dict({"Cache-Control": "no-store"}, **(headers or {})))


def _ctype(path):
    return TYPES.get(Path(path).suffix.lower()) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def file_response(path, req=None):
    """A file with HTTP Range support (video players seek with it). Only ever opened for reading."""
    path = Path(path)
    size = path.stat().st_size
    head = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    rng = (req.headers.get("Range") or "").strip() if req is not None else ""
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng) if rng else None
    if m and (m.group(1) or m.group(2)):               # one simple range; anything fancier gets the whole file
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:
            start, end = max(0, size - int(m.group(2))), size - 1
        if start >= size or start > end:
            return Response(b"", "text/plain", 416, dict(head, **{"Content-Range": f"bytes */{size}"}))
        return Response(ctype=_ctype(path), code=206, file=path, start=start, end=end,
                        headers=dict(head, **{"Content-Range": f"bytes {start}-{end}/{size}"}))
    return Response(ctype=_ctype(path), code=200, file=path, start=0, end=size - 1, headers=head)


class Request:
    def __init__(self, handler, method, path, query, body, rest=""):
        self.handler, self.method, self.path, self.query, self.body, self.rest = \
            handler, method, path, query, body, rest
        self.headers = handler.headers


# ---------- the HTTP side ----------
class Handler(BaseHTTPRequestHandler):
    server_version = "BroClips"

    def log_message(self, *a):          # quiet: errors are logged where they happen
        pass

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _allowed(self, method):
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in self.server.hosts:
            return False                                    # DNS-rebinding guard
        origin = self.headers.get("Origin")
        if origin is not None and origin.strip().lower() not in self.server.origins:
            return False                                    # another website in the same browser
        if method == "POST":
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            site = (self.headers.get("Sec-Fetch-Site") or "same-origin").lower()
            if ctype != "application/json" or site not in ("same-origin", "none"):
                return False
        return True

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError("Bad request.", 400)
        if n < 0:
            raise ApiError("Bad request.", 400)
        if n > MAX_BODY:
            raise ApiError("Too much data.", 413)
        raw = self.rfile.read(n) if n else b""
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ApiError("Bad request (not JSON).", 400)
        if not isinstance(body, dict):
            raise ApiError("Bad request.", 400)
        return body

    def _dispatch(self, method):
        try:
            u = urlparse(self.path)
            path = unquote(u.path)
            if not self._allowed(method):
                resp = json_response({"ok": False, "error": "Not allowed."}, 403)
            else:
                body = self._body() if method == "POST" else {}
                query = {k: v[0] for k, v in parse_qs(u.query).items()}
                fn, rest = _find(method, path)
                if fn is not None:
                    out = fn(Request(self, method, path, query, body, rest))
                    resp = out if isinstance(out, Response) else json_response({"ok": True} if out is None else out)
                else:
                    resp = (_static(path) if method == "GET" else None) or \
                        json_response({"ok": False, "error": "Not found."}, 404)
        except ApiError as ex:
            resp = json_response({"ok": False, "error": str(ex)}, ex.code)
        except Exception as ex:
            log.exception("request %s %s failed", method, self.path)
            resp = json_response({"ok": False, "error": f"Something went wrong ({type(ex).__name__}: {ex})."}, 500)
        self._send(resp)

    def _send(self, resp):
        try:
            self.send_response(resp.code)
            self.send_header("Content-Type", resp.ctype)
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in resp.headers.items():
                self.send_header(k, v)
            head_only = self.command == "HEAD"
            if resp.file is None:
                self.send_header("Content-Length", str(len(resp.body)))
                self.end_headers()
                if not head_only and resp.body:
                    self.wfile.write(resp.body)
                return
            left = max(0, resp.end - resp.start + 1)
            self.send_header("Content-Length", str(left))
            self.end_headers()
            if head_only or not left:
                return
            with open(resp.file, "rb") as f:
                f.seek(resp.start)
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass


class AppServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = not config.IS_WINDOWS     # on Windows SO_REUSEADDR would let two copies share the port

    def __init__(self, port):
        self.on_quit = None
        super().__init__(("127.0.0.1", port), Handler)
        real = self.server_address[1]
        self.port = real
        self.hosts = {f"127.0.0.1:{real}", f"localhost:{real}"}
        self.origins = {f"http://{h}" for h in self.hosts}

    def server_bind(self):
        if config.IS_WINDOWS and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def quit(self):
        """Asked by /api/quit (a newer BroClips wants the port): stop serving; main() then exits."""
        if self.on_quit:
            self.on_quit()
        else:
            threading.Thread(target=self.shutdown, daemon=True).start()


def make_server(port=config.PORT):
    """The server, bound but not yet serving (port 0 = any free port, for tests)."""
    return AppServer(port)


def _static(path):
    name = "index.html" if path in ("", "/") else path.lstrip("/")
    if name == "favicon.ico":
        name = "broclips.ico"
    web = config.WEB_DIR.resolve()
    f = (web / name).resolve()
    if not f.suffix and f.with_suffix(".html").is_file():
        f = f.with_suffix(".html")
    if web not in f.parents or not f.is_file():
        return None
    return Response(f.read_bytes(), _ctype(f), 200, {"Cache-Control": "no-cache"})


# ---------- small helpers ----------
def first_run():
    return not (config.data_dir() / "settings.json").exists()


def _ui_state():
    return load_json(config.cache_dir() / "ui.json", {}) or {}


def _set_ui_state(**kw):
    s = _ui_state()
    s.update(kw)
    save_json(s, config.cache_dir() / "ui.json")


def _console_python():
    """python.exe next to pythonw.exe: run hidden (CREATE_NO_WINDOW) it shows no console at all."""
    exe = Path(sys.executable)
    py = exe.with_name("python.exe")
    return str(py if exe.name.lower() == "pythonw.exe" and py.exists() else exe)


_dialog_lock = threading.Lock()
_FILES_CODE = ("import json,sys,tkinter as t,tkinter.filedialog as f\n"
               "r=t.Tk();r.withdraw();r.attributes('-topmost',1)\n"
               "p=f.askopenfilenames(parent=r,title=sys.argv[1],initialdir=sys.argv[2] or None,"
               "filetypes=[('Videos',sys.argv[3]),('All files','*.*')])\n"
               "print(json.dumps(list(r.tk.splitlist(p))))")
_FOLDER_CODE = ("import json,sys,tkinter as t,tkinter.filedialog as f\n"
                "r=t.Tk();r.withdraw();r.attributes('-topmost',1)\n"
                "print(json.dumps(f.askdirectory(parent=r,title=sys.argv[1],initialdir=sys.argv[2] or None,"
                "mustexist=False) or ''))")


def _dialog(code, *args):
    """A Windows file/folder dialog: tkinter in a separate hidden-console process (no window of ours)."""
    if not _dialog_lock.acquire(blocking=False):
        raise ApiError("A file window is already open — look for it (it may be behind this window).", 409)
    try:
        p = subprocess.run([_console_python(), "-c", code, *[str(a) for a in args]], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NO_WINDOW, timeout=900)
    except subprocess.TimeoutExpired:
        return None
    finally:
        _dialog_lock.release()
    lines = (p.stdout or "").strip().splitlines()
    if p.returncode != 0 or not lines:
        log.warning("file dialog failed (%s): %s", p.returncode, (p.stderr or "")[-500:])
        raise ApiError("The file window could not open. You can paste the file paths instead.", 500)
    try:
        return json.loads(lines[-1])
    except ValueError:
        raise ApiError("The file window gave a strange answer. You can paste the file paths instead.", 500)


def _deep_merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def key_name_for(base_url):
    """Which stored key an OpenAI-compatible address may use: a known service's own key, else 'custom'."""
    u = str(base_url or "").strip().rstrip("/").lower()
    for p in LLM_PRESETS + STT_PRESETS:
        if p["base_url"] and u == p["base_url"].lower():
            return p["key"]
    return "custom"


# ---------- settings ----------
_DROP = object()
ENUMS = {("thinking", "provider"): tuple(LLMS), ("listening", "provider"): tuple(STTS),
         ("encoder",): ("auto", "nvenc", "x264"), ("listening", "local", "device"): ("auto", "cuda", "cpu")}
RANGES = {("max_shorts",): (1, 30), ("thinking", "ollama", "num_ctx"): (2048, 262144),
          ("thinking", "ollama", "manage", "port"): (1024, 65535)}
URL_FIELDS = {("thinking", "ollama", "url"), ("thinking", "openai", "base_url"), ("listening", "openai", "base_url")}


def _programs(v):
    if isinstance(v, str):
        v = v.splitlines()
    if not isinstance(v, list):
        return _DROP
    out = []
    for x in v:
        name = Path(str(x).strip().strip('"')).name[:100] if str(x).strip() else ""
        if name and name.lower() not in {o.lower() for o in out}:
            out.append(name)
    return out[:20]


def _boxes(v):
    if not isinstance(v, list):
        return _DROP
    out = []
    for b in v:
        if not isinstance(b, (list, tuple)):
            continue
        try:
            x, y, w, h = (int(round(float(n))) for n in b)
        except (TypeError, ValueError):
            continue
        if 0 <= x <= 7680 and 0 <= y <= 4320 and 0 < w <= 7680 and 0 < h <= 4320:
            out.append([x, y, w, h])
    return out[:20]


def _clean(value, default, path=()):
    """Keep only keys that exist in config.DEFAULTS, converted to the default's type; a bad value is dropped (the
    saved one stays)."""
    if isinstance(default, dict):
        if not isinstance(value, dict):
            return _DROP
        out = {}
        for k, dv in default.items():
            if k in value and not (path + (k,) in {("thinking", "openai", "key_name"),
                                                   ("listening", "openai", "key_name")}):
                v = _clean(value[k], dv, path + (k,))
                if v is not _DROP:
                    out[k] = v
        return out
    if path in ENUMS:
        return value if value in ENUMS[path] else _DROP
    if path == ("pause_while_running",):
        return _programs(value)
    if path == ("blur_boxes",):
        return _boxes(value)
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)
    if isinstance(default, (int, float)):
        try:
            n = float(value)
        except (TypeError, ValueError):
            return _DROP
        lo, hi = RANGES.get(path, (-1e12, 1e12))
        n = max(lo, min(hi, n))
        return int(round(n)) if isinstance(default, int) else n
    if isinstance(default, str):
        if not isinstance(value, (str, int, float)):
            return _DROP
        s = str(value).strip()[:1000]
        if path in URL_FIELDS and s and not re.match(r"https?://", s, re.I):
            return _DROP
        if path == ("language",) and not re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?|auto", s):
            return _DROP
        return s
    return _DROP


def clean_settings(new):
    out = _clean(new or {}, config.DEFAULTS)
    return {} if out is _DROP else out


def save_settings(partial):
    """Merge a (partial) settings dict onto the saved settings and save. Key names are derived, never sent."""
    merged = _deep_merge(config.settings(), clean_settings(partial))
    for role in ("thinking", "listening"):
        o = merged[role]["openai"]
        o["key_name"] = key_name_for(o.get("base_url"))
    config.save_settings(merged)
    return merged


def move_data_dir(new):
    """Point BroClips at another data folder; settings come along (copied), the API keys move with it."""
    new = Path(str(new)).expanduser()
    old = config.data_dir()
    if not new.is_absolute():
        raise ApiError("Choose a full folder path (for example E:\\BroClips).")
    try:
        new.mkdir(parents=True, exist_ok=True)
        if new.resolve() == old.resolve():
            return
        if (old / "settings.json").exists() and not (new / "settings.json").exists():
            shutil.copy2(old / "settings.json", new / "settings.json")
        if (old / "secrets.json").exists() and not (new / "secrets.json").exists():
            shutil.copy2(old / "secrets.json", new / "secrets.json")
            (old / "secrets.json").unlink()
        config.set_data_dir(new)
    except OSError as ex:
        raise ApiError(f"Cannot use that folder: {ex}")
    log.info("data folder moved: %s -> %s", old, new)


def settings_payload():
    return {"settings": config.settings(), "keys": config.masked_secrets(), "first_run": first_run(),
            "data_dir": str(config.data_dir()), "projects_dir": str(config.projects_dir()),
            "default_data_dir": str(config.default_data_dir()), "llm_presets": LLM_PRESETS,
            "stt_presets": STT_PRESETS, "assets": assets.status()}


# ---------- API: app ----------
@route("GET", "/api/version")
def api_version(req):
    return {"app": config.APP_NAME, "version": config.VERSION, "busy": jobs.busy()}


@route("POST", "/api/quit")
def api_quit(req):
    if jobs.running():                          # the waiting line is saved; only a running job blocks
        return {"ok": False, "error": "BroClips is working on a job right now."}
    threading.Timer(0.3, req.handler.server.quit).start()
    return {"ok": True}


@route("GET", "/api/state")
def api_state(req):
    return {"work": jobs.state(), "version": config.VERSION, "first_run": first_run()}


@route("POST", "/api/open_folder")
def api_open_folder(req):
    b = req.body
    which = {"data": config.data_dir, "projects": config.projects_dir, "logs": config.logs_dir,
             "models": config.models_dir, "assets": config.assets_dir}
    if b.get("project"):
        try:
            d = projects.video_dir(b["project"], b["video"]) if b.get("video") else projects.project_dir(b["project"])
        except ValueError:
            raise ApiError("Unknown project.", 404)
    elif b.get("which") in which:
        d = which[b["which"]]()
    else:
        raise ApiError("Unknown folder.")
    if not d.is_dir():
        raise ApiError("That folder does not exist yet.", 404)
    if config.IS_WINDOWS:
        os.startfile(str(d))
    else:
        webbrowser.open(d.as_uri())
    return {"ok": True, "path": str(d)}


# ---------- API: projects and videos ----------
@route("GET", "/api/projects")
def api_projects(req):
    return {"projects": projects.list_projects()}


@route("POST", "/api/projects")
def api_create_project(req):
    return {"ok": True, "project": projects.create(req.body.get("name", ""))}


def _ids(rest):
    parts = [p for p in rest.split("/") if p]
    if not parts or not all(projects.valid_id(p) or p in ("videos", "delete", "remove", "make", "cancel")
                            for p in parts):
        raise ApiError("Not found.", 404)
    return parts


@route("GET", "/api/projects/", prefix=True)
def api_project_get(req):
    parts = _ids(req.rest)
    if len(parts) == 1:
        p = projects.get(parts[0])
        if p is None:
            raise ApiError("This project does not exist (any more).", 404)
        return {"project": p, "videos": projects.list_videos(parts[0])}
    if len(parts) == 3 and parts[1] == "videos":
        v = projects.video_state(parts[0], parts[2])
        if v is None:
            raise ApiError("This video is not in the project (any more).", 404)
        return {"video": v, "project": projects.get(parts[0])}
    raise ApiError("Not found.", 404)


@route("POST", "/api/projects/", prefix=True)
def api_project_post(req):
    parts = _ids(req.rest)
    pid = parts[0]
    if projects.get(pid) is None:
        raise ApiError("This project does not exist (any more).", 404)
    b = req.body
    if len(parts) == 1:                                             # save the project's settings
        fields = b.get("fields") if isinstance(b.get("fields"), dict) else b
        return {"ok": True, "project": projects.update(pid, fields)}
    if parts[1:] == ["delete"]:
        cur = jobs.running()
        if cur and cur.get("project") == pid:
            raise ApiError("BroClips is working on this project right now. Press ⏹ Stop first, then delete it.", 409)
        jobs.cancel(project=pid)
        if not projects.delete(pid):
            raise ApiError("Could not move the project to the Recycle Bin (is one of its files open?).", 409)
        return {"ok": True}
    if parts[1:] == ["videos"]:                                     # add videos by path
        paths = b.get("paths")
        if isinstance(paths, str):
            paths = paths.splitlines()
        if not isinstance(paths, list) or not paths:
            raise ApiError("No file paths given.")
        out = projects.add_videos(pid, [str(p) for p in paths][:500])
        if out["added"]:
            _set_ui_state(last_video_dir=str(Path(out["added"][-1]["src"]).parent))
        return dict(out, ok=True)
    if len(parts) == 4 and parts[1] == "videos":
        vid, action = parts[2], parts[3]
        v = projects.video_state(pid, vid)
        if v is None:
            raise ApiError("This video is not in the project (any more).", 404)
        if action == "remove":
            cur = jobs.running()
            if cur and cur.get("project") == pid and cur.get("video") == vid:
                raise ApiError("BroClips is working on this video right now. Press ⏹ Stop first.", 409)
            jobs.cancel(project=pid, video=vid)
            if not projects.remove_video(pid, vid):
                raise ApiError("Could not remove it (is one of its files open?).", 409)
            return {"ok": True, "msg": "Removed from this project. The video file itself was not touched."}
        if action == "make":
            if not jobs.registered("make"):
                raise ApiError("Making clips comes in the next version of BroClips. For now you can set up "
                               "projects, add videos and choose your AI.", 409)
            if not v["src_exists"]:
                raise ApiError(f"The video file is not there any more: {v.get('src')}", 409)
            job = jobs.enqueue("make", project=pid, video=vid, args={"again": bool(b.get("again"))},
                               title=v.get("name") or vid)
            return {"ok": True, "job": job}
        if action == "cancel":
            return {"ok": True, "stopped": jobs.cancel(project=pid, video=vid)}
    raise ApiError("Not found.", 404)


@route("GET", "/files/", prefix=True)
def files(req):
    """/files/<pid>/<vid>/<path>: what BroClips made for a video (players seek with Range). Only files inside that
    video's own folder; anything else is refused."""
    parts = req.rest.split("/", 2)
    if len(parts) < 3 or not parts[2]:
        raise ApiError("Not found.", 404)
    pid, vid, sub = parts
    if not (projects.valid_id(pid) and projects.valid_id(vid)):
        raise ApiError("Not allowed.", 403)
    base = projects.video_dir(pid, vid).resolve()
    try:
        f = (base / sub).resolve()
    except (OSError, ValueError):
        raise ApiError("Not allowed.", 403)
    if base not in f.parents:
        raise ApiError("Not allowed.", 403)
    if not f.is_file():
        raise ApiError("Not found.", 404)
    return file_response(f, req)


@route("POST", "/api/pick_videos")
def api_pick_videos(req):
    start = str(req.body.get("initial") or _ui_state().get("last_video_dir") or "")
    paths = _dialog(_FILES_CODE, "Choose your videos (they are only read, never changed)",
                    start if start and Path(start).is_dir() else "", VIDEO_PATTERNS)
    return {"ok": True, "paths": [str(Path(p)) for p in (paths or [])]}


@route("POST", "/api/pick_folder")
def api_pick_folder(req):
    start = str(req.body.get("initial") or "")
    path = _dialog(_FOLDER_CODE, str(req.body.get("title") or "Choose a folder")[:120],
                   start if start and Path(start).is_dir() else "")
    return {"ok": True, "path": str(Path(path)) if path else ""}


# ---------- API: settings, keys, AI tests ----------
@route("GET", "/api/settings")
def api_settings(req):
    return settings_payload()


@route("POST", "/api/settings")
def api_save_settings(req):
    b = req.body
    new = b.get("settings") if isinstance(b.get("settings"), dict) else {}
    keys = b.get("keys") if isinstance(b.get("keys"), dict) else {}
    out_dir = str(new.get("output_dir") or "").strip()
    if out_dir:                                     # a bad output folder would break every project page
        p = Path(out_dir).expanduser()
        if not p.is_absolute():
            raise ApiError("The output folder must be a full path (for example E:\\BroClips\\projects).")
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise ApiError(f"Cannot use that output folder: {ex}")
    new_dir = str(b.get("data_dir") or "").strip()
    if new_dir and Path(new_dir).expanduser().resolve() != config.data_dir().resolve():
        if jobs.busy():
            raise ApiError("Wait until the work line is empty, then change the data folder.", 409)
        move_data_dir(new_dir)
    save_settings(new)
    for name, value in keys.items():
        v = str(value or "").strip()
        if name in KEY_NAMES and v and not v.startswith("…") and len(v) <= 500:    # empty = keep the old key
            config.set_secret(name, v)
    return dict(settings_payload(), ok=True)


@route("POST", "/api/keys/remove")
def api_remove_key(req):
    name = req.body.get("name")
    if name not in KEY_NAMES:
        raise ApiError("Unknown key.")
    config.set_secret(name, "")
    return dict(settings_payload(), ok=True)


_test_lock = threading.Lock()


@route("POST", "/api/providers/test")
def api_test_provider(req):
    """🧪 Test: a tiny real call to the chosen AI (saved settings), in a thread with a time limit."""
    role = req.body.get("role")
    if role not in ("thinking", "listening"):
        raise ApiError("Unknown AI role.")
    if not _test_lock.acquire(blocking=False):
        return {"ok": False, "message": "A test is already running — wait a moment."}
    result = {}

    def work():
        try:
            from . import providers
            p = providers.get_llm() if role == "thinking" else providers.get_stt()
            if getattr(p, "local", False) and jobs.running():
                result["r"] = (False, "BroClips is busy with a job right now. Test again when it is done "
                                      "(only one AI runs at a time).")
                return
            result["r"] = (True, str(p.test() or "OK"))
            if getattr(p, "local", False):
                try:
                    p.unload()
                except Exception:
                    pass
        except ProviderError as ex:
            result["r"] = (False, str(ex))
        except ImportError as ex:
            missing = getattr(ex, "name", "") or ""
            result["r"] = (False, "This AI option is not available in this version of BroClips yet."
                           if missing.startswith(__package__) else
                           f"Something is not installed: {missing or ex}. Run install.bat again.")
        except Exception as ex:
            log.exception("provider test (%s) failed", role)
            result["r"] = (False, f"The test failed ({type(ex).__name__}: {ex}).")
        finally:
            _test_lock.release()

    t = threading.Thread(target=work, name="broclips-ai-test", daemon=True)
    t.start()
    t.join(TEST_TIMEOUT)
    if t.is_alive():
        return {"ok": False, "message": f"No answer after {TEST_TIMEOUT} seconds. Is the AI running? A big model "
                                        "can take a while to load the first time — try again in a minute."}
    ok, msg = result.get("r", (False, "No answer."))
    return {"ok": ok, "message": msg}


@route("GET", "/api/providers/models")
def api_models(req):
    """Installed Ollama models, for the model dropdown."""
    try:
        from . import providers
        lister = getattr(providers.get_llm(provider="ollama"), "list_models", None)
        if lister:
            return {"ok": True, "models": sorted(str(m) for m in lister() or [])}
    except Exception as ex:
        log.info("list_models: %s", ex)
    o = config.settings()["thinking"]["ollama"]
    url = f"http://127.0.0.1:{o['manage']['port']}" if o.get("manage", {}).get("enabled") else o.get("url", "")
    if not re.match(r"https?://", str(url)):
        return {"ok": False, "models": [], "error": "The Ollama address must start with http://"}
    try:
        with urllib.request.urlopen(str(url).rstrip("/") + "/api/tags", timeout=4) as r:
            data = json.loads(r.read() or b"{}")
        return {"ok": True, "models": sorted(m["name"] for m in data.get("models", []) if m.get("name"))}
    except Exception:
        return {"ok": False, "models": [], "error": "Ollama is not answering at this address. Is it running? "
                                                     "Install it from ollama.com, or start it from the Start menu."}


# ---------- API: the work line, downloads ----------
@route("POST", "/api/jobs/enqueue")
def api_enqueue(req):
    b = req.body
    kind = str(b.get("kind") or "")
    if not jobs.registered(kind):
        raise ApiError("This kind of work is not in this version of BroClips.", 409)
    pid, vid = b.get("project"), b.get("video")
    if pid is not None and projects.get(pid) is None:
        raise ApiError("Unknown project.", 404)
    if vid is not None and projects.video_state(pid, vid) is None:
        raise ApiError("Unknown video.", 404)
    args = b.get("args") if isinstance(b.get("args"), dict) else {}
    return {"ok": True, "job": jobs.enqueue(kind, project=pid, video=vid, args=args,
                                            title=str(b.get("title") or "")[:80])}


@route("POST", "/api/jobs/cancel")
def api_cancel(req):
    b = req.body
    if not (b.get("id") or b.get("project")):
        raise ApiError("Say which job to stop.")
    return {"ok": True, "stopped": jobs.cancel(job_id=b.get("id"), project=b.get("project"), video=b.get("video"))}


@route("POST", "/api/jobs/run_now")
def api_run_now(req):
    return {"ok": True, "started": jobs.run_now(req.body.get("id"))}


@route("GET", "/api/assets")
def api_assets(req):
    return assets.status()


@route("POST", "/api/assets/download")
def api_assets_download(req):
    what = req.body.get("what")
    if what == "essential":
        job = jobs.enqueue("assets", args={"essential": True}, title="Fonts and emoji")
    elif what in assets.CUTOUT:
        job = jobs.enqueue("assets", args={"cutout": what}, title=assets.CUTOUT[what]["label"].capitalize())
    else:
        raise ApiError("Unknown download.")
    return {"ok": True, "job": job}


# ---------- starting the app ----------
def _url(port):
    return f"http://127.0.0.1:{port}/"


def open_window(url):
    """The app in its own clean window (Edge/Chrome --app), else the default browser."""
    pf, pf86, local = (os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"))
    for exe in (Path(pf86) / "Microsoft/Edge/Application/msedge.exe", Path(pf) / "Microsoft/Edge/Application/msedge.exe",
                Path(pf) / "Google/Chrome/Application/chrome.exe", Path(pf86) / "Google/Chrome/Application/chrome.exe",
                Path(local) / "Google/Chrome/Application/chrome.exe"):
        if not exe.is_absolute() or not exe.is_file():
            continue
        try:
            subprocess.Popen([str(exe), f"--app={url}", "--window-size=1400,920"], creationflags=NO_WINDOW)
            return
        except OSError:
            continue
    webbrowser.open(url)


def already_running(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def other_version(port):
    """{"app", "version", "busy"} of the program on our port, or None if it is not BroClips."""
    try:
        with urllib.request.urlopen(_url(port) + "api/version", timeout=2) as r:
            info = json.loads(r.read())
        return info if isinstance(info, dict) and info.get("app") == config.APP_NAME else None
    except Exception:
        return None


def _vtuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", str(v or ""))[:4]) or (0,)


def replace_old_version(port):
    """An older, idle BroClips on our port is asked to quit. True if the port is free now."""
    info = other_version(port)
    if not info or _vtuple(info.get("version")) >= _vtuple(config.VERSION):
        return False
    try:
        req = urllib.request.Request(_url(port) + "api/quit", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            if not json.loads(r.read() or b"{}").get("ok"):
                return False                    # it is busy with a job: keep it
    except Exception:
        return False
    for _ in range(40):
        time.sleep(0.25)
        if not already_running(port):
            return True
    return False


def _tell_user(msg):
    """A message the user sees even without a console (pythonw)."""
    log.error(msg)
    if sys.stdout is None and config.IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, config.APP_NAME, 0x40)
        except Exception:
            pass
    else:
        print(msg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m broclips", description="BroClips: your recordings -> Shorts, "
                                                                        "a clean long video, titles and thumbnails.")
    ap.add_argument("--port", type=int, default=config.PORT, help=f"web port (default {config.PORT})")
    ap.add_argument("--no-window", action="store_true", help="only start the server, do not open a window")
    a = ap.parse_args(argv)
    url = _url(a.port)
    if already_running(a.port) and not replace_old_version(a.port):
        if other_version(a.port):
            if not a.no_window:
                open_window(url)                # it is already running: just show it
            if sys.stdout is not None:
                print(f"BroClips is already running on {url}" + ("" if a.no_window else " - opened its window")
                      + ". To see its log here, close it first (Help page -> 'Close BroClips completely'), then "
                        "start run.bat again.")
            return 0
        _tell_user(f"BroClips could not start: port {a.port} is used by another program. Close that program, or "
                   f"start BroClips with another port: python -m broclips --port {a.port + 1}")
        return 1
    try:
        srv = make_server(a.port)
    except OSError as ex:
        _tell_user(f"BroClips could not start on port {a.port}: {ex}")
        return 1
    lower_priority()                            # games stay smooth; ffmpeg/AI children inherit it
    jobs.start()
    if not a.no_window:
        threading.Timer(0.6, open_window, [url]).start()
    log.info("BroClips %s started on %s (data folder: %s)", config.VERSION, url, config.data_dir())
    try:
        srv.serve_forever()
    finally:
        jobs.stop(3)
        srv.server_close()
        log.info("BroClips stopped")
        logging.shutdown()
    return 0


def _optional(name):
    """Import a later milestone's module if it exists (it registers its job kinds and routes)."""
    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ModuleNotFoundError as ex:
        if ex.name != f"{__package__}.{name}":
            log.warning("could not load %s: %s", name, ex)
    except Exception:
        log.exception("could not load %s", name)
    return None


for _name in OPTIONAL_MODULES:          # last, so they can `from .server import route` while this module loads
    _optional(_name)

"""Ollama (local, free) as the Thinking AI (SPEC §4.1) — ported from LoL Clips brain.py (ensure_server, warm,
chat_json, unload).

Ollama itself forces every answer into the JSON schema (`format`), thinking is off (`think: false`: much faster for a
12B model), and the model stays loaded 10 minutes between questions (`keep_alive`); `unload()` frees the graphics
card after the job (SPEC §2.2). Loading a 7 GB model from a hard disk can take minutes, hence the long time-outs.
"Start a private server" (opts["manage"]) runs `ollama serve` with its own models folder and port, without a window
(for example a second server on port 11435 that keeps its models on another disk)."""
import os
import shutil
import subprocess
import threading
from pathlib import Path

from ..util import NO_WINDOW, log
from . import _http
from ._http import HTTPError, b64, drop_rejected, extract_json, fit_schema
from .base import LLM, ProviderError

KEEP_ALIVE = "10m"
RETRIES = 2                 # extra attempts when the answer is not valid (SPEC §4.1)
RETRY_WAIT_S = 20           # a crashed runner / a model still loading needs a moment (LoL Clips)
LOAD_TIMEOUT_S = 1200
NOT_RUNNING = ("Ollama is not running — install it from ollama.com (it then starts with Windows), or add an API key "
               "in Settings.")
_START = threading.Lock()


def _default_exe():
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if local.is_file():
        return str(local)
    return shutil.which("ollama") or ""


def _norm(name):
    name = str(name or "").strip()
    return name if ":" in name else name + ":latest"


class OllamaLLM(LLM):
    name = "ollama"
    local = True

    def __init__(self, opts=None):
        self.opts = dict(opts or {})
        self.model = str(self.opts.get("model") or "gemma3:12b").strip()
        self.num_ctx = int(self.opts.get("num_ctx") or 16384)
        m = self.opts.get("manage") or {}
        self.manage = dict(m) if m.get("enabled") else None
        if self.manage:
            self.url = f"http://127.0.0.1:{int(self.manage.get('port') or 11435)}"
        else:
            self.url = str(self.opts.get("url") or "http://127.0.0.1:11434").rstrip("/")
        self._warm = False
        self._vision = None

    # ---------- HTTP ----------
    def _get(self, path, timeout=30):
        return _http.get_json(self.url + path, timeout=timeout, retries=0, label="Ollama")

    def _post(self, path, body, timeout=60):
        return _http.post_json(self.url + path, body, timeout=timeout, retries=0, label="Ollama")

    def _alive(self):
        try:
            self._get("/api/version", timeout=3)
            return True
        except ProviderError:
            return False

    def _missing(self):
        if self.manage and self.manage.get("models_dir"):
            return (f"Model {self.model} is not in the private Ollama models folder ({self.manage['models_dir']}). "
                    f"Pick an installed model in Settings.")
        return f"Model {self.model} is not installed. In a terminal run: ollama pull {self.model}"

    # ---------- server ----------
    def ensure_server(self):
        """Make sure an Ollama server answers; start the private one when Settings ask for it."""
        if self._alive():
            return
        if not self.manage:
            raise ProviderError(NOT_RUNNING)
        exe = str(self.manage.get("exe") or "").strip() or _default_exe()
        if not exe or not Path(exe).is_file():
            raise ProviderError("The Ollama program was not found. Install Ollama from ollama.com, or set its path "
                                "in Settings (Thinking AI → private server).")
        port = int(self.manage.get("port") or 11435)
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MAX_LOADED_MODELS="1",
                   OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_NOPRUNE="1",
                   OLLAMA_LOAD_TIMEOUT="20m")               # a 7 GB model loads slowly from a hard disk
        if self.manage.get("models_dir"):
            env["OLLAMA_MODELS"] = str(self.manage["models_dir"])
        with _START:
            if self._alive():                               # another thread started it meanwhile
                return
            log.info("starting a private Ollama server on port %d (models in %s)", port,
                     self.manage.get("models_dir") or "the default folder")
            subprocess.Popen([exe, "serve"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
            for _ in range(30):
                _http._sleep(1)
                if self._alive():
                    return
        raise ProviderError(f"The private Ollama server did not start on port {port}.")

    def _opts(self):
        return {"num_ctx": self.num_ctx}

    def warm(self):
        """Load the model once, with a long time-out. Uses the same options as chat_json: a different context size
        would make Ollama load the whole model again."""
        if self._warm:
            return
        self.ensure_server()
        for attempt in range(3):
            try:
                self._post("/api/generate", {"model": self.model, "keep_alive": KEEP_ALIVE, "options": self._opts()},
                           timeout=LOAD_TIMEOUT_S)
                self._warm = True
                return
            except HTTPError as ex:
                if ex.status == 404:
                    raise ProviderError(self._missing()) from None
                if attempt == 2:
                    raise ProviderError(f"Ollama could not load {self.model}: {ex}") from None
                log.warning("loading %s failed (%s), retrying", self.model, ex)
                _http._sleep(RETRY_WAIT_S)
                self.ensure_server()

    @property
    def vision(self):
        """Can the model look at pictures? (Ollama's /api/show lists "vision" in its capabilities.)"""
        if self._vision is None:
            try:
                self.ensure_server()
                info = self._post("/api/show", {"model": self.model}, timeout=30)
            except ProviderError:
                return False                                # not cached: ask again next time
            self._vision = "vision" in (info.get("capabilities") or [])
        return self._vision

    # ---------- the interface ----------
    def chat_json(self, system, user, schema, images=None, temperature=0.2, max_tokens=1500):
        self.warm()
        msg = {"role": "user", "content": user}
        if images:
            if self.vision:
                msg["images"] = [b64(b) for b in images]
            else:
                log.info("%s cannot look at pictures — asking with text only", self.model)
        body = {"model": self.model, "stream": False, "think": False, "format": schema, "keep_alive": KEEP_ALIVE,
                "messages": [{"role": "system", "content": system}, msg],
                "options": {**self._opts(), "temperature": temperature, "num_predict": int(max_tokens)}}
        attempt = 0
        while attempt <= RETRIES:
            try:
                r = self._post("/api/chat", body, timeout=LOAD_TIMEOUT_S)
            except HTTPError as ex:
                if ex.status == 404:
                    raise ProviderError(self._missing()) from None
                if ex.status == 400 and drop_rejected(ex, body, [("think",)]):    # an Ollama without `think`
                    continue
                if attempt == RETRIES:
                    if ex.status == 0 and not self._alive():
                        raise ProviderError(NOT_RUNNING) from None
                    raise
                log.warning("Ollama request failed (%s), retrying", ex)    # model still loading / runner hiccup
                attempt += 1
                _http._sleep(RETRY_WAIT_S)
                self._warm = False
                self.warm()
                continue
            txt = (r.get("message") or {}).get("content") or ""
            data = fit_schema(extract_json(txt), schema)
            if data is not None:
                return data
            log.warning("Ollama's answer did not fit the schema (attempt %d): %s", attempt + 1, txt[:200])
            body["options"]["seed"] = 1000 + attempt                        # a different answer next time
            attempt += 1
        return None

    def unload(self):
        """Free the graphics card: unload every model this server has loaded (and only those)."""
        try:
            loaded = self._get("/api/ps", timeout=10).get("models") or []
            for m in loaded:
                self._post("/api/generate", {"model": m.get("name") or m.get("model"), "keep_alive": 0}, timeout=60)
        except ProviderError:
            pass
        self._warm = False

    def list_models(self):
        """Installed model names (for the Settings dropdown)."""
        self.ensure_server()
        names = (m.get("name") or m.get("model") for m in (self._get("/api/tags").get("models") or []))
        return sorted(n for n in names if n)

    def test(self):
        self.ensure_server()
        version = self._get("/api/version").get("version", "?")
        if _norm(self.model) not in {_norm(n) for n in self.list_models()}:
            raise ProviderError(self._missing())
        return f"Ready: Ollama {version}, model {self.model}" + (" (can see pictures)" if self.vision else "")

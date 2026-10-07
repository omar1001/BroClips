"""The work line (SPEC §2.2): ONE worker thread runs ONE job at a time, so a home PC never has two heavy things
(speech-to-text, AI, rendering, a big download) going at once.

A job is a dict {id, kind, project, video, args, created, title}. Modules teach the line their kinds with
register(kind, fn); the worker calls fn(job, progress, cancelled):
  progress(step, label, detail="", frac=None)  what the UI shows (step 1..steps, frac 0..1 inside the step); it is
                                               also where Stop and "pause while these programs run" take effect:
                                               it raises Cancelled / Paused (BaseExceptions, so a job's own
                                               `except Exception` cannot swallow them)
  cancelled() -> bool                          True after the user pressed Stop
The return value (a short text) becomes the job's result message; a JobError / ProviderError message is shown as
is; any other exception is logged with its traceback and turned into a plain message.

The line is saved to <data>/queue.json after every change. A job that was running when the app quit goes back to the
front when it starts again (pipeline stages are resumable). Identical jobs are not added twice. While a program from
Settings "pause while these programs run" is open (e.g. a game's .exe) nothing starts, and a running job is
stopped at its next progress() and put back at the front — unless the user pressed "Run now".
After every job the local AI models are unloaded (one model in memory at a time; 16 GB RAM PCs)."""
import json
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import config
from .providers.base import ProviderError
from .util import load_json, log, save_json

KEEP_RESULTS = 100          # finished-job messages kept (newest last)
MAX_RESTARTS = 3            # a job that keeps killing the app is taken out of the line after this many restarts
WAIT_TEXT = "Waiting until {prog} closes — or ▶ Run now"


class Cancelled(BaseException):
    """Raised by progress() after the user pressed Stop."""


class Paused(BaseException):
    """Raised by progress() when a 'pause while running' program started; args[0] = its name."""


class JobError(Exception):
    """A plain-English problem a job wants to show as is (no traceback needed)."""


class UnknownKind(ValueError):
    """enqueue() of a kind nobody registered (that feature is not in this version)."""


_KINDS = {}                 # kind -> {"fn", "steps", "title"}


def register(kind, fn, steps=1, title=""):
    """Teach the work line a kind of job (see the module docstring for fn's signature)."""
    _KINDS[kind] = {"fn": fn, "steps": max(1, int(steps)), "title": title or kind}


def registered(kind):
    return kind in _KINDS


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _norm_exe(name):
    n = str(name or "").strip().lower()
    return n[:-4] if n.endswith(".exe") else n


def _which_running(names):
    """The first of these programs (exe names, '.exe' optional, any case) that is running now, or ''."""
    want = {_norm_exe(n): str(n).strip() for n in (names or []) if _norm_exe(n)}
    if not want:
        return ""
    try:
        import psutil
        for p in psutil.process_iter(["name"]):
            hit = want.get(_norm_exe(p.info.get("name")))
            if hit:
                return hit
    except Exception:                                   # psutil missing or a process vanished mid-scan
        return ""
    return ""


def _pause_programs():
    try:
        return config.settings().get("pause_while_running") or []
    except Exception:
        return []


def _unload_models():
    """Free local AI memory after a job: unload every provider whose code was used in this run (a provider that
    was never imported has nothing loaded, and importing it just to unload would waste time and memory)."""
    try:
        from . import providers
    except Exception:
        return
    for table, get in ((providers.LLMS, providers.get_llm), (providers.STTS, providers.get_stt)):
        for name, spec in table.items():
            if f"{providers.__name__}.{spec.split(':')[0]}" not in sys.modules:
                continue
            try:
                get(provider=name).unload()
            except Exception as ex:
                log.info("unload %s: %s", name, ex)


def _key(job):
    return json.dumps([job.get("kind"), job.get("project"), job.get("video"), job.get("args") or {}],
                      sort_keys=True, ensure_ascii=False, default=str)


def _brief(job):
    return {k: job.get(k) for k in ("id", "kind", "project", "video", "title", "force", "created")}


def _plain(ex):
    text = (str(ex).strip().splitlines() or [""])[0][:200]
    return (f"⚠️ Something went wrong ({type(ex).__name__}{': ' + text if text else ''}). "
            f"Details are in the log folder: {config.logs_dir()}")


class WorkLine:
    """The line itself. The app uses the module-level LINE; tests make their own with a temp file and short waits."""

    def __init__(self, path=None, idle_wait=1.0, pause_wait=5.0, check_every=5.0):
        self._path = Path(path) if path else None       # None = <data>/queue.json at the moment of use
        self.idle_wait, self.pause_wait, self.check_every = idle_wait, pause_wait, check_every
        self.lock = threading.RLock()
        self._wake = threading.Event()
        self._cancel = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.queue = []             # waiting jobs, first = next
        self.current = None         # the running job
        self.info = {}              # its progress
        self.results = []           # finished jobs: {id, kind, project, video, title, status, msg, finished}
        self.waiting_for = ""       # the program the line waits for ('' = not waiting)
        self._loaded = False

    # ---------- saving ----------
    def path(self):
        return self._path or (config.data_dir() / "queue.json")

    def load(self):
        """Read the saved line; a job that was running when the app stopped goes first again."""
        d = load_json(self.path(), {}) or {}
        if isinstance(d, list):
            d = {"jobs": d}
        results = [r for r in (d.get("results") or []) if isinstance(r, dict)]
        out = []
        for j in d.get("jobs") or []:
            if not (isinstance(j, dict) and j.get("kind") and j.get("id")):
                continue
            if j.pop("running", False):
                j["restarts"] = int(j.get("restarts") or 0) + 1
                if j["restarts"] > MAX_RESTARTS:
                    log.warning("job %s (%s) dropped: it was running when the app stopped %d times",
                                j["id"], j["kind"], j["restarts"])
                    results.append(self._result(j, "failed", "⚠️ This job stopped BroClips several times, so it "
                                                              "was taken out of the line. Details are in the log."))
                    continue
            out.append(j)
        with self.lock:
            self.queue, self.results, self._loaded = out, results[-KEEP_RESULTS:], True
            self._save()

    def _save(self):
        """Call with the lock held. The running job is saved first, marked, so a restart resumes it."""
        jobs = ([dict(self.current, running=True)] if self.current else []) + self.queue
        try:
            save_json({"jobs": jobs, "results": self.results[-KEEP_RESULTS:]}, self.path())
        except OSError as ex:
            log.warning("could not save the work line: %s", ex)

    @staticmethod
    def _result(job, status, msg):
        return dict(_brief(job), status=status, msg=msg, finished=_now())

    # ---------- what the app calls ----------
    def enqueue(self, kind, project=None, video=None, args=None, title="", force=False):
        """Add a job (or return the identical one already waiting/running). Returns a copy of the job."""
        if kind not in _KINDS:
            raise UnknownKind(kind)
        job = {"id": uuid.uuid4().hex[:10], "kind": kind, "project": project, "video": video, "args": args or {},
               "created": _now(), "title": title or _KINDS[kind]["title"], "force": bool(force)}
        key = _key(job)
        with self.lock:
            cur = self.current if self.current and not self._cancel.is_set() else None
            for j in ([cur] if cur else []) + self.queue:
                if _key(j) == key:
                    if force:
                        j["force"] = True
                        self._save()
                    return dict(j)
            self.queue.append(job)
            self._save()
        self._wake.set()
        return dict(job)

    def cancel(self, job_id=None, project=None, video=None):
        """Stop a job by id, or every job of a project (or of one video). Waiting ones leave the line; the running
        one stops at its next progress() call. Returns how many were stopped."""
        def match(j):
            if job_id:
                return j["id"] == job_id
            return project is not None and j.get("project") == project and (video is None or j.get("video") == video)
        n = 0
        with self.lock:
            keep = []
            for j in self.queue:
                if match(j):
                    n += 1
                    self.results.append(self._result(j, "stopped", "Stopped."))
                else:
                    keep.append(j)
            self.queue = keep
            if self.current and match(self.current):
                self._cancel.set()
                n += 1
            self._save()
        self._wake.set()
        return n

    def run_now(self, job_id=None):
        """"▶ Run now": start even while a pause program is open (one job, or every waiting one)."""
        hit = False
        with self.lock:
            for j in self.queue + ([self.current] if self.current else []):
                if job_id is None or j["id"] == job_id:
                    j["force"] = True
                    hit = True
            self._save()
        self._wake.set()
        return hit

    def running(self):
        with self.lock:
            return dict(self.current) if self.current else None

    def busy(self):
        with self.lock:
            return bool(self.current or self.queue)

    def job_for(self, project, video=None):
        """The running or waiting job of this video (or project): {"status": "working"|"queued"|"waiting", ...}."""
        def mine(j):
            return j.get("project") == project and (video is None or j.get("video") == video)
        with self.lock:
            if self.current and mine(self.current):
                return {"status": "working", "job": _brief(self.current), "position": 0,
                        "progress": dict(self.info), "stopping": self._cancel.is_set()}
            for k, j in enumerate(self.queue):
                if mine(j):
                    waiting = bool(self.waiting_for) and not j.get("force")
                    return {"status": "waiting" if waiting else "queued", "job": _brief(j), "position": k + 1,
                            "waiting_for": self.waiting_for if waiting else ""}
        return None

    def result_for(self, project, video=None, kind=None):
        """The newest finished-job message of this video (optionally of one kind), or None."""
        with self.lock:
            for r in reversed(self.results):
                if r.get("project") == project and (video is None or r.get("video") == video) \
                        and (kind is None or r.get("kind") == kind):
                    return dict(r)
        return None

    def state(self):
        """Everything the header and pages show about the line."""
        with self.lock:
            cur = self.current
            out = {"busy": bool(cur or self.queue), "queue": len(self.queue),
                   "waiting": [_brief(j) for j in self.queue],
                   "waiting_for": self.waiting_for if self.queue else "",
                   "current": None, "last": dict(self.results[-1]) if self.results else None}
            if cur:
                out["current"] = dict(_brief(cur), stopping=self._cancel.is_set(),
                                      **{k: self.info.get(k) for k in ("step", "of", "label", "detail", "frac", "pct")})
        out["text"] = self._text(out)
        return out

    @staticmethod
    def _text(s):
        cur, n = s["current"], s["queue"]
        if cur:
            if cur["stopping"]:
                return f"⏹ Stopping {cur['title']}…"
            label = (cur.get("label") or "Starting").rstrip("…. ")
            return f"⏳ {cur['title']} — {label}… {cur.get('pct') or 0} %" + (f"  ({n} more waiting)" if n else "")
        if s["waiting_for"]:
            return "⏸ " + WAIT_TEXT.format(prog=s["waiting_for"]) + (f"  ({n} jobs waiting)" if n > 1 else "")
        if n:
            return "⏳ Starting…"
        return "💤 Ready"

    # ---------- the worker ----------
    def start(self):
        with self.lock:
            if self._thread and self._thread.is_alive():
                return
            if not self._loaded:
                self.load()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="broclips-work-line", daemon=True)
            self._thread.start()

    def stop(self, timeout=5.0):
        """Stop the worker after the running job (used when the app quits and by tests)."""
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)

    def _loop(self):
        while not self._stop.is_set():
            with self.lock:
                job = self.queue[0] if self.queue else None
            if job is None:
                with self.lock:
                    self.waiting_for = ""
                self._wake.wait(self.idle_wait)
                self._wake.clear()
                continue
            prog = "" if job.get("force") else _which_running(_pause_programs())
            with self.lock:
                self.waiting_for = prog
            if prog:
                self._wake.wait(self.pause_wait)
                self._wake.clear()
                continue
            with self.lock:
                if not self.queue or self.queue[0] is not job:
                    continue                        # the line changed meanwhile (cancel / run now)
                self.queue.pop(0)
                self.current, self.info = job, {}
                self._cancel.clear()
                job["started"] = _now()
                self._save()
            self._run(job)

    def _run(self, job):
        kind = _KINDS.get(job["kind"])
        status, msg, paused_for = "done", "", ""
        log.info("job %s started: %s %s/%s", job["id"], job["kind"], job.get("project"), job.get("video"))
        try:
            if kind is None:
                raise JobError("This kind of work is not in this version of BroClips.")
            out = kind["fn"](job, self._progress_fn(job, kind["steps"]), self._cancel.is_set)
            if self._cancel.is_set():
                status, msg = "stopped", "Stopped."
            else:
                msg = out if isinstance(out, str) and out.strip() else "✅ Done"
        except Cancelled:
            status, msg = "stopped", "Stopped."
        except Paused as ex:
            status, paused_for = "paused", (ex.args[0] if ex.args else "a program")
        except (JobError, ProviderError) as ex:
            log.warning("job %s (%s): %s", job["id"], job["kind"], ex)
            status, msg = "failed", "⚠️ " + str(ex)
        except BaseException as ex:                 # never let one job kill the worker
            log.exception("job %s (%s) failed", job["id"], job["kind"])
            status, msg = "failed", _plain(ex)
        _unload_models()
        with self.lock:
            self.current, self.info = None, {}
            if status == "paused":                  # back to the front; it continues after the program closes
                job.pop("started", None)
                self.queue.insert(0, job)
                self.waiting_for = paused_for
                log.info("job %s paused: %s is running", job["id"], paused_for)
            else:
                self.results.append(self._result(job, status, msg))
                self.results = self.results[-KEEP_RESULTS:]
                log.info("job %s %s: %s", job["id"], status, msg)
            self._save()

    def _progress_fn(self, job, steps):
        last = {"check": time.monotonic(), "write": 0.0, "step": None}

        def progress(step, label, detail="", frac=None):
            try:
                step = max(1, min(int(step or 1), steps))
            except (TypeError, ValueError):
                step = 1
            try:
                f = None if frac is None else max(0.0, min(1.0, float(frac)))
            except (TypeError, ValueError):
                f = None
            info = {"job": job["id"], "kind": job["kind"], "step": step, "of": steps, "label": str(label or ""),
                    "detail": str(detail or ""), "frac": f, "pct": int(round(((step - 1) + (f or 0.0)) / steps * 100)),
                    "time": time.time()}
            with self.lock:
                if self.current is job:
                    self.info = info
            now = time.monotonic()
            if job.get("project") and job.get("video") and (step != last["step"] or now - last["write"] >= 1.0):
                last["write"], last["step"] = now, step
                _write_progress(job, info)
            if self._cancel.is_set():
                raise Cancelled()
            if not job.get("force") and now - last["check"] >= self.check_every:
                last["check"] = now
                prog = _which_running(_pause_programs())
                if prog:
                    raise Paused(prog)
        return progress


def _write_progress(job, info):
    """<video folder>/progress.json — only for a video that still exists (never re-creates a removed one)."""
    try:
        from .projects import video_dir
        d = video_dir(job["project"], job["video"])
        if d.is_dir():
            save_json(info, d / "progress.json")
    except (OSError, ValueError):
        pass                                        # a progress note must never stop the real work


LINE = WorkLine()


# ---------- module-level shortcuts to the app's line ----------
def start():
    LINE.start()


def stop(timeout=5.0):
    LINE.stop(timeout)


def enqueue(kind, project=None, video=None, args=None, title="", force=False):
    return LINE.enqueue(kind, project=project, video=video, args=args, title=title, force=force)


def cancel(job_id=None, project=None, video=None):
    return LINE.cancel(job_id=job_id, project=project, video=video)


def run_now(job_id=None):
    return LINE.run_now(job_id)


def state():
    return LINE.state()


def running():
    return LINE.running()


def busy():
    return LINE.busy()


def job_for(project, video=None):
    return LINE.job_for(project, video)


def result_for(project, video=None, kind=None):
    return LINE.result_for(project, video, kind)

"""The video page's results API (SPEC §9 "Video page"): the Shorts (files, versions, keep / skip marks, titles), the
long video + chapters, the texts per platform, the "What the AI is doing" log, and the actions — keep / skip, delete a
Short (Recycle Bin), simple edits (start / end, the title on top, caption words -> the Short is made again), the
wishes box + 🔁 Find new moments, the layout of this video.

  GET  /api/results/<pid>/<vid>                       everything the page shows (polled every 2 s)
  POST /api/results/<pid>/<vid>/mark       {n, mark}  mark = "keep" | "skip" | ""
  POST /api/results/<pid>/<vid>/delete     {n}        the Short goes to the Recycle Bin
  POST /api/results/<pid>/<vid>/edit       {n, start?, end?, hook?, captions?: [{i, text}]}   -> short_edit job
  POST /api/results/<pid>/<vid>/wishes     {text}
  POST /api/results/<pid>/<vid>/find_again {text?}    saves the wishes, then -> find_again job
  POST /api/results/<pid>/<vid>/layout     {layout, remake}                                    -> short_edit "all"
Files are served by the server's /files/<pid>/<vid>/... route (with ?v=<version> so a new version plays)."""
import json
from pathlib import Path

from . import config, jobs, marathon, projects, shorts, transcript
from .server import ApiError, route
from .util import load_json

LOG_LINES = 400
MAX_SHORT_S = 180


def _ids(rest, want_action=False):
    parts = [p for p in str(rest or "").split("/") if p]
    if len(parts) != (3 if want_action else 2) or not (projects.valid_id(parts[0]) and projects.valid_id(parts[1])):
        raise ApiError("Not found.", 404)
    pid, vid = parts[0], parts[1]
    if not (projects.video_dir(pid, vid) / "video.json").is_file():
        raise ApiError("This video is not in the project (any more).", 404)
    return parts


def read_log(path, limit=LOG_LINES):
    """The last `limit` notes of the AI log (a half-written last line is skipped)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    out = []
    for line in text.splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _is_cloud():
    s = config.settings()["thinking"]
    name = s.get("provider")
    if name in ("gemini", "anthropic"):
        return True
    if name == "openai":
        try:
            from .providers._http import is_local_url
            return not is_local_url((s.get("openai") or {}).get("base_url"))
        except Exception:
            return True
    return False


def payload(pid, vid):
    d = projects.video_dir(pid, vid)
    meta = load_json(d / "video.json", {}) or {}
    proj = projects.get(pid) or {}
    base = f"/files/{pid}/{vid}/"
    out_shorts = []
    for s in shorts.listing(d):
        item = {k: s.get(k) for k in ("n", "title", "hook", "why", "type", "score", "marathon", "mark", "version",
                                      "dirty", "dur", "layout", "s", "e", "hinted", "src", "phrases", "rev")}
        item["segments"] = s.get("segments") or []
        item["file"] = f"{base}shorts/short_{s['n']:02d}.mp4?v={s.get('mtime') or 0}" if s.get("has_file") else None
        item["at"] = transcript.mmss(s.get("s"))
        out_shorts.append(item)
    long = None
    if (d / "long.mp4").is_file():
        li = load_json(d / "long.json", {}) or {}
        long = {k: li.get(k) for k in ("dur", "src_dur", "jump_cuts", "silence_s", "cut_s")}
        long["file"] = f"{base}long.mp4?v={int((d / 'long.mp4').stat().st_mtime)}"
    texts = load_json(d / "texts.json")
    if long is not None:
        long["chapters"] = ((texts or {}).get("long") or {}).get("chapters") or []
    dur = float(meta.get("dur") or (load_json(d / "info.json", {}) or {}).get("dur") or 0)
    return {"ok": True, "shorts": out_shorts, "long": long, "texts": texts, "log": read_log(d / "ai_log.jsonl"),
            "wishes": meta.get("wishes") or "", "layout": meta.get("layout") or "",
            "layout_now": shorts.layout_for(proj, meta), "stages": load_json(d / "stages.json", {}) or {},
            "has_transcript": (d / "transcript.json").is_file(),
            "marathon": {"on": bool(proj.get("marathon")), "questions": marathon.estimate(dur) if dur else 0,
                         "cloud": _is_cloud()}}


@route("GET", "/api/results/", prefix=True)
def api_results(req):
    pid, vid = _ids(req.rest)
    return payload(pid, vid)


def _busy_with(pid, vid, kinds=("make", "find_again", "short_edit")):
    j = jobs.job_for(pid, vid)
    return bool(j and j["status"] == "working" and (j.get("job") or {}).get("kind") in kinds)


def _short(d, b):
    try:
        n = int(b.get("n"))
    except (TypeError, ValueError):
        raise ApiError("Which Short?")
    data = shorts.load(d, n)
    if not isinstance(data, dict):
        raise ApiError("That Short is not there any more.", 404)
    return n, data


def _num(v, name):
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ApiError(f"Bad {name}.")


@route("POST", "/api/results/", prefix=True)
def api_results_post(req):
    pid, vid, action = _ids(req.rest, want_action=True)
    d = projects.video_dir(pid, vid)
    b = req.body
    if action == "mark":
        n, _ = _short(d, b)
        mark = b.get("mark") if b.get("mark") in ("keep", "skip") else ""
        with shorts.LOCK:
            data = shorts.load(d, n) or {}
            data["mark"] = mark
            shorts.save(d, n, data)
        return {"ok": True, "mark": mark}
    if action == "delete":
        n, _ = _short(d, b)
        if _busy_with(pid, vid):
            raise ApiError("BroClips is working on this video right now. Wait until it is done, then delete.", 409)
        files = [p for p in shorts.paths(d, n) if p.exists()]
        if not projects.to_recycle_bin(files):
            raise ApiError("Could not delete it — is the Short open in another program? Close it and try again.", 409)
        return {"ok": True, "msg": f"Short {n} is in the Recycle Bin."}
    if action == "edit":
        return _edit(pid, vid, d, b)
    if action in ("wishes", "find_again"):
        if "text" in b:
            projects.set_video(pid, vid, {"wishes": str(b.get("text") or "")[:3000]})
        if action == "wishes":
            return {"ok": True}
        if not (d / "transcript.json").is_file():
            raise ApiError("Make this video first (✨ Make). Then you can look for new moments.", 409)
        j = jobs.job_for(pid, vid)
        if j and (j.get("job") or {}).get("kind") in ("make", "find_again"):
            raise ApiError("This video is already in the work line. Wait until it is done.", 409)
        meta = load_json(d / "video.json", {}) or {}
        src = Path(str(meta.get("src") or ""))
        if not src.is_file():
            raise ApiError(f"The video file is not there any more: {src}", 409)
        job = jobs.enqueue("find_again", project=pid, video=vid, title=meta.get("name") or vid)
        return {"ok": True, "job": job}
    if action == "layout":
        layout = b.get("layout")
        if layout not in ("", "auto") + shorts.LAYOUTS:
            raise ApiError("Unknown layout.")
        projects.set_video(pid, vid, {"layout": "" if layout == "auto" else layout})
        if not b.get("remake"):
            return {"ok": True}
        meta = load_json(d / "video.json", {}) or {}
        now_layout = shorts.layout_for(projects.get(pid), meta)
        n_done = 0
        with shorts.LOCK:
            for s in shorts.listing(d):
                data = shorts.load(d, s["n"]) or {}
                data.update(layout=now_layout, dirty=True, rev=int(data.get("rev") or 0) + 1)
                shorts.save(d, s["n"], data)
                n_done += 1
        if not n_done:
            return {"ok": True, "msg": "Saved. It is used when the Shorts are made."}
        job = jobs.enqueue("short_edit", project=pid, video=vid, args={"n": "all", "rev": jobs._now()},
                           title=(meta.get("name") or vid) + " — all Shorts")
        return {"ok": True, "job": job}
    raise ApiError("Not found.", 404)


def _edit(pid, vid, d, b):
    """Start / end ± (absolute new values in source seconds), the title on top, caption words -> make it again."""
    n, data = _short(d, b)
    meta = load_json(d / "video.json", {}) or {}
    if not Path(str(meta.get("src") or "")).is_file():
        raise ApiError("The video file is not there any more, so the Short cannot be made again.", 409)
    total = float(meta.get("dur") or (load_json(d / "info.json", {}) or {}).get("dur") or 1e9)
    segs = shorts.normalize(shorts.of(data))
    changed = []
    if "start" in b or "end" in b:
        start = _num(b.get("start", segs[0][0]), "start")
        end = _num(b.get("end", segs[-1][1]), "end")
        start, end = max(0.0, round(start, 2)), min(total, round(end, 2))
        if end - start < 3:
            raise ApiError("A Short must be at least 3 seconds long.")
        if len(segs) == 1:
            segs = [[start, end, segs[0][2]]]
        else:
            if start >= segs[0][1] - 1 or end <= segs[-1][0] + 1:
                raise ApiError("That would cut away a whole part of this Short.")
            segs[0][0], segs[-1][1] = start, end
        if abs(segs[0][0] - data["segments"][0][0]) > 0.01 or abs(segs[-1][1] - data["segments"][-1][1]) > 0.01:
            changed.append("start / end")
        if shorts.out_len(segs) > MAX_SHORT_S:
            raise ApiError("A Short can be at most 3 minutes long.")
    hook = data.get("hook") or ""
    if "hook" in b:
        new = " ".join(str(b.get("hook") or "").split())[:120]
        if new != hook:
            hook = new
            changed.append("title")
    fixes = list(data.get("caption_fixes") or [])
    phr = data.get("phrases") or []
    for c in b.get("captions") or []:
        try:
            i = int(c.get("i"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not 0 <= i < len(phr):
            continue
        text = " ".join(str(c.get("text") or "").split())[:200]
        if text == phr[i].get("text"):
            continue
        fixes = [f for f in fixes if not (abs(f["a"] - phr[i]["a"]) < 0.01 and abs(f["b"] - phr[i]["b"]) < 0.01)]
        fixes.append({"a": phr[i]["a"], "b": phr[i]["b"], "text": text})
        if "captions" not in changed:
            changed.append("captions")
    if not changed:
        return {"ok": True, "msg": "Nothing was changed."}
    with shorts.LOCK:
        cur = shorts.load(d, n) or data
        cur.update(segments=[[round(a, 3), round(x, 3), sp] for a, x, sp in segs], s=round(segs[0][0], 3),
                   e=round(segs[-1][1], 3), hook=hook, caption_fixes=fixes, dirty=True,
                   rev=int(cur.get("rev") or 0) + 1)
        shorts.save(d, n, cur)
        rev = cur["rev"]
    job = jobs.enqueue("short_edit", project=pid, video=vid, args={"n": n, "rev": rev},
                       title=f"{meta.get('name') or vid} — Short {n}")
    return {"ok": True, "job": job, "changed": changed}

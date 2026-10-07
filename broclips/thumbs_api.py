"""The 🎨 Thumbnail studio's API (SPEC §8, §9). The page (web/studio.html) only SHOWS pictures; thumbs.py draws every
one of them, so what the user sees is exactly the file they upload. server.py imports this module by itself
(OPTIONAL_MODULES) - it adds its routes to the registry and the `cutout` / `thumbai` job kinds (via thumbs.py).

GET  /api/thumbs/list?project=&video=      {"thumbs": [{"n", "file", "v"}]}   (file = thumbs/thumb_N.jpg, relative to
                                           the video folder: /files/<pid>/<vid>/<file>?v=<v>; the video page uses this)
GET  /api/thumbs/info?project=&video=      everything the studio needs to start
GET  /api/thumbs/aistate?project=&video=&job=   ✨ AI job: working / queued / waiting / done (+ message, numbers made)
GET  /api/thumbs/cut?project=&video=&id=   a subject's cut-out: ready / working / queued / waiting / no-model / error
GET  /api/thumbs/strip?project=&video=&i=  12 seconds spread over moment i
GET  /api/thumbs/tile?project=&video=&t=   a small picture of the video (320 wide)
GET  /api/thumbs/subject?id=               a small preview of a subject (cut-out, or the uploaded picture)
GET  /api/thumbs/emoji/<file>.png · /api/thumbs/font/<file>   the 3D emoji and fonts (for the page's buttons)
POST /api/thumbs/open {project, video, n|null}           a design to edit (null = a new one)
POST /api/thumbs/render {project, video, design}         JPEG + header X-Boxes (where each thing is, in pixels)
POST /api/thumbs/layout {project, video, design, preset?}   a style applied (or, without preset, a new picture
                                                         followed: zoom / circle / arrow go to its action)
POST /api/thumbs/save {project, video, n|null, design}   · /delete {.., n} · /delete_many {.., ns}  (Recycle Bin)
POST /api/thumbs/auto · /more {.., count} · /ai {.., prompt, count} · /brief {.., text}
POST /api/thumbs/subject/frame {.., t, slot} · /subject/upload {.., name, data (base64), slot} · /subject/use {.., id,
     slot}
Everything that writes thumbnails is refused while "Make" works on (or waits for) that video."""
import base64
import binascii
import json
import re

from . import projects, thumbs
from .server import ApiError, bytes_response, file_response, route
from .util import log, mmss


def _pv(src):
    pid, vid = str(src.get("project") or ""), str(src.get("video") or "")
    if not (projects.valid_id(pid) and projects.valid_id(vid)) or \
            not (projects.video_dir(pid, vid) / "video.json").is_file():
        raise ApiError("This video is not in the project (any more).", 404)
    return pid, vid


def _ctx(src):
    pid, vid = _pv(src)
    try:
        return thumbs.context(pid, vid)
    except thumbs.ThumbError as ex:
        raise ApiError(str(ex), 409)


def _not_making(pid, vid):
    if thumbs.making(pid, vid):
        raise ApiError(thumbs.MAKING, 409)


def _plain(ex):
    if isinstance(ex, thumbs.ThumbError):
        return str(ex)
    log.warning("thumbnail studio: %s: %s", type(ex).__name__, ex)
    return f"Could not do that ({type(ex).__name__}: {str(ex)[-200:]})."


def _list(pid, vid):
    return thumbs.thumb_list(pid, vid)


# ---------- GET ----------
@route("GET", "/api/thumbs/list")
def api_list(req):
    pid, vid = _pv(req.query)
    return {"thumbs": _list(pid, vid)}


@route("GET", "/api/thumbs/info")
def api_info(req):
    ctx = _ctx(req.query)
    pid, vid = ctx.pid, ctx.vid
    moments = thumbs._moments(ctx)
    used = []
    for f in thumbs.thumb_files(pid, vid):
        try:
            used.append(float(json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))["bg"]["t"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    subs, main, second = thumbs.subjects(pid, vid)
    fonts = [{"key": k, "name": v[1], "file": v[0], "script": v[4], "help": thumbs.FONT_HELP.get(k, "")}
             for k, v in thumbs.FONTS.items() if thumbs._font_path(k)]
    return {"project": pid, "video": vid, "name": ctx.name, "title": thumbs.video_title(ctx), "dur": mmss(ctx.dur),
            "dur_s": ctx.dur, "key": ctx.key, "thumbs": _list(pid, vid), "presets": thumbs.PRESETS,
            "style_help": thumbs.STYLE_HELP, "fonts": fonts, "fonts_missing": len(fonts) < len(thumbs.FONTS),
            "styles": [{"key": k, "name": v["name"], "fill": v["fill"], "box": v.get("box")}
                       for k, v in thumbs.STYLES.items()],
            "glows": thumbs.GLOWS,
            "emoji": [{"e": e, "file": thumbs.EMOJI[e] + ".png"} for e in thumbs.EMOJI_ORDER if thumbs.emoji_file(e)],
            "texts": thumbs.suggestions(ctx, moments), "cands": thumbs.candidates(ctx, moments, used),
            "moments": [{"i": i, "title": thumbs.split_emoji(str(m.get("title") or m.get("hook") or ""))[0]
                         or f"Moment {i}"} for i, m in enumerate(moments, 1)],
            "raqm": thumbs.RAQM, "cut_model": bool(thumbs.cutout_model()), "brief": thumbs.brief(pid, vid),
            "subjects": subs, "main": main, "second": second, "video_type": ctx.video_type,
            "lang": thumbs.content_language(ctx), "making": thumbs.making(pid, vid)}


@route("GET", "/api/thumbs/aistate")
def api_aistate(req):
    pid, vid = _pv(req.query)
    return dict(thumbs.ai_state(pid, vid, req.query.get("job") or None), thumbs=_list(pid, vid))


@route("GET", "/api/thumbs/cut")
def api_cut(req):
    _pv(req.query)
    state, extra = thumbs.cut_state(req.query.get("id"))
    out = {"state": state}
    if state in ("error", "no-model"):
        out["msg"] = extra
    elif state == "waiting":
        out["waiting_for"] = extra
    elif state in ("working", "queued"):
        out["job"] = extra
    return out


@route("GET", "/api/thumbs/strip")
def api_strip(req):
    ctx = _ctx(req.query)
    moments = thumbs._moments(ctx)
    try:
        i = int(req.query.get("i") or 0)
    except ValueError:
        i = 0
    return {"times": thumbs.strip(ctx, moments[i - 1]) if 0 < i <= len(moments) else []}


@route("GET", "/api/thumbs/tile")
def api_tile(req):
    ctx = _ctx(req.query)
    try:
        f = thumbs._tile(ctx, float(req.query.get("t") or 0))
    except (ValueError, thumbs.ThumbError) as ex:
        raise ApiError(str(ex), 404)
    return bytes_response(f.read_bytes(), "image/jpeg", headers={"Cache-Control": "max-age=86400"})


@route("GET", "/api/thumbs/subject")
def api_subject_preview(req):
    sid = str(req.query.get("id") or "")
    if not thumbs.SID.fullmatch(sid):
        raise ApiError("Not found.", 404)
    small = thumbs.cut_dir() / "small" / f"{sid}.png"
    src = thumbs.cut_file(sid)
    if not src.exists():
        src = thumbs._upload_src(sid)
        small = small.with_name(f"{sid}.src.png")
    if not src.exists():
        raise ApiError("Not found.", 404)
    if not small.exists() or small.stat().st_mtime < src.stat().st_mtime:
        im = thumbs.Image.open(src)
        im.thumbnail((220, 220))
        small.parent.mkdir(parents=True, exist_ok=True)
        tmp = small.with_name(small.stem + ".part.png")
        im.save(tmp)
        tmp.replace(small)
    return bytes_response(small.read_bytes(), "image/png", headers={"Cache-Control": "max-age=600"})


@route("GET", "/api/thumbs/emoji/", prefix=True)
def api_emoji(req):
    name = req.rest
    if not re.fullmatch(r"[a-z0-9_]+\.png", name) or name[:-4] not in thumbs.EMOJI.values():
        raise ApiError("Not found.", 404)
    f = thumbs.assets.emoji_dir() / name
    if not f.is_file():
        raise ApiError("Not found.", 404)
    r = file_response(f, req)
    r.headers["Cache-Control"] = "max-age=86400"
    return r


@route("GET", "/api/thumbs/font/", prefix=True)
def api_font(req):
    name = req.rest
    if name not in {v[0] for v in thumbs.FONTS.values()}:
        raise ApiError("Not found.", 404)
    f = thumbs.assets.fonts_dir() / name
    if not f.is_file():
        raise ApiError("Not found.", 404)
    r = file_response(f, req)
    r.headers["Cache-Control"] = "max-age=86400"
    return r


# ---------- POST ----------
@route("POST", "/api/thumbs/open")
def api_open(req):
    ctx = _ctx(req.body)
    n = req.body.get("n")
    try:
        n = int(n) if n not in (None, "", "new") else None
        d, note = thumbs.open_design(ctx, n)
    except (ValueError, TypeError):
        raise ApiError("Unknown thumbnail.")
    except Exception as ex:
        raise ApiError(_plain(ex))
    return {"design": d, "note": note, "cut": _cut_states(d)}


def _cut_states(d):
    out = {}
    for k in ("subject", "subject2"):
        c = d.get(k)
        if c:
            out[c["id"]] = thumbs.cut_state(c["id"])[0]
    return out


@route("POST", "/api/thumbs/render")
def api_render(req):
    ctx = _ctx(req.body)
    try:
        img, boxes = thumbs.draw(req.body.get("design"), ctx)
    except Exception as ex:
        raise ApiError(_plain(ex))
    return bytes_response(thumbs.jpeg(img, 85), "image/jpeg", headers={"X-Boxes": json.dumps(boxes)})


@route("POST", "/api/thumbs/layout")
def api_layout(req):
    ctx = _ctx(req.body)
    d = thumbs.normal(req.body.get("design"))
    want = req.body.get("preset")
    try:
        act = thumbs.action_at(ctx, d["bg"]["t"])
        if not want:
            return {"design": thumbs.follow(d, act, ctx.fa, ctx.avoid)}
        out = thumbs.layout(d, want, act, ctx.fa, ctx.avoid)
    except Exception as ex:
        raise ApiError(_plain(ex))
    note = ""
    if out["preset"] != want and want in dict(thumbs.PRESETS):
        name = dict(thumbs.PRESETS)[out["preset"]]
        if want == "split":
            note = "⚔️ VS needs two subjects (part 4: Subject 1 and Subject 2)."
        else:
            busy = d["subject"] and not thumbs.ready(d["subject"]["id"])
            note = "🧍 Subject needs a subject (part 4)" + (" — it is still being cut out." if busy else ".")
        note += f" Used {name} for now."
    return {"design": out, "note": note, "cut": _cut_states(out)}


@route("POST", "/api/thumbs/save")
def api_save(req):
    ctx = _ctx(req.body)
    _not_making(ctx.pid, ctx.vid)
    n = req.body.get("n")
    try:
        n = thumbs.save(ctx, int(n) if n not in (None, "", "new") else None, req.body.get("design"))
    except (ValueError, TypeError):
        raise ApiError("Unknown thumbnail.")
    except Exception as ex:
        raise ApiError(_plain(ex))
    log.info("thumbnail studio: %s/%s thumb_%d saved", ctx.pid, ctx.vid, n)
    return {"ok": True, "n": n, "thumbs": _list(ctx.pid, ctx.vid)}


def _numbers(v):
    out = []
    for x in v if isinstance(v, list) else [v]:
        try:
            n = int(x)
        except (TypeError, ValueError):
            continue
        if 0 < n < 100000:
            out.append(n)
    return out


@route("POST", "/api/thumbs/delete")
def api_delete(req):
    return _delete(req, _numbers(req.body.get("n")))


@route("POST", "/api/thumbs/delete_many")
def api_delete_many(req):
    return _delete(req, _numbers(req.body.get("ns")))


def _delete(req, ns):
    pid, vid = _pv(req.body)
    _not_making(pid, vid)
    if not ns:
        raise ApiError("Say which thumbnails to delete.")
    ok = all(thumbs.delete(pid, vid, n) for n in ns)
    return {"ok": ok, "error": "" if ok else "Some could not be deleted (is one of them open in another program?).",
            "thumbs": _list(pid, vid)}


@route("POST", "/api/thumbs/auto")
def api_auto(req):
    """✨ 3 new automatic ones; the ones there are stay."""
    ctx = _ctx(req.body)
    _not_making(ctx.pid, ctx.vid)
    first = thumbs.next_n(ctx.pid, ctx.vid)
    made = thumbs.auto(ctx.pid, ctx.vid, first_n=first, wait_cut=False)
    log.info("thumbnail studio: %s/%s - %d new automatic thumbnails from #%d", ctx.pid, ctx.vid, len(made), first)
    return {"ok": bool(made), "first": first, "made": len(made), "thumbs": _list(ctx.pid, ctx.vid),
            "error": "" if made else "No picture could be made from this video."}


@route("POST", "/api/thumbs/more")
def api_more(req):
    """🎲 Random mix: instant, as often as wanted."""
    ctx = _ctx(req.body)
    _not_making(ctx.pid, ctx.vid)
    try:
        count = int(req.body.get("count") or 6)
    except (TypeError, ValueError):
        count = 6
    made = thumbs.more(ctx, count)
    return {"ok": bool(made), "made": made, "thumbs": _list(ctx.pid, ctx.vid),
            "error": "" if made else "No new combinations left for this video — try ✨ Make with AI."}


@route("POST", "/api/thumbs/ai")
def api_ai(req):
    """✨ Make with AI: a `thumbai` job in the work line (one heavy thing at a time; it waits for the programs in
    Settings' "pause while" list)."""
    ctx = _ctx(req.body)
    _not_making(ctx.pid, ctx.vid)
    raw = str(req.body.get("count") or "").strip()
    count = int(raw) if raw.isdigit() else None
    job, n = thumbs.start_ai(ctx.pid, ctx.vid, req.body.get("prompt") or "", count, title=f"✨ {ctx.name}"[:80])
    return {"ok": True, "job": job, "count": n}


@route("POST", "/api/thumbs/brief")
def api_brief(req):
    pid, vid = _pv(req.body)
    thumbs.set_brief(pid, vid, req.body.get("text") or "")
    return {"ok": True}


def _slot(b):
    return 2 if str(b.get("slot")) == "2" else 1


@route("POST", "/api/thumbs/subject/frame")
def api_subject_frame(req):
    """✂️ Cut out from this picture: a `cutout` job (the studio polls /api/thumbs/cut)."""
    ctx = _ctx(req.body)
    try:
        t = round(min(max(float(req.body.get("t") or 0), 0.0), ctx.dur), 1)
    except (TypeError, ValueError):
        raise ApiError("Unknown picture.")
    sid = thumbs.frame_sid(ctx, t)
    thumbs.remember_subject(ctx.pid, ctx.vid, sid, how="frame", slot=_slot(req.body), t=t)
    if thumbs.ready(sid):
        return {"ok": True, "id": sid, "state": "ready"}
    if not thumbs.cutout_model():
        return {"ok": False, "id": sid, "state": "no-model", "error": thumbs.NO_MODEL}
    job = thumbs.start_cut(ctx.pid, ctx.vid, sid, t, title=f"✂️ {ctx.name}"[:80])
    return {"ok": True, "id": sid, "state": thumbs.cut_state(sid)[0], "job": job}


@route("POST", "/api/thumbs/subject/upload")
def api_subject_upload(req):
    """⬆️ Upload an image (sent as base64 inside JSON - the server only takes JSON). A PNG with transparency is
    ready at once; any other picture gets a `cutout` job."""
    ctx = _ctx(req.body)
    raw = str(req.body.get("data") or "")
    raw = raw.split(",", 1)[1] if raw.startswith("data:") and "," in raw else raw
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        raise ApiError("That file could not be read. Try another picture.")
    name = str(req.body.get("name") or "")[:120]
    try:
        sid, done = thumbs.add_upload(data, name)
    except thumbs.ThumbError as ex:
        raise ApiError(str(ex))
    thumbs.remember_subject(ctx.pid, ctx.vid, sid, how="upload", slot=_slot(req.body), name=name)
    if done:
        return {"ok": True, "id": sid, "state": "ready"}
    if not thumbs.cutout_model():
        return {"ok": False, "id": sid, "state": "no-model", "error": thumbs.NO_MODEL}
    job = thumbs.start_cut(ctx.pid, ctx.vid, sid, title=f"✂️ {ctx.name}"[:80])
    return {"ok": True, "id": sid, "state": thumbs.cut_state(sid)[0], "job": job}


@route("POST", "/api/thumbs/subject/use")
def api_subject_use(req):
    """A subject picked from the list becomes this video's subject 1 (or 2): random mix and AI use it too."""
    pid, vid = _pv(req.body)
    sid = str(req.body.get("id") or "")
    if not thumbs.SID.fullmatch(sid):
        raise ApiError("Unknown picture.")
    thumbs.remember_subject(pid, vid, sid, slot=_slot(req.body))
    return {"ok": True, "state": thumbs.restart_cut(pid, vid, sid, title="✂️ Cut-out")}

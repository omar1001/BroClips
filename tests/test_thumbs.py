"""Thumbnails (SPEC §8): design cleaning, styles and free-space placement, text wrapping, the count / style words of a
request, subject cut-side fades, uploads, and the studio API on a tiny generated video (no network, no GPU, no AI
model: the cut-out model is absent, the AI is the fake one)."""
import hashlib
import http.client
import io
import json
import math
import shutil
import subprocess
import threading

import numpy as np
import pytest
from PIL import Image

from broclips import config, jobs, projects, server, thumbs

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
FAKE_EMOJI = ["😂", "😱", "🤯", "😳", "🔥", "❌", "👀"]


@pytest.fixture(autouse=True)
def fresh_caches():
    for c in (thumbs._LAYERS, thumbs._FONTS, thumbs._LINES, thumbs._CUT_ERR, thumbs._AI_LAST):
        c.clear()
    yield


@pytest.fixture
def emoji_files():
    """Small stand-in emoji pictures (the real Fluent ones are downloaded by assets.py)."""
    d = thumbs.assets.emoji_dir()
    d.mkdir(parents=True, exist_ok=True)
    for e in FAKE_EMOJI:
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        im.paste((255, 200, 0, 255), (8, 8, 56, 56))
        im.save(d / f"{thumbs.EMOJI[e]}.png")
    return d


def subject(sid="up-testsubject01", bottom=True, size=(300, 400)):
    """A synthetic cut-out: an orange shape that touches the bottom edge (a person cut off by the picture) or not."""
    w, h = size
    a = np.zeros((h, w), np.uint8)
    a[40:h if bottom else h - 40, 60:w - 60] = 255
    rgb = Image.new("RGB", (w, h), (255, 120, 0))
    thumbs._save_cut(sid, rgb, a, "upload")
    return sid


# ---------------------------------------------------------------- designs
def test_normal_cleans_a_design(emoji_files):
    d = thumbs.normal({"preset": "champ", "bg": {"t": "12.34", "z": 99, "cx": -3, "punch": 7, "shade": "x"},
                       "text": {"s": "Hello 🔥", "size": 5000, "style": "nope", "rot": 90, "w": 0},
                       "text2": {"s": ""}, "subject": {"id": "../../etc/passwd"}, "subject2": {"id": "up-abcd1234"},
                       "stickers": [{"k": "emoji", "e": "😂"}, {"k": "emoji", "e": "🦄"}, {"k": "bomb"},
                                    {"k": "arrow", "color": "red"}], "evil": 1})
    assert d["preset"] == "zoom" and "evil" not in d
    assert d["bg"]["t"] == 12.34 and d["bg"]["z"] == 4.0 and d["bg"]["cx"] == 0 and d["bg"]["punch"] == 2
    assert d["bg"]["shade"] is None
    t = d["text"]
    assert t["s"] == "Hello" and t["size"] == 320 and t["style"] == "yellow" and t["rot"] == 30 and t["w"] == 0.15
    assert t["font"] == "montserrat"
    assert d["text2"] is None and d["subject"] is None
    assert d["subject2"]["id"] == "up-abcd1234" and d["subject2"]["line"] == "#ffffff" and d["subject2"]["on"] is True
    assert [s["k"] for s in d["stickers"]] == ["emoji", "arrow"] and d["stickers"][1]["color"] is None
    assert thumbs.normal(None)["text"]["s"] == "" and thumbs.normal("junk")["stickers"] == []
    # Arabic words never get a Latin-only font (it has no Arabic letters)
    ar = thumbs.normal({"text": {"s": "دمجي مبيأثرش", "font": "anton"}})
    assert ar["text"]["font"] == "lalezar"
    assert thumbs.normal({"text": {"s": "دمجي", "font": "cairo"}})["text"]["font"] == "cairo"
    assert thumbs.normal({"text": {"s": "Hi", "font": "jomhuria"}})["text"]["font"] == "jomhuria"


def test_split_emoji_and_scripts():
    assert thumbs.split_emoji("Never do this! ❌") == ("Never do this!", ["❌"])
    assert thumbs.split_emoji("دمجي {LOUD} مبيأثرش 😂") == ("دمجي مبيأثرش", ["😂"])
    assert thumbs.is_arabic("hello يا") and not thumbs.is_arabic("hello")
    assert thumbs.default_font("لعبة") == "lalezar" and thumbs.default_font("Game") == "montserrat"


def test_wrap_splits_a_long_phrase_into_two_balanced_lines():
    t = thumbs.normal({"text": {"s": "for loops explained in two minutes", "w": 0.5, "size": 150}})["text"]
    lines = thumbs._wrap(t)
    assert len(lines) == 2 and " ".join(lines) == t["s"]
    f = thumbs._font(t["font"], thumbs._size_of(t))
    a, b = (f.getlength(x) for x in lines)
    assert max(a, b) / min(a, b) < 2.2                               # about equal widths
    short = thumbs.normal({"text": {"s": "WOW", "w": 0.9}})["text"]
    assert thumbs._wrap(short) == ["WOW"]
    own = thumbs.normal({"text": {"s": "one\ntwo\nthree\nfour"}})["text"]
    assert thumbs._wrap(own) == ["one", "two", "three"]               # the user's own breaks, at most 3
    lay = thumbs._text_layer(thumbs.normal({"text": {"s": "BIG WORDS", "w": 0.6}})["text"])
    assert lay.width <= 0.6 * thumbs.W + 4 and lay.getbbox()          # shrunk to its width, something drawn


def test_window_and_aim_steer_around_avoid_boxes():
    bg = {"z": 1.0, "cx": 0.5, "cy": 0.5}
    assert thumbs._window(bg, 9 / 16)[2:] == pytest.approx((1.0, (9 / 16) / (16 / 9)))     # a phone video: a band
    assert thumbs._window(bg)[2:] == pytest.approx((1.0, 1.0))
    avoid = [(0.6, 0.0, 1.0, 0.4)]                                     # e.g. a watermark top right
    free, steered = {"z": 1.8, "cx": 0.5, "cy": 0.5}, {"z": 1.8, "cx": 0.5, "cy": 0.5}
    thumbs._aim(free, 0.62, 0.42, 0.5, 0.5)
    thumbs._aim(steered, 0.62, 0.42, 0.5, 0.5, avoid=avoid)

    def cover(b):
        x0, y0, ww, wh = thumbs._window(b)
        return max(0, min(x0 + ww, 1) - max(x0, 0.6)) * max(0, min(y0 + wh, 0.4) - max(y0, 0))
    assert cover(steered) < cover(free)
    u, v = thumbs._where(steered, 0.62, 0.42)
    assert 0.12 <= u <= 0.88 and 0.12 <= v <= 0.88                   # the action stays well inside


def test_spot_and_free_arrow_keep_away_from_boxes():
    box = (0, 0, 700, 400)                                             # pixels
    x, y = thumbs._spot([(0.1, 0.1), (0.9, 0.9), (0.2, 0.3)], [box], 50)
    assert (x, y) == (0.9, 0.9)
    a = thumbs._free_arrow(0.5, 0.5, [(0, 0, 1280, 300)])              # the whole top band is taken
    assert a and a["y"] * thumbs.H > 300 - 0.26 * thumbs.H * 0.5
    assert thumbs._rect_dist(10, 10, (0, 0, 20, 20)) == 0


@pytest.mark.parametrize("preset", ["zoom", "reaction", "label", "headline"])
def test_layout_keeps_words_emoji_and_arrow_apart(emoji_files, preset):
    d = thumbs.new_design(30.0, "This loop never ends 😱")
    out = thumbs.layout(d, preset, (0.62, 0.40, 2500))
    assert out["preset"] == preset
    tb = thumbs._text_box(out["text"])
    assert tb[1] >= 0 and tb[3] <= thumbs.H                           # the words stay inside the picture
    for s in out["stickers"]:
        if s["k"] == "emoji" and preset != "reaction":
            assert thumbs._rect_dist(s["x"] * thumbs.W, s["y"] * thumbs.H, tb) > 0
        if s["k"] == "arrow":
            assert thumbs._rect_dist(s["x"] * thumbs.W, s["y"] * thumbs.H, tb) > 0
    if preset == "reaction":
        face = next(s for s in out["stickers"] if s["k"] == "emoji")
        assert face["e"] in thumbs.FACES and face["s"] > 0.5
        assert not any(s["k"] == "circle" for s in out["stickers"])
    if preset == "label":
        assert out["text"]["style"] in thumbs.BOX_STYLES
    else:
        assert out["text"]["style"] not in thumbs.BOX_STYLES
    if preset == "headline":
        assert out["bg"]["blur"] > 0 and out["bg"]["dim"] > 0 and out["text"]["size"] >= 230


def test_calm_pictures_get_no_circle_and_less_zoom(emoji_files):
    d = thumbs.new_design(5.0, "Calm")
    strong = thumbs.layout(d, "zoom", (0.5, 0.5, 3000))
    calm = thumbs.layout(d, "zoom", (0.5, 0.5, 100))
    assert any(s["k"] == "circle" for s in strong["stickers"])
    assert not any(s["k"] in ("circle", "arrow") for s in calm["stickers"])
    assert calm["bg"]["z"] < strong["bg"]["z"]


def test_subject_styles_fall_back_without_a_ready_subject(emoji_files):
    d = thumbs.new_design(5.0, "Me vs you", subject="up-notcutyet0001")
    assert thumbs.layout(d, "subject", (0.5, 0.5, 3000))["preset"] == "reaction"
    sid = subject()
    d = thumbs.new_design(5.0, "Me vs you", subject=sid)
    out = thumbs.layout(d, "split", (0.5, 0.5, 3000))
    assert out["preset"] == "subject" and out["subject"]["on"] is True
    assert out["subject"]["y"] + out["subject"]["h"] / 2 == pytest.approx(1.04, abs=0.01)   # stands on the bottom
    sid2 = subject("up-testsubject02", bottom=False, size=(400, 200))
    d = thumbs.new_design(5.0, "Me vs you", subject=sid, subject2=sid2)
    out = thumbs.layout(d, "split", (0.5, 0.5, 3000))
    assert out["preset"] == "split" and out["text2"]["s"] == "VS"
    assert out["subject"]["x"] < 0.5 < out["subject2"]["x"]
    assert out["subject2"]["flip"] is False                           # an uploaded picture is never mirrored
    assert out["subject2"]["h"] * thumbs.H * 2 <= 0.45 * thumbs.W + 1    # a wide one is fitted to its half
    hidden = thumbs.layout(out, "zoom", (0.5, 0.5, 3000))
    assert hidden["subject"]["on"] is False and hidden["subject"]["id"] == sid      # kept, only hidden


def test_cut_sides_fade_the_subject_and_its_outline():
    sid = subject()
    meta = thumbs.cut_meta(sid)
    assert meta["sides"] == ["bottom"] and meta["size"] == [180, 360]
    c = thumbs._subject_spec({"id": sid, "h": 0.8, "line": "#ffffff", "glow": "#38c8ff"})
    lay = np.asarray(thumbs._subject_layer(c).getchannel("A"), np.float32)
    th = int(0.8 * thumbs.H)
    pad = (lay.shape[0] - th) // 2
    mid = lay[pad + th // 2, :].max()
    bottom = lay[pad + th - 2, :].max()
    below = lay[pad + th + 2:, :].max()                               # the outline / glow under the cut edge
    assert mid == 255 and bottom < 60 and below < 60
    whole = subject("up-testsubject03", bottom=False)
    assert thumbs.cut_meta(whole)["sides"] == []
    lay2 = np.asarray(thumbs._subject_layer(dict(c, id=whole)).getchannel("A"), np.float32)
    assert lay2[lay2.shape[0] // 2 + 20, :].max() == 255
    assert thumbs.fade_mask(10, 10, ["left"])[:, 0].max() == 0 and thumbs.fade_mask(10, 10, [])[5, 5] == 1


def test_uploads_png_with_transparency_is_ready_other_pictures_need_the_model():
    rgba = Image.new("RGBA", (200, 120), (0, 0, 0, 0))
    rgba.paste((30, 200, 90, 255), (40, 20, 160, 100))
    b = io.BytesIO()
    rgba.save(b, "PNG")
    sid, done = thumbs.add_upload(b.getvalue(), "logo.png")
    assert done and sid.startswith("up-") and thumbs.ready(sid) and thumbs.cut_meta(sid)["sides"] == []
    assert thumbs.add_upload(b.getvalue()) == (sid, True)             # the same file again: instant
    j = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(j, "JPEG")
    sid2, done2 = thumbs.add_upload(j.getvalue(), "me.jpg")
    assert not done2 and thumbs._upload_src(sid2).is_file()
    assert thumbs.cut_state(sid2)[0] == "no-model"                     # no cut-out model in a test data folder
    with pytest.raises(thumbs.ThumbError):
        thumbs.add_upload(b"not a picture at all, just some text bytes", "x.png")
    assert thumbs.cut_state("../evil")[0] == "error"


def test_count_in():
    assert thumbs.count_in("make 4 thumbnails about my mistake") == 4
    assert thumbs.count_in("give me 3 new designs") == 3
    assert thumbs.count_in("اعمل ٥ صور مضحكة") == 5
    assert thumbs.count_in("عايز 7 تصاميم") == 7
    assert thumbs.count_in("1v5 clutch, funny") == 6                  # no count word next to the number
    assert thumbs.count_in("top 10 tips", default=3) == 3
    assert thumbs.count_in("99 pictures") == 12 and thumbs.count_in("0 ideas") == 1
    assert thumbs.count_in("") == 6


def test_wants_turns_clear_words_into_rules():
    assert thumbs._wants("first a zoom, then a reaction") == ["zoom", "reaction"]
    assert thumbs._wants("a reaction one and a ZOOM one") == ["reaction", "zoom"]
    assert thumbs._wants("me vs my brother", has_subject=True, two=True)[0] == "split"
    assert thumbs._wants("me vs my brother", has_subject=True, two=False) == []      # VS needs two subjects
    assert thumbs._wants("عايز واحد زوم وواحد ضد", True, True) == ["zoom", "split"]
    assert thumbs._wants("big text only please, a headline") == ["headline"]
    assert thumbs._wants("with my face") == [] and thumbs._wants("with my face", has_subject=True) == ["subject"]
    assert thumbs._wants("label box") == ["label"]


# ---------------------------------------------------------------- the studio API on a tiny real video
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")


@pytest.fixture
def srv(monkeypatch):
    monkeypatch.setattr(jobs, "_unload_models", lambda: None)
    s = server.make_server(0)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s
    s.shutdown()
    s.server_close()


def call(s, method, path, body=None, headers=None, raw=False):
    c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=120)
    h = {"Host": f"127.0.0.1:{s.port}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    payload = r.read()
    hdrs = {k.lower(): v for k, v in r.getheaders()}
    c.close()
    if raw:
        return r.status, hdrs, payload
    return r.status, json.loads(payload or b"{}")


@pytest.fixture
def video(tmp_path, srv, monkeypatch, emoji_files):
    """A 6 s 640x360 clip with moving pictures and sound, added to a new project. -> (pid, vid, src)"""
    src = tmp_path / "rec" / "Lesson 1.mp4"
    src.parent.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=6", "-f", "lavfi",
                    "-i", "sine=f=330:d=6", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(src)], check=True, creationflags=NO_WINDOW)
    trash = tmp_path / "recycle-bin"
    trash.mkdir()

    def recycle(paths):
        for p in paths:
            shutil.move(str(p), str(trash / p.name))
        return True
    monkeypatch.setattr(projects, "to_recycle_bin", recycle)
    pid = call(srv, "POST", "/api/projects", {"name": "Lessons"})[1]["project"]["id"]
    call(srv, "POST", f"/api/projects/{pid}", {"fields": {"video_type": "screen", "language": "en"}})
    vid = call(srv, "POST", f"/api/projects/{pid}/videos", {"paths": [str(src)]})[1]["added"][0]["id"]
    return pid, vid, src


@needs_ffmpeg
def test_studio_api_render_save_list_delete(srv, video):
    pid, vid, src = video
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    q = f"project={pid}&video={vid}"
    assert call(srv, "GET", f"/api/thumbs/list?{q}")[1] == {"thumbs": []}
    status, info = call(srv, "GET", f"/api/thumbs/info?{q}")
    assert status == 200 and info["name"] == "Lesson 1" and info["cands"] and info["cut_model"] is False
    assert [p[0] for p in info["presets"]] == thumbs.PRESET_KEYS and len(info["emoji"]) == len(FAKE_EMOJI)
    d = call(srv, "POST", "/api/thumbs/open", {"project": pid, "video": vid, "n": None})[1]["design"]
    d["text"]["s"] = "FOR LOOPS"
    status, h, jpg = call(srv, "POST", "/api/thumbs/render", {"project": pid, "video": vid, "design": d}, raw=True)
    assert status == 200 and h["content-type"] == "image/jpeg" and jpg[:2] == b"\xff\xd8"
    assert Image.open(io.BytesIO(jpg)).size == (1280, 720)
    boxes = json.loads(h["x-boxes"])
    assert "text" in boxes and len(boxes["text"]) == 4
    r = call(srv, "POST", "/api/thumbs/layout", {"project": pid, "video": vid, "design": d, "preset": "headline"})[1]
    assert r["design"]["preset"] == "headline" and r["note"] == ""
    r = call(srv, "POST", "/api/thumbs/layout", {"project": pid, "video": vid, "design": d, "preset": "split"})[1]
    assert r["design"]["preset"] != "split" and "two subjects" in r["note"]
    r = call(srv, "POST", "/api/thumbs/save", {"project": pid, "video": vid, "n": None, "design": d})[1]
    assert r["ok"] and r["n"] == 1 and r["thumbs"][0]["file"] == "thumbs/thumb_1.jpg"
    status, h, body = call(srv, "GET", f"/files/{pid}/{vid}/thumbs/thumb_1.jpg", raw=True)
    assert status == 200 and body[:2] == b"\xff\xd8" and len(body) < 2_000_000
    saved = json.loads((thumbs.thumbs_dir(pid, vid) / "thumb_1.json").read_text(encoding="utf-8"))
    assert saved["text"]["s"] == "FOR LOOPS"
    again = call(srv, "POST", "/api/thumbs/open", {"project": pid, "video": vid, "n": 1})[1]["design"]
    assert again == thumbs.normal(saved)
    r = call(srv, "POST", "/api/thumbs/more", {"project": pid, "video": vid, "count": 2})[1]
    assert r["ok"] and r["made"] == [2, 3]
    r = call(srv, "POST", "/api/thumbs/auto", {"project": pid, "video": vid})[1]
    assert r["ok"] and r["first"] == 4 and r["made"] == 3
    presets = [json.loads((thumbs.thumbs_dir(pid, vid) / f"thumb_{n}.json").read_text(encoding="utf-8"))["preset"]
               for n in (4, 5, 6)]
    assert len(set(presets)) == 3                                       # 3 different styles
    assert [t["n"] for t in call(srv, "GET", f"/api/thumbs/list?{q}")[1]["thumbs"]] == [1, 2, 3, 4, 5, 6]
    r = call(srv, "POST", "/api/thumbs/delete_many", {"project": pid, "video": vid, "ns": [1, 2, "x"]})[1]
    assert r["ok"] and [t["n"] for t in r["thumbs"]] == [3, 4, 5, 6]
    assert not (thumbs.thumbs_dir(pid, vid) / "thumb_1.json").exists()
    status, h, tile = call(srv, "GET", f"/api/thumbs/tile?{q}&t=2.5", raw=True)
    assert status == 200 and Image.open(io.BytesIO(tile)).width == 320
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before     # the user's video: untouched


@needs_ffmpeg
def test_saving_waits_while_the_video_is_being_made(srv, video, monkeypatch):
    pid, vid, _ = video
    monkeypatch.setattr(jobs, "_KINDS", dict(jobs._KINDS))
    jobs.register("make", lambda *a: None, steps=6, title="Make")
    jobs.enqueue("make", project=pid, video=vid)
    status, r = call(srv, "POST", "/api/thumbs/save", {"project": pid, "video": vid, "n": None, "design": {}})
    assert status == 409 and "being made" in r["error"]
    assert call(srv, "POST", "/api/thumbs/more", {"project": pid, "video": vid})[0] == 409
    assert call(srv, "GET", f"/api/thumbs/info?project={pid}&video={vid}")[1]["making"] is True
    jobs.cancel(project=pid, video=vid)
    assert call(srv, "POST", "/api/thumbs/save", {"project": pid, "video": vid, "n": None, "design": {}})[0] == 200


@needs_ffmpeg
def test_subject_upload_and_frame_cutout_without_a_model(srv, video):
    pid, vid, _ = video
    rgba = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    rgba.paste((240, 60, 60, 255), (60, 40, 240, 300))
    b = io.BytesIO()
    rgba.save(b, "PNG")
    import base64
    data = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    r = call(srv, "POST", "/api/thumbs/subject/upload", {"project": pid, "video": vid, "name": "me.png", "data": data})[1]
    assert r["ok"] and r["state"] == "ready" and r["id"].startswith("up-")
    status, h, png = call(srv, "GET", f"/api/thumbs/subject?id={r['id']}", raw=True)
    assert status == 200 and png[:4] == b"\x89PNG"
    assert thumbs.main_subjects(pid, vid) == (r["id"], None)          # remembered as this video's subject
    info = call(srv, "GET", f"/api/thumbs/info?project={pid}&video={vid}")[1]
    assert info["main"] == r["id"] and info["subjects"][0]["state"] == "ready"
    d = thumbs.new_design(2.0, "ME", subject=r["id"])
    out = call(srv, "POST", "/api/thumbs/layout", {"project": pid, "video": vid, "design": d, "preset": "subject"})[1]
    assert out["design"]["preset"] == "subject" and out["design"]["subject"]["on"]
    status, h, _ = call(srv, "POST", "/api/thumbs/render", {"project": pid, "video": vid, "design": out["design"]},
                        raw=True)
    assert status == 200 and "subject" in json.loads(h["x-boxes"])
    status, r2 = call(srv, "POST", "/api/thumbs/subject/frame", {"project": pid, "video": vid, "t": 3})
    assert r2["state"] == "no-model" and "Settings" in r2["error"]
    assert call(srv, "GET", f"/api/thumbs/cut?project={pid}&video={vid}&id={r2['id']}")[1]["state"] == "no-model"
    assert call(srv, "GET", "/api/thumbs/subject?id=../../secrets")[0] == 404
    assert call(srv, "GET", "/api/thumbs/emoji/..%2F..%2Fsecrets.json")[0] == 404
    assert call(srv, "GET", f"/api/thumbs/info?project={pid}&video=nope-0000")[0] == 404


@needs_ffmpeg
def test_ai_ideas_follow_the_rules_and_send_no_pictures_to_a_text_only_ai(srv, video):
    from broclips.providers.fake import FakeLLM
    pid, vid, _ = video
    ctx = thumbs.context(pid, vid)
    idea = {"text": "BIG MISTAKE", "style": "label", "picture": 1, "color": "white", "font": "anton", "emoji": "😱",
            "why": "curiosity"}
    llm = FakeLLM({"answers": [{"ideas": [dict(idea), dict(idea, text="WAIT WHAT", picture=2)]}]})
    made, msg = thumbs.ai_ideas(ctx, "first a zoom then a reaction please", 2, llm=llm)
    assert made == [1, 2] and "Made 2" in msg
    q = llm.calls[0]
    assert q["images"] == 0 and "MAKE EXACTLY 2" in q["user"] and "English" in q["user"]
    assert q["schema"]["properties"]["ideas"]["maxItems"] == 2
    presets = [json.loads((thumbs.thumbs_dir(pid, vid) / f"thumb_{n}.json").read_text(encoding="utf-8"))["preset"]
               for n in made]
    assert presets == ["zoom", "reaction"]                              # the user's words beat the AI's styles
    # the work-line job with the Fake AI from Settings
    config.save_settings({"thinking": {"provider": "fake"}})
    job = {"id": "job123", "project": pid, "video": vid, "args": {"prompt": "", "count": 1}}
    out = thumbs._ai_job(job, lambda *a, **k: None, lambda: False)
    assert out.startswith("✅") and thumbs.ai_state(pid, vid, "job123")["state"] == "done"
    status, r = call(srv, "POST", "/api/thumbs/ai", {"project": pid, "video": vid, "prompt": "make 3 ideas"})
    assert status == 200 and r["count"] == 3 and r["job"]["kind"] == "thumbai"
    st = call(srv, "GET", f"/api/thumbs/aistate?project={pid}&video={vid}&job={r['job']['id']}")[1]
    assert st["state"] == "queued" and "thumbs" in st


@needs_ffmpeg
def test_auto_for_the_pipeline_uses_moments_and_texts(srv, video):
    pid, vid, _ = video
    d = projects.video_dir(pid, vid)
    (d / "moments.json").write_text(json.dumps([
        {"s": 0.5, "e": 2.8, "title": "The setup", "hook": "Watch this loop", "why": "a tip", "type": "tip",
         "score": 9, "segments": [[0.5, 2.8, 1.0]]},
        {"s": 3.0, "e": 5.5, "title": "The mistake", "hook": "It never stops", "why": "funny fail", "type": "funny",
         "score": 7, "segments": [[3.0, 5.5, 1.0]]}]), encoding="utf-8")
    (d / "texts.json").write_text(json.dumps({"long": {"titles": ["Python for loops in 2 minutes | beginners"]},
                                              "shorts": [{"n": 1, "title": "Loops made easy"}]}), encoding="utf-8")
    paths = thumbs.auto(pid, vid, first_n=1, wait_cut=True)
    assert [p.name for p in paths] == ["thumb_1.jpg", "thumb_2.jpg", "thumb_3.jpg"]
    designs = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) for p in paths]
    assert len({x["preset"] for x in designs}) == 3 and designs[2]["preset"] == "headline"    # a screen recording
    assert {x["text"]["style"] for x in designs} >= {"white"}
    assert all(x["text"]["s"] for x in designs)
    words = [x["s"] for x in thumbs.suggestions(thumbs.context(pid, vid))]
    assert words[:3] == ["Watch this loop", "The setup", "It never stops"] and "Python for loops in 2 minutes" in words
    assert thumbs.video_title(thumbs.context(pid, vid)).startswith("Python for loops")
    assert math.isclose(len(thumbs.strip(thumbs.context(pid, vid), thumbs._moments(thumbs.context(pid, vid))[0])), 12)

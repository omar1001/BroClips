"""pipeline.py + results_api.py end to end with a fake Thinking AI and fake listening on a generated clip (ffmpeg
testsrc2 + a tone, real CPU renders, no network, no GPU): all six steps, resuming after a pause, nothing made twice,
an edit through the API -> the Short is made again, keep / find new moments, the results payload."""
import http.client
import json
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

import broclips
from broclips import config, jobs, longvideo, pipeline, projects, providers, server, shorts

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DUR = 52.0


def words():
    """Two blocks of six sentences (0.5-18 s and 30-48 s, silence between) + a made-up 'Thanks for watching!'."""
    out = []
    for block, t0 in ((0, 0.5), (1, 30.0)):
        for k in range(6):
            for j, w in enumerate(f"block {block} sentence {k} here.".split()):
                s = t0 + k * 3.0 + j * 0.5
                out.append({"w": w, "s": round(s, 2), "e": round(s + 0.45, 2), "p": 0.95})
    for j, w in enumerate("Thanks for watching!".split()):
        out.append({"w": w, "s": 50.0 + j * 0.4, "e": 50.35 + j * 0.4, "p": 0.9})
    return out


class Brain:
    """The fake Thinking AI: answers by the question's schema."""
    name, model, vision, local = "fake", "script", False, True

    def __init__(self):
        self.calls, self.mode = [], "first"

    def chat_json(self, system, user, schema, images=None, temperature=0.2, max_tokens=1500):
        self.calls.append(user)
        props = schema.get("properties") or {}
        if "payoff" in json.dumps(schema) and "TRANSCRIPT PART" in user:
            if self.mode == "first":
                picks = [("L001", "L006", "First bit", 8), ("L007", "L012", "Second bit", 7)]
            else:
                picks = [("L001", "L006", "Same as kept", 9), ("L009", "L012", "A new bit", 6)]
            return {"notes": "two good bits", "moments": [
                {"start": a, "end": b, "payoff": b, "type": "funny", "why": f"why {t}", "title": t, "hook": f"hook {t}",
                 "score": s} for a, b, t, s in picks]}
        if "long_titles" in props:
            return {"long_titles": ["Title A", "Title B", "Title C"], "long_description": "A tiny test video.",
                    "long_hashtags": ["test video", "#BroClips"],
                    "shorts": [{"n": 1, "title": "Short one", "caption": "Caption one", "hashtags": ["one"]},
                               {"n": 2, "title": "Short two", "caption": "Caption two", "hashtags": ["two"]}]}
        if "chapters" in props:
            return {"chapters": [{"start": "L001", "name": "Start"}, {"start": "L005", "name": "Middle"},
                                 {"start": "L009", "name": "Second part"}]}
        return None

    def unload(self):
        pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    src = tmp_path / "rec" / "My test video.mp4"
    src.parent.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=320x180:r=10:d={DUR}",
                    "-f", "lavfi", "-i", f"sine=f=300:d={DUR}", "-af", "volume=enable='between(t,18.5,29.5)+gt(t,48.5)'"
                    ":volume=0", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(src)], check=True, creationflags=NO_WINDOW)
    before = (src.stat().st_size, src.stat().st_mtime)
    config.save_settings({"thinking": {"provider": "fake"}, "listening": {"provider": "fake"}, "encoder": "x264"})
    brain = Brain()
    monkeypatch.setattr(providers, "get_llm", lambda *a, **k: brain)
    thumbs = types.ModuleType("broclips.thumbs")
    thumbs.calls, thumbs.existing = [], []
    thumbs.auto = lambda pid, vid, first_n=1, **k: thumbs.calls.append((pid, vid, first_n))
    thumbs.thumb_list = lambda pid, vid: list(thumbs.existing)
    thumbs.next_n = lambda pid, vid: len(thumbs.existing) + 1
    monkeypatch.setitem(sys.modules, "broclips.thumbs", thumbs)        # M3's module is not part of this test
    monkeypatch.setattr(broclips, "thumbs", thumbs, raising=False)
    trash = tmp_path / "recycle"
    trash.mkdir()

    def recycle(paths):
        for p in paths:
            shutil.move(str(p), str(trash / f"{Path(p).parent.name}-{Path(p).name}"))
        return True
    monkeypatch.setattr(projects, "to_recycle_bin", recycle)
    p = projects.create("Test project")
    projects.update(p["id"], {"video_type": "other", "max_shorts": 2, "platforms": ["youtube", "shorts", "tiktok"]})
    vid = projects.add_videos(p["id"], [str(src)])["added"][0]["id"]
    vdir = projects.video_dir(p["id"], vid)
    (vdir / "audio16k.wav.fake.json").write_text(json.dumps({"language": "en", "words": words()}), encoding="utf-8")
    return types.SimpleNamespace(pid=p["id"], vid=vid, vdir=vdir, src=src, before=before, brain=brain,
                                 thumbs=thumbs, trash=trash)


def run(kind, s, args=None, job_id="job1", progress=None):
    seen = []

    def prog(step, label, detail="", frac=None):
        seen.append((step, label, detail))
        if progress:
            progress(step, label, detail, frac)
    fn = {"make": pipeline.make, "find_again": pipeline.find_again, "short_edit": pipeline.short_edit}[kind]
    msg = fn({"id": job_id, "kind": kind, "project": s.pid, "video": s.vid, "args": args or {}}, prog, lambda: False)
    return msg, seen


def test_make_everything_then_nothing_twice(setup, monkeypatch):
    s = setup
    msg, seen = run("make", s)
    assert msg == "✅ 2 Shorts, the long video (0:36 of 0:52) and titles and texts ready"
    assert [st for st in dict.fromkeys(x[0] for x in seen)] == [1, 2, 3, 4, 5, 6]
    d = s.vdir
    for n in (1, 2):
        mp4, js = shorts.paths(d, n)
        data = json.loads(js.read_text(encoding="utf-8"))
        assert mp4.is_file() and data["version"] == 1 and data["layout"] == "fit" and data["phrases"]
        assert data["title"] in ("First bit", "Second bit") and data["hook"].startswith("hook ")
    tr = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
    assert tr["dropped"][0]["text"] == "Thanks for watching!" and "silence" in tr["dropped"][0]["why"]
    long = json.loads((d / "long.json").read_text(encoding="utf-8"))
    assert (d / "long.mp4").is_file() and len(long["ranges"]) == 2 and long["dur"] == pytest.approx(36.0, abs=0.3)
    texts = json.loads((d / "texts.json").read_text(encoding="utf-8"))
    assert texts["long"]["titles"] == ["Title A", "Title B", "Title C"]
    assert [c["name"] for c in texts["long"]["chapters"]] == ["Start", "Middle", "Second part"]
    assert texts["long"]["copy"]["description"].startswith("A tiny test video.\n\n0:00 Start")
    assert texts["shorts"]["1"]["copy"]["tiktok"] == "Caption one #one"
    assert "#Shorts" in texts["shorts"]["2"]["copy"]["shorts"]["description"]
    stages = json.loads((d / "stages.json").read_text(encoding="utf-8"))
    assert all(stages.get(k) for k in ("prepare", "listen", "moments", "shorts", "long", "texts", "thumbs"))
    assert s.thumbs.calls == [(s.pid, s.vid, 1)]
    log = (d / "ai_log.jsonl").read_text(encoding="utf-8")
    for kind in ("listen", "clean", "idea", "render", "done"):
        assert f'"k": "{kind}"' in log
    assert json.loads((d / "video.json").read_text(encoding="utf-8"))["status"] == "done"
    assert (s.src.stat().st_size, s.src.stat().st_mtime) == s.before                   # the source is untouched
    # the same job again: every stage is done, nothing is asked or made twice
    asked = len(s.brain.calls)
    monkeypatch.setattr(shorts, "render", lambda *a, **k: pytest.fail("rendered twice"))
    monkeypatch.setattr(longvideo, "render", lambda *a, **k: pytest.fail("rendered twice"))
    assert run("make", s)[0] == msg and len(s.brain.calls) == asked


def test_a_pause_continues_where_it_stopped(setup):
    s = setup
    projects.update(s.pid, {"max_shorts": 1, "platforms": ["shorts"]})
    state = {"n": 0}

    def pause_in_step_4(step, label, detail, frac):
        if step == 4:
            state["n"] += 1
            if state["n"] == 1:
                raise jobs.Paused("League of Legends.exe")
    with pytest.raises(jobs.Paused):
        run("make", s, progress=pause_in_step_4)
    scans = sum("TRANSCRIPT PART" in c for c in s.brain.calls)
    msg, seen = run("make", s)
    assert msg.startswith("✅ 1 Short") and "long video" not in msg
    assert sum("TRANSCRIPT PART" in c for c in s.brain.calls) == scans == 1            # not asked again
    assert 2 not in [x[0] for x in seen] and 3 not in [x[0] for x in seen]           # listening + search were done
    assert not (s.vdir / "long.mp4").exists()                                         # YouTube is not ticked


def call(srv, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=20)
    h = {"Host": f"127.0.0.1:{srv.port}"}
    data = None
    if body is not None:
        data, h["Content-Type"] = json.dumps(body).encode("utf-8"), "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    out = r.status, json.loads(r.read() or b"{}")
    c.close()
    return out


def test_results_api_edit_keep_and_find_new_moments(setup, monkeypatch):
    s = setup
    run("make", s)
    monkeypatch.setattr(jobs, "_unload_models", lambda: None)
    srv = server.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"/api/results/{s.pid}/{s.vid}"
        st, r = call(srv, "GET", base)
        assert st == 200 and len(r["shorts"]) == 2 and r["long"]["file"].startswith(f"/files/{s.pid}/{s.vid}/long.mp4")
        assert r["shorts"][0]["file"].startswith(f"/files/{s.pid}/{s.vid}/shorts/short_01.mp4?v=")
        assert r["texts"]["long"]["titles"][0] == "Title A" and r["log"] and r["marathon"]["questions"] > 0
        assert call(srv, "GET", f"/api/results/{s.pid}/nope-1234")[0] == 404
        # keep short 1, edit short 2 (title on top + 1 s earlier start + one caption phrase)
        assert call(srv, "POST", base + "/mark", {"n": 1, "mark": "keep"})[1]["mark"] == "keep"
        two = r["shorts"][1]
        st, e = call(srv, "POST", base + "/edit", {"n": 2, "hook": "A better title", "start": two["s"] - 1.0,
                                                   "captions": [{"i": 0, "text": "fixed words"}]})
        assert st == 200 and e["job"]["kind"] == "short_edit" and set(e["changed"]) == {"title", "start / end",
                                                                                         "captions"}
        assert call(srv, "POST", base + "/edit", {"n": 2, "start": 10.0, "end": 11.0})[0] == 400   # too short
        assert call(srv, "POST", base + "/wishes", {"text": "0:40 the second part\nmake it fun"})[1]["ok"]
    finally:
        srv.shutdown()
        srv.server_close()
    data = shorts.load(s.vdir, 2)
    assert data["dirty"] and data["hook"] == "A better title" and data["caption_fixes"][0]["text"] == "fixed words"
    msg, _ = run("short_edit", s, {"n": 2, "rev": data["rev"]})
    after = shorts.load(s.vdir, 2)
    assert msg == "✅ Short 2 made again" and after["version"] == 2 and not after["dirty"]
    assert after["phrases"][0]["text"] == "fixed words" and after["dur"] == pytest.approx(two["dur"] + 1.0, abs=0.05)
    # find new moments: the kept Short stays, the other one goes to the Recycle Bin, the wishes are asked first
    s.brain.mode = "again"
    s.thumbs.existing = [{"n": 1}, {"n": 2}, {"n": 3}]                # thumbnails exist: new ones come after them
    v1 = shorts.load(s.vdir, 1)["version"]
    msg, seen = run("find_again", s, job_id="job2")
    assert msg == "✅ 2 Shorts ready (new moments)"
    now = shorts.listing(s.vdir)
    assert [x["title"] for x in now] == ["First bit", "A new bit"]
    assert now[0]["mark"] == "keep" and now[0]["version"] == v1
    assert any(p.name.endswith("short_02.mp4") for p in s.trash.iterdir())
    assert any("remembers a good moment around 0:40" in c for c in s.brain.calls)
    assert not (s.vdir / "shorts_new").exists()
    assert s.thumbs.calls[-1] == (s.pid, s.vid, 4)
    log = (s.vdir / "ai_log.jsonl").read_text(encoding="utf-8")
    assert "already a Short you keep" in log and "Find new moments" in log

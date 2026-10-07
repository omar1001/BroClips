"""The local web app (SPEC §3, §9): pages, projects API, settings (keys masked, an empty key field keeps the key),
/files with HTTP Range and nothing outside a video's folder, other websites refused, AI test, quit, downloads."""
import http.client
import json
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pytest

from broclips import config, jobs, projects, server

SECRET = "AIzaSy-TEST-KEY-never-shown-7f3c1234"


@pytest.fixture
def srv(monkeypatch):
    monkeypatch.setattr(jobs, "_unload_models", lambda: None)          # never touch a real Ollama from a test
    monkeypatch.setattr(projects, "_probe", lambda p: {"dur": 61.5, "has_video": True, "n_audio": 1})
    s = server.make_server(0)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    yield s
    s.shutdown()
    s.server_close()


def call(s, method, path, body=None, headers=None, raw=False, timeout=15):
    c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=timeout)
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
def recording(tmp_path):
    src = tmp_path / "recordings" / "Lesson one.mp4"
    src.parent.mkdir()
    src.write_bytes(b"\x00" * 2048)
    return src


@pytest.fixture
def recycle_bin(monkeypatch, tmp_path):
    trash = tmp_path / "recycle-bin"
    trash.mkdir()

    def fake(paths):
        for p in paths:
            shutil.move(str(p), str(trash / Path(p).name))
        return True
    monkeypatch.setattr(projects, "to_recycle_bin", fake)


def test_pages_and_static_files(srv):
    for path, ctype in (("/", "text/html"), ("/settings.html", "text/html"), ("/help.html", "text/html"),
                        ("/settings", "text/html"), ("/app.css", "text/css"), ("/app.js", "text/javascript"),
                        ("/icons.js", "text/javascript"), ("/studio.html", "text/html"),
                        ("/broclips.png", "image/png"), ("/favicon.ico", "image/x-icon")):
        status, hdrs, body = call(srv, "GET", path, raw=True)
        assert status == 200 and hdrs["content-type"].startswith(ctype), path
        assert len(body) > 100
    assert b"BroClips" in call(srv, "GET", "/", raw=True)[2]
    assert call(srv, "GET", "/nope.html")[0] == 404
    assert call(srv, "GET", "/../server.py")[0] == 404
    assert call(srv, "GET", "/..%2fserver.py")[0] == 404
    assert call(srv, "HEAD", "/app.css", raw=True)[2] == b""


def test_version_state_and_first_run(srv):
    status, v = call(srv, "GET", "/api/version")
    assert status == 200 and v["app"] == "BroClips" and v["version"] == config.VERSION and v["busy"] is False
    st = call(srv, "GET", "/api/state")[1]
    assert st["first_run"] is True and st["work"]["text"] == "💤 Ready"
    assert call(srv, "POST", "/api/settings", {"settings": {}})[1]["ok"] is True       # "Skip for now"
    assert call(srv, "GET", "/api/state")[1]["first_run"] is False


def test_projects_and_videos_crud(srv, recording, recycle_bin):
    before = recording.read_bytes()
    status, r = call(srv, "POST", "/api/projects", {"name": "Python lessons"})
    assert status == 200 and r["ok"]
    pid = r["project"]["id"]
    assert [p["id"] for p in call(srv, "GET", "/api/projects")[1]["projects"]] == [pid]
    r = call(srv, "POST", f"/api/projects/{pid}", {"fields": {"prompt": "Short lessons", "marathon": True,
                                                                "video_type": "screen"}})[1]
    assert r["project"]["prompt"] == "Short lessons" and r["project"]["marathon"] is True

    status, r = call(srv, "POST", f"/api/projects/{pid}/videos",
                     {"paths": [str(recording), str(recording.parent / "missing.mp4")]})
    assert status == 200 and len(r["added"]) == 1 and len(r["skipped"]) == 1
    vid = r["added"][0]["id"]
    got = call(srv, "GET", f"/api/projects/{pid}")[1]
    assert got["project"]["videos"] == 1 and got["videos"][0]["dur"] == 61.5
    v = call(srv, "GET", f"/api/projects/{pid}/videos/{vid}")[1]["video"]
    assert v["status"] == "new" and v["src"] == str(recording.resolve())

    assert call(srv, "POST", f"/api/projects/{pid}/videos", {"paths": []})[0] == 400
    assert call(srv, "GET", f"/api/projects/{pid}/videos/nope-0000")[0] == 404
    assert call(srv, "GET", "/api/projects/..%2F..%2Fsecrets.json")[0] == 404

    status, r = call(srv, "POST", f"/api/projects/{pid}/videos/{vid}/remove", {})
    assert status == 200 and "not touched" in r["msg"]
    assert call(srv, "POST", f"/api/projects/{pid}/delete", {})[1]["ok"] is True
    assert call(srv, "GET", f"/api/projects/{pid}")[0] == 404
    assert recording.read_bytes() == before                             # the user's file: untouched


def test_make_waits_for_the_pipeline(srv, recording, monkeypatch):
    pid = call(srv, "POST", "/api/projects", {"name": "Lessons"})[1]["project"]["id"]
    vid = call(srv, "POST", f"/api/projects/{pid}/videos", {"paths": [str(recording)]})[1]["added"][0]["id"]
    monkeypatch.setattr(jobs, "_KINDS", {k: v for k, v in jobs._KINDS.items() if k != "make"})
    status, r = call(srv, "POST", f"/api/projects/{pid}/videos/{vid}/make", {})
    assert status == 409 and "next version" in r["error"]

    jobs.register("make", lambda *a: None, steps=6, title="Make")      # what M2's pipeline will register
    status, r = call(srv, "POST", f"/api/projects/{pid}/videos/{vid}/make", {"again": False})
    assert status == 200 and r["job"]["kind"] == "make" and r["job"]["title"] == "Lesson one"
    again = call(srv, "POST", f"/api/projects/{pid}/videos/{vid}/make", {"again": False})[1]
    assert again["job"]["id"] == r["job"]["id"]                         # a double click adds nothing
    assert call(srv, "GET", f"/api/projects/{pid}/videos/{vid}")[1]["video"]["status"] == "queued"
    assert call(srv, "GET", "/api/state")[1]["work"]["queue"] == 1
    assert call(srv, "POST", f"/api/projects/{pid}/videos/{vid}/cancel", {})[1]["stopped"] == 1
    assert call(srv, "GET", f"/api/projects/{pid}/videos/{vid}")[1]["video"]["status"] == "stopped"


def test_settings_mask_keys_and_an_empty_field_keeps_the_key(srv):
    status, r = call(srv, "POST", "/api/settings", {"settings": {"thinking": {"provider": "gemini"}},
                                                    "keys": {"gemini": SECRET, "evil": "x"}})
    assert status == 200 and r["keys"] == {"gemini": "…1234"}
    assert r["settings"]["thinking"]["provider"] == "gemini"
    for path in ("/api/settings", "/api/state"):
        assert SECRET.encode() not in call(srv, "GET", path, raw=True)[2]
    assert config.secret("gemini") == SECRET
    assert SECRET not in (config.data_dir() / "settings.json").read_text(encoding="utf-8")

    r = call(srv, "POST", "/api/settings", {"settings": {}, "keys": {"gemini": ""}})[1]             # empty: kept
    assert r["keys"] == {"gemini": "…1234"} and config.secret("gemini") == SECRET
    call(srv, "POST", "/api/settings", {"settings": {}, "keys": {"gemini": "…1234"}})               # echo: kept
    assert config.secret("gemini") == SECRET
    r = call(srv, "POST", "/api/settings", {"settings": {}, "keys": {"anthropic": "sk-ant-NEWKEY9999"}})[1]
    assert r["keys"] == {"gemini": "…1234", "anthropic": "…9999"}

    r = call(srv, "POST", "/api/keys/remove", {"name": "gemini"})[1]
    assert r["keys"] == {"anthropic": "…9999"} and config.secret("gemini") == ""
    assert call(srv, "POST", "/api/keys/remove", {"name": "nope"})[0] == 400


def test_settings_are_cleaned(srv):
    r = call(srv, "POST", "/api/settings", {"settings": {
        "encoder": "bogus", "max_shorts": 999, "evil": 1, "allow_images_to_cloud": "true",
        "thinking": {"provider": "not-an-ai", "openai": {"base_url": "https://openrouter.ai/api/v1/",
                                                         "key_name": "openai", "model": "some/model"}},
        "listening": {"provider": "fake", "openai": {"base_url": "javascript:alert(1)"}},
        "pause_while_running": ["C:\\Games\\League of Legends.exe", "", "x.exe", "X.EXE"],
        "blur_boxes": [[1550, 990, 360, 80], ["a"], [0, 0, 0, 5], [1, 2, 3]]}})[1]
    s = r["settings"]
    assert s["encoder"] == "auto" and s["max_shorts"] == 30 and "evil" not in s
    assert s["allow_images_to_cloud"] is True
    assert s["thinking"]["provider"] == "ollama"                        # unknown provider: the old one stays
    assert s["thinking"]["openai"]["model"] == "some/model"
    assert s["thinking"]["openai"]["key_name"] == "openrouter"          # derived from the address, never sent
    assert s["listening"]["provider"] == "fake"
    assert s["listening"]["openai"]["base_url"] == config.DEFAULTS["listening"]["openai"]["base_url"]
    assert s["pause_while_running"] == ["League of Legends.exe", "x.exe"]
    assert s["blur_boxes"] == [[1550, 990, 360, 80]]
    r = call(srv, "POST", "/api/settings", {"settings": {"thinking": {"openai": {"base_url": "https://my.host/v1"}}}})
    assert r[1]["settings"]["thinking"]["openai"]["key_name"] == "custom"   # the OpenAI key never goes elsewhere
    status, r = call(srv, "POST", "/api/settings", {"settings": {"output_dir": "relative\\folder"}})
    assert status == 400 and "full path" in r["error"]


def test_data_folder_can_move(srv, tmp_path):
    call(srv, "POST", "/api/settings", {"settings": {"max_shorts": 4}, "keys": {"gemini": SECRET}})
    new = tmp_path / "elsewhere" / "BroClips"
    r = call(srv, "POST", "/api/settings", {"settings": {}, "data_dir": str(new)})[1]
    assert r["data_dir"] == str(new) and config.data_dir() == new
    assert r["settings"]["max_shorts"] == 4 and config.secret("gemini") == SECRET          # they came along
    assert (new / "secrets.json").is_file()


def test_files_are_served_with_ranges(srv):
    pid = call(srv, "POST", "/api/projects", {"name": "Lessons"})[1]["project"]["id"]
    d = projects.video_dir(pid, "lesson-abcd") / "shorts"
    d.mkdir(parents=True)
    data = bytes(range(256)) * 4
    (d / "short_01.mp4").write_bytes(data)
    url = f"/files/{pid}/lesson-abcd/shorts/short_01.mp4"
    status, h, body = call(srv, "GET", url, raw=True)
    assert status == 200 and body == data and h["accept-ranges"] == "bytes" and h["content-type"] == "video/mp4"
    status, h, body = call(srv, "GET", url, headers={"Range": "bytes=10-19"}, raw=True)
    assert status == 206 and body == data[10:20] and h["content-range"] == "bytes 10-19/1024"
    status, h, body = call(srv, "GET", url, headers={"Range": "bytes=1000-"}, raw=True)
    assert status == 206 and body == data[1000:] and h["content-range"] == "bytes 1000-1023/1024"
    status, h, body = call(srv, "GET", url, headers={"Range": "bytes=-5"}, raw=True)
    assert status == 206 and body == data[-5:]
    status, h, body = call(srv, "GET", url, headers={"Range": "bytes=5000-"}, raw=True)
    assert status == 416 and h["content-range"] == "bytes */1024"
    status, h, body = call(srv, "HEAD", url, raw=True)
    assert status == 200 and body == b"" and h["content-length"] == "1024"
    assert call(srv, "GET", f"/files/{pid}/lesson-abcd/shorts/none.mp4")[0] == 404


def test_files_refuses_everything_outside_a_video_folder(srv):
    call(srv, "POST", "/api/settings", {"settings": {}, "keys": {"gemini": SECRET}})
    pid = call(srv, "POST", "/api/projects", {"name": "Lessons"})[1]["project"]["id"]
    projects.video_dir(pid, "lesson-abcd").mkdir(parents=True)
    (config.projects_dir() / pid / "project-secret.txt").write_text(SECRET, encoding="utf-8")
    for path in (f"/files/{pid}/lesson-abcd/../../../secrets.json",
                 f"/files/{pid}/lesson-abcd/..%2F..%2F..%2Fsecrets.json",
                 f"/files/{pid}/lesson-abcd/..%5C..%5C..%5Csecrets.json",
                 f"/files/{pid}/lesson-abcd/../project-secret.txt",
                 f"/files/{pid}/lesson-abcd/C:%5CWindows%5Cwin.ini",
                 f"/files/{pid}/..%2F..%2Fsecrets.json/x",
                 "/files/..%2F../secrets.json/x/y",
                 f"/files/{pid}/lesson-abcd/" + quote(str(config.data_dir() / "secrets.json"))):
        status, h, body = call(srv, "GET", path, raw=True)
        assert status in (403, 404), path
        assert SECRET.encode() not in body, path


def test_other_websites_are_refused(srv):
    port = srv.port
    assert call(srv, "GET", "/api/settings", headers={"Host": f"evil.example:{port}"})[0] == 403    # DNS rebinding
    bad = [{"Origin": "http://evil.example"}, {"Origin": "null"}, {"Content-Type": "text/plain"},
           {"Sec-Fetch-Site": "cross-site"}]
    for h in bad:
        status, r = call(srv, "POST", "/api/projects", {"name": "x"}, headers=h)
        assert status == 403, h
    assert call(srv, "GET", "/api/projects")[1]["projects"] == []
    ok = call(srv, "POST", "/api/projects", {"name": "mine"},
              headers={"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin"})
    assert ok[0] == 200
    assert call(srv, "GET", "/api/version", headers={"Host": f"localhost:{port}"})[0] == 200


def test_provider_test_with_the_fake_ai(srv):
    call(srv, "POST", "/api/settings", {"settings": {"thinking": {"provider": "fake"},
                                                     "listening": {"provider": "fake"}}})
    assert call(srv, "POST", "/api/providers/test", {"role": "thinking"})[1] == {"ok": True, "message": "Fake AI ready"}
    assert call(srv, "POST", "/api/providers/test", {"role": "listening"})[1]["message"] == "Fake listening ready"
    assert call(srv, "POST", "/api/providers/test", {"role": "dancing"})[0] == 400


def test_provider_test_reports_plain_errors(srv, monkeypatch):
    from broclips.providers import fake
    from broclips.providers.base import ProviderError

    def broken(self):
        raise ProviderError("No key yet. Paste your key in Settings.")
    monkeypatch.setattr(fake.FakeLLM, "test", broken)
    call(srv, "POST", "/api/settings", {"settings": {"thinking": {"provider": "fake"}}})
    assert call(srv, "POST", "/api/providers/test", {"role": "thinking"})[1] == \
        {"ok": False, "message": "No key yet. Paste your key in Settings."}


def test_quit_only_when_no_job_runs(srv, monkeypatch):
    monkeypatch.setattr(jobs, "_KINDS", dict(jobs._KINDS))
    monkeypatch.setattr(jobs, "_which_running", lambda names: "")
    gate = threading.Event()
    jobs.register("block", lambda *a: gate.wait(5))
    jobs.LINE.idle_wait = 0.02
    jobs.enqueue("block")
    jobs.start()
    end = time.monotonic() + 5
    while not jobs.running() and time.monotonic() < end:
        time.sleep(0.01)
    assert call(srv, "GET", "/api/version")[1]["busy"] is True
    r = call(srv, "POST", "/api/quit", {})[1]
    assert r["ok"] is False and "working" in r["error"]
    gate.set()
    while jobs.running() and time.monotonic() < end:
        time.sleep(0.01)
    assert call(srv, "POST", "/api/quit", {})[1] == {"ok": True}
    end = time.monotonic() + 5
    while time.monotonic() < end:                                       # the server stops by itself
        try:
            call(srv, "GET", "/api/version", timeout=0.5)
        except OSError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the server did not stop")


def test_downloads_and_jobs_go_into_the_work_line(srv):
    status, r = call(srv, "POST", "/api/assets/download", {"what": "lite"})
    assert status == 200 and r["job"]["kind"] == "assets" and r["job"]["args"] == {"cutout": "lite"}
    assert call(srv, "POST", "/api/assets/download", {"what": "lite"})[1]["job"]["id"] == r["job"]["id"]
    assert call(srv, "POST", "/api/assets/download", {"what": "everything"})[0] == 400
    st = call(srv, "GET", "/api/state")[1]["work"]
    assert st["queue"] == 1 and st["waiting"][0]["title"].startswith("Light cut-out model")
    assert call(srv, "POST", "/api/jobs/enqueue", {"kind": "rm-rf"})[0] == 409
    assert call(srv, "POST", "/api/jobs/cancel", {"id": r["job"]["id"]})[1]["stopped"] == 1
    assert call(srv, "POST", "/api/jobs/cancel", {})[0] == 400
    a = call(srv, "GET", "/api/assets")[1]
    assert a["fonts"]["of"] == 10 and a["emoji"]["of"] == 47 and a["cutout"]["path"] == ""
    assert call(srv, "POST", "/api/open_folder", {"which": "C:\\Windows"})[0] == 400

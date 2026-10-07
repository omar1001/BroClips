"""assets.py without the internet: copying from a folder you already have (--from), status, which model is used,
resumable hash-checked downloads (a local HTTP server plays the internet), the work-line job and the CLI."""
import hashlib
import http.server
import threading

import pytest

from broclips import assets, config, jobs


def make_source(tmp_path):
    """Like an assets folder from another install (LoL Clips layout): fonts/, emoji/, bin/fribidi-0.dll."""
    src = tmp_path / "old-assets"
    for sub in ("fonts", "emoji", "bin"):
        (src / sub).mkdir(parents=True)
    (src / "fonts" / "Lalezar-Regular.ttf").write_bytes(b"f" * 30000)
    (src / "fonts" / "OFL-Lalezar.txt").write_text("SIL Open Font License", encoding="utf-8")
    (src / "emoji" / "skull.png").write_bytes(b"p" * 3000)
    (src / "bin" / "fribidi-0.dll").write_bytes(b"d" * 5000)
    return src


def test_the_download_list_matches_the_spec():
    names = {d.name for d, _, _ in assets._wanted_essential()}
    for f in ("Lalezar-Regular.ttf", "Alexandria[wght].ttf", "Cairo[slnt,wght].ttf", "Changa[wght].ttf",
              "Marhey[wght].ttf", "Jomhuria-Regular.ttf", "Montserrat[wght].ttf", "Montserrat-ExtraBold.ttf",
              "Anton-Regular.ttf", "BebasNeue-Regular.ttf", "OFL-Montserrat.txt", "LICENSE-fluentui-emoji.txt",
              "face_with_tears_of_joy.png", "rocket.png"):
        assert f in names, f
    assert len(assets.EMOJI) == 47
    urls = {d.name: u for d, u, _ in assets._wanted_essential()}
    assert urls["Cairo[slnt,wght].ttf"].endswith("/ofl/cairo/Cairo%5Bslnt%2Cwght%5D.ttf")
    assert urls["face_with_tears_of_joy.png"].endswith("/assets/Face%20with%20tears%20of%20joy/3D/"
                                                       "face_with_tears_of_joy_3d.png")


def test_copy_from_an_existing_folder(tmp_path):
    src = make_source(tmp_path)
    assert assets.run(essential=True, src=src, download=False) == []
    assert (assets.fonts_dir() / "Lalezar-Regular.ttf").read_bytes() == b"f" * 30000
    assert (assets.emoji_dir() / "skull.png").is_file()
    assert (src / "fonts" / "Lalezar-Regular.ttf").is_file()           # copied, not moved
    st = assets.status()
    assert st["fonts"] == {"have": 1, "of": 10} and st["emoji"] == {"have": 1, "of": 47}
    assert st["fribidi"] is (True if config.IS_WINDOWS else None)
    assert st["essential_ok"] is False
    assert assets.copy_from(src) == 0                                   # nothing new the second time


def test_a_damaged_model_copy_is_not_used(tmp_path):
    src = tmp_path / "models"
    src.mkdir()
    (src / assets.CUTOUT["lite"]["file"]).write_bytes(b"not the real model")
    assert assets.copy_from(src, essential=False, variants=["lite"]) == 0
    assert not assets.cutout_path("lite").exists()
    assert assets.cutout_model_path() is None


def test_which_cut_out_model_is_used(tmp_path):
    lite = assets.cutout_path("lite")
    lite.write_bytes(b"x")
    assert assets.cutout_model_path() == lite
    full = assets.cutout_path("general")
    full.write_bytes(b"x")
    assert assets.cutout_model_path() == full
    own = tmp_path / "mine.onnx"
    own.write_bytes(b"x")
    s = config.settings()
    s["cutout_model"] = str(own)
    config.save_settings(s)
    assert assets.cutout_model_path() == own


@pytest.fixture
def web():
    """A local HTTP server playing the internet (supports Range like GitHub does)."""
    data = bytes(range(256)) * 400
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            seen.append(rng)
            start = int(rng[6:-1]) if rng else 0
            body = data[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/model.onnx", data, seen
    srv.shutdown()
    srv.server_close()


def test_download_resumes_and_checks_the_hash(web, tmp_path):
    url, data, seen = web
    dst = tmp_path / "out" / "model.onnx"
    dst.parent.mkdir()
    (tmp_path / "out" / "model.onnx.part").write_bytes(data[:1000])     # an earlier download that was stopped
    got = []
    assets.get(url, dst, min_bytes=1000, md5=hashlib.md5(data).hexdigest(), rate=50_000_000,
               on_bytes=lambda done, total: got.append((done, total)))
    assert dst.read_bytes() == data and seen == ["bytes=1000-"]
    assert got[-1] == (len(data), len(data))
    assert not (tmp_path / "out" / "model.onnx.part").exists()

    bad = tmp_path / "out" / "bad.onnx"
    with pytest.raises(RuntimeError, match="checksum"):
        assets.get(url, bad, min_bytes=10, md5="0" * 32, rate=50_000_000)
    assert not bad.exists() and not (tmp_path / "out" / "bad.onnx.part").exists()


def test_stop_keeps_the_part_file_for_later(web, tmp_path):
    url, data, _ = web
    dst = tmp_path / "model.onnx"
    with pytest.raises(jobs.Cancelled):
        assets.get(url, dst, rate=50_000_000, on_bytes=lambda d, t: None, cancelled=lambda: True)
    assert not dst.exists()


def test_the_assets_job(monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "run", lambda **kw: calls.append(kw) or [])
    msg = assets._job({"args": {"cutout": "lite"}}, lambda *a, **k: None, lambda: False)
    assert msg == "✅ Downloaded the light cut-out model (214 MB)" and calls[0]["variant"] == "lite"
    monkeypatch.setattr(assets, "run", lambda **kw: ["Could not download x.ttf."])
    with pytest.raises(jobs.JobError, match="Could not download"):
        assets._job({"args": {"essential": True}}, lambda *a, **k: None, lambda: False)
    assert jobs.registered("assets")


def test_cli(tmp_path, capsys):
    assert assets.main([]) == 2                                         # nothing chosen: shows the help
    src = make_source(tmp_path)
    assert assets.main(["--from", str(src), "--no-download"]) == 0
    assert "Fonts 1/10 | emoji 1/47" in capsys.readouterr().out

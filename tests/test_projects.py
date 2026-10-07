"""Projects and videos on disk (SPEC §3): ids, defaults, validation, adding videos by reference only, removing only
BroClips' own folders (source videos are never touched), status from the work line."""
import os
import re
import shutil
from pathlib import Path

import pytest

from broclips import config, jobs, projects


@pytest.fixture
def recording(tmp_path):
    """A pretend recording OUTSIDE the data folder: BroClips may only read it."""
    src = tmp_path / "recordings" / "My Lesson 1.mp4"
    src.parent.mkdir()
    src.write_bytes(b"\x00\x01" * 500)
    return src


@pytest.fixture(autouse=True)
def fake_probe(monkeypatch):
    monkeypatch.setattr(projects, "_probe", lambda p: {"dur": 123.4, "fps": 60.0, "width": 1920, "height": 1080,
                                                       "n_audio": 1, "has_video": True, "codec": "h264"})


@pytest.fixture
def recycle_bin(monkeypatch, tmp_path):
    """The Recycle Bin, replaced by a folder (tests never fill the real one)."""
    trash = tmp_path / "recycle-bin"
    trash.mkdir()
    moved = []

    def fake(paths):
        for p in paths:
            shutil.move(str(p), str(trash / Path(p).name))
            moved.append(Path(p))
        return True
    monkeypatch.setattr(projects, "to_recycle_bin", fake)
    return moved


def fingerprint(p):
    st = p.stat()
    return p.read_bytes(), st.st_size, st.st_mtime_ns


def test_create_list_get():
    a = projects.create("Python Lessons #1")
    b = projects.create("دروس الرياضيات")              # Arabic name: the id falls back to a safe slug
    assert re.fullmatch(r"python-lessons-1-[a-z0-9]{4}", a["id"])
    assert re.fullmatch(r"project-[a-z0-9]{4}", b["id"])
    assert b["name"] == "دروس الرياضيات"
    assert a["prompt"] == "" and a["platforms"] == config.PROJECT_DEFAULTS["platforms"] and a["videos"] == 0
    assert (config.projects_dir() / a["id"] / "project.json").is_file()
    ids = [p["id"] for p in projects.list_projects()]
    assert sorted(ids) == sorted([a["id"], b["id"]])
    assert projects.get(a["id"])["name"] == "Python Lessons #1"
    assert projects.get("missing-0000") is None
    assert projects.create("   ")["name"] == config.PROJECT_DEFAULTS["name"]


def test_new_projects_use_the_settings_defaults():
    s = config.settings()
    s["language"], s["max_shorts"] = "ar", 5
    config.save_settings(s)
    p = projects.create("Arabic videos")
    assert p["language"] == "ar" and p["max_shorts"] == 5


def test_update_merges_and_validates():
    p = projects.create("Demo")
    out = projects.update(p["id"], {
        "name": "  Better   name ", "prompt": "Funny League moments in Egyptian Arabic", "video_type": "screen",
        "platforms": ["tiktok", "junk", "youtube"], "marathon": "true", "jump_cuts": False, "max_shorts": 999,
        "layout": "sideways", "language": "../etc", "silence_s": "0.1", "evil": 1})
    assert out["name"] == "Better name"
    assert out["prompt"] == "Funny League moments in Egyptian Arabic"
    assert out["video_type"] == "screen"
    assert out["platforms"] == ["youtube", "tiktok"]
    assert out["marathon"] is True and out["jump_cuts"] is False
    assert out["max_shorts"] == 30 and out["silence_s"] == 0.3
    assert out["layout"] == "auto" and out["language"] == "auto"       # bad values keep the old ones
    assert "evil" not in out
    assert projects.update(p["id"], {"language": "ar"})["language"] == "ar"
    assert projects.get(p["id"])["prompt"] == "Funny League moments in Egyptian Arabic"     # kept
    assert projects.update("missing-0000", {"name": "x"}) is None


def test_add_videos_only_references_the_file(recording):
    before = fingerprint(recording)
    p = projects.create("Lessons")
    out = projects.add_videos(p["id"], [str(recording)])
    assert out["skipped"] == [] and len(out["added"]) == 1
    v = out["added"][0]
    assert re.fullmatch(r"my-lesson-1-[a-z0-9]{4}", v["id"])
    assert v["src"] == str(recording.resolve()) and v["name"] == "My Lesson 1" and v["dur"] == 123.4
    assert v["status"] == "new" and v["src_exists"] is True
    assert v["probe"]["fps"] == 60.0
    assert fingerprint(recording) == before                            # not changed
    assert not list(config.data_dir().rglob("*.mp4"))                  # not copied
    assert projects.get(p["id"])["videos"] == 1
    assert [x["id"] for x in projects.list_videos(p["id"])] == [v["id"]]


def test_add_videos_explains_what_it_skips(recording, tmp_path, monkeypatch):
    p = projects.create("Lessons")
    projects.add_videos(p["id"], [str(recording)])
    notes = tmp_path / "notes.txt"
    notes.write_text("hi", encoding="utf-8")
    inside = projects.video_dir(p["id"], "made-abcd") / "shorts" / "short_01.mp4"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    out = projects.add_videos(p["id"], [str(recording), str(tmp_path / "nope.mp4"), "relative/video.mp4",
                                        str(notes), str(inside), "", f'"{recording}"'])
    why = [s["why"] for s in out["skipped"]]
    assert out["added"] == []
    assert why[0] == "Already in this project."
    assert why[1].startswith("File not found") and why[2].startswith("File not found")
    assert why[3].startswith("This is not a video file")
    assert why[4].startswith("This file was made by BroClips")
    assert why[5] == "Already in this project."                        # quotes from "Copy as path" are fine

    second = tmp_path / "recordings" / "silent.mp4"
    second.write_bytes(b"x")
    monkeypatch.setattr(projects, "_probe", lambda path: {"dur": 5, "has_video": True, "n_audio": 0})
    assert "no sound" in projects.add_videos(p["id"], [str(second)])["skipped"][0]["why"]

    def broken(path):
        raise RuntimeError("ffprobe says the file is damaged")
    monkeypatch.setattr(projects, "_probe", broken)
    assert "damaged" in projects.add_videos(p["id"], [str(second)])["skipped"][0]["why"]
    with pytest.raises(ValueError):
        projects.add_videos("missing-0000", [str(recording)])


def test_remove_video_keeps_the_source(recording, recycle_bin):
    before = fingerprint(recording)
    p = projects.create("Lessons")
    v = projects.add_videos(p["id"], [str(recording)])["added"][0]
    d = projects.video_dir(p["id"], v["id"])
    assert projects.remove_video(p["id"], v["id"]) is True
    assert recycle_bin == [d] and not d.exists()
    assert fingerprint(recording) == before
    assert projects.list_videos(p["id"]) == []
    assert projects.remove_video(p["id"], v["id"]) is True              # already gone: fine


def test_delete_project_keeps_the_sources(recording, recycle_bin):
    before = fingerprint(recording)
    p = projects.create("Lessons")
    projects.add_videos(p["id"], [str(recording)])
    assert projects.delete(p["id"]) is True
    assert recycle_bin == [config.projects_dir() / p["id"]]
    assert projects.get(p["id"]) is None
    assert fingerprint(recording) == before


def test_bad_ids_never_reach_the_disk():
    for bad in ("..", "../x", "a/b", "A-B", "", "x" * 60, "a\\b"):
        assert not projects.valid_id(bad)
        assert projects.get(bad) is None
        assert projects.video_state("ok-1234", bad) is None
        with pytest.raises(ValueError):
            projects.video_dir("ok-1234", bad)
        with pytest.raises(ValueError):
            projects.delete(bad)


def test_trash_outside_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "IS_WINDOWS", False)
    f = tmp_path / "old-folder"
    f.mkdir()
    (f / "a.txt").write_text("x", encoding="utf-8")
    assert projects.to_recycle_bin([f]) is True
    assert not f.exists()
    assert [x.name.startswith("old-folder-") for x in (config.data_dir() / "trash").iterdir()] == [True]


def test_video_status_follows_the_work_line(recording, monkeypatch):
    monkeypatch.setattr(jobs, "_KINDS", dict(jobs._KINDS))
    jobs.register("make", lambda *a: None, steps=6)
    p = projects.create("Lessons")
    v = projects.add_videos(p["id"], [str(recording)])["added"][0]
    j = jobs.enqueue("make", project=p["id"], video=v["id"], title=v["name"])
    st = projects.video_state(p["id"], v["id"])
    assert st["status"] == "queued" and st["job"]["position"] == 1
    jobs.LINE.waiting_for = "League of Legends.exe"
    assert projects.video_state(p["id"], v["id"])["status"] == "waiting"
    jobs.cancel(job_id=j["id"])
    st = projects.video_state(p["id"], v["id"])
    assert st["status"] == "stopped" and st["msg"] == "Stopped."


def test_set_video_and_outputs(recording):
    p = projects.create("Lessons")
    v = projects.add_videos(p["id"], [str(recording)])["added"][0]
    d = projects.video_dir(p["id"], v["id"])
    (d / "shorts").mkdir()
    (d / "shorts" / "short_01.mp4").write_bytes(b"x")
    (d / "shorts" / "short_02.part.mp4").write_bytes(b"x")              # half-written: not counted
    (d / "long.mp4").write_bytes(b"x")
    projects.set_video(p["id"], v["id"], {"status": "done"})
    st = projects.video_state(p["id"], v["id"])
    assert st["status"] == "done"
    assert st["outputs"] == {"shorts": 1, "long": True, "thumbs": 0, "texts": False}
    os.remove(recording)
    assert projects.video_state(p["id"], v["id"])["src_exists"] is False

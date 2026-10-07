"""The work line (SPEC §2.2): order, one job at a time, cancel, resume after a restart, no duplicates, pause while a
program runs (+ Run now), plain error messages, models unloaded after every job."""
import json
import logging
import threading
import time

import pytest

from broclips import jobs


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def line(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_which_running", lambda names: "")
    unloads = []
    monkeypatch.setattr(jobs, "_unload_models", lambda: unloads.append(1))
    w = jobs.WorkLine(tmp_path / "queue.json", idle_wait=0.02, pause_wait=0.02, check_every=0.0)
    w.unloads = unloads
    yield w
    w.stop(2)


@pytest.fixture
def kind(monkeypatch):
    """Register job kinds for one test only."""
    monkeypatch.setattr(jobs, "_KINDS", dict(jobs._KINDS))

    def add(name, fn, steps=1):
        jobs.register(name, fn, steps=steps, title=name)
    return add


def test_register(kind):
    kind("x", lambda *a: None, steps=6)
    assert jobs.registered("x") and not jobs.registered("nope")
    assert jobs._KINDS["x"]["steps"] == 6


def test_runs_in_order_one_at_a_time(line, kind):
    done, active, peak, lock = [], [0], [0], threading.Lock()

    def fn(job, progress, cancelled):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        done.append(job["args"]["n"])
        with lock:
            active[0] -= 1
    kind("t", fn)
    for n in range(4):
        line.enqueue("t", args={"n": n})
    line.start()
    assert wait_for(lambda: len(line.results) == 4)
    assert done == [0, 1, 2, 3]
    assert peak[0] == 1
    assert [r["status"] for r in line.results] == ["done"] * 4
    assert [r["msg"] for r in line.results] == ["✅ Done"] * 4
    assert len(line.unloads) == 4                       # local AI freed after every job


def test_result_message_comes_from_the_job(line, kind):
    kind("t", lambda job, progress, cancelled: "✅ 5 Shorts made")
    j = line.enqueue("t", project="p1", video="v1")
    line.start()
    assert wait_for(lambda: line.results)
    assert line.result_for("p1", "v1")["msg"] == "✅ 5 Shorts made"
    assert line.result_for("p1", "v1", kind="t")["id"] == j["id"]
    assert line.result_for("p1", "other") is None


def test_cancel_waiting_and_running(line, kind):
    started, ran = threading.Event(), []

    def slow(job, progress, cancelled):
        started.set()
        for i in range(500):
            progress(1, "Working", frac=i / 500)
            time.sleep(0.01)
        ran.append("slow finished")
    kind("slow", slow)
    kind("quick", lambda job, progress, cancelled: ran.append("quick"))
    a, b = line.enqueue("slow"), line.enqueue("quick")
    assert line.cancel(job_id=b["id"]) == 1            # waiting: it just leaves the line
    line.start()
    assert started.wait(5)
    assert line.cancel(job_id=a["id"]) == 1            # running: stops at its next progress()
    assert wait_for(lambda: not line.busy())
    assert ran == []
    assert {r["id"]: r["status"] for r in line.results} == {a["id"]: "stopped", b["id"]: "stopped"}


def test_cancelled_flag_without_progress(line, kind):
    started = threading.Event()

    def fn(job, progress, cancelled):
        started.set()
        while not cancelled():
            time.sleep(0.01)
        return "should not count as done"
    kind("t", fn)
    line.enqueue("t", project="p", video="v")
    line.start()
    assert started.wait(5)
    assert line.cancel(project="p", video="v") == 1
    assert wait_for(lambda: line.results)
    assert line.results[-1]["status"] == "stopped" and line.results[-1]["msg"] == "Stopped."


def test_identical_jobs_are_not_added_twice(line, kind):
    gate = threading.Event()
    kind("t", lambda job, progress, cancelled: gate.wait(5))
    a = line.enqueue("t", project="p", video="v", args={"again": False})
    b = line.enqueue("t", project="p", video="v", args={"again": False})
    c = line.enqueue("t", project="p", video="v", args={"again": True})
    assert a["id"] == b["id"] and c["id"] != a["id"]
    assert len(line.queue) == 2
    line.start()
    assert wait_for(lambda: line.running() is not None)
    assert line.enqueue("t", project="p", video="v", args={"again": False})["id"] == a["id"]   # the running one
    gate.set()


def test_unknown_kind(line, kind):
    with pytest.raises(jobs.UnknownKind):
        line.enqueue("not-a-kind")


def test_running_job_goes_first_after_a_restart(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(jobs, "_which_running", lambda names: "")
    monkeypatch.setattr(jobs, "_unload_models", lambda: None)
    path = tmp_path / "queue.json"
    gate = threading.Event()

    def blocking(job, progress, cancelled):
        progress(2, "Listening")
        gate.wait(5)                                  # the app "closes" while this runs
    kind("work", blocking)
    first = jobs.WorkLine(path, idle_wait=0.02, pause_wait=0.02)
    a = first.enqueue("work", project="p", video="v1")
    b = first.enqueue("work", project="p", video="v2")
    first.start()
    assert wait_for(lambda: first.running() is not None)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [j["id"] for j in saved["jobs"]] == [a["id"], b["id"]]
    assert saved["jobs"][0]["running"] is True

    order = []                                        # "the app starts again": a new line on the same file
    kind("work", lambda job, progress, cancelled: order.append((job["id"], job.get("restarts"))))
    second = jobs.WorkLine(path, idle_wait=0.02, pause_wait=0.02)
    second.start()
    assert wait_for(lambda: len(order) == 2)
    assert order == [(a["id"], 1), (b["id"], None)]
    first._stop.set()
    gate.set()
    first.stop(2)
    second.stop(2)


def test_job_that_keeps_crashing_the_app_is_dropped(tmp_path, kind):
    kind("work", lambda *a: None)
    path = tmp_path / "queue.json"
    bad = {"id": "abc", "kind": "work", "project": "p", "video": "v", "args": {}, "title": "Lesson",
           "running": True, "restarts": jobs.MAX_RESTARTS}
    path.write_text(json.dumps({"jobs": [bad], "results": []}), encoding="utf-8")
    w = jobs.WorkLine(path)
    w.load()
    assert w.queue == []
    assert w.results[-1]["status"] == "failed" and "taken out of the line" in w.results[-1]["msg"]


def test_waits_while_a_pause_program_runs(line, kind, monkeypatch):
    game = {"on": True}
    monkeypatch.setattr(jobs, "_which_running", lambda names: "League of Legends.exe" if game["on"] else "")
    ran = []
    kind("t", lambda job, progress, cancelled: ran.append(job["id"]))
    j = line.enqueue("t", project="p", video="v")
    line.start()
    assert wait_for(lambda: line.state()["waiting_for"] == "League of Legends.exe")
    time.sleep(0.1)
    assert ran == []
    st = line.state()
    assert st["text"] == "⏸ Waiting until League of Legends.exe closes — or ▶ Run now"
    assert line.job_for("p", "v")["status"] == "waiting"
    game["on"] = False                                 # the game is closed
    assert wait_for(lambda: ran == [j["id"]])


def test_run_now_starts_even_while_the_program_runs(line, kind, monkeypatch):
    monkeypatch.setattr(jobs, "_which_running", lambda names: "League of Legends.exe")
    ran = []

    def fn(job, progress, cancelled):
        progress(1, "Working")                         # a forced job is not paused again
        ran.append(job["id"])
    kind("t", fn)
    j = line.enqueue("t")
    line.start()
    assert wait_for(lambda: line.state()["waiting_for"])
    assert line.run_now() is True
    assert wait_for(lambda: ran == [j["id"]])
    assert line.results[-1]["status"] == "done"


def test_job_pauses_when_the_program_starts_during_it(line, kind, monkeypatch):
    prog = {"name": ""}
    monkeypatch.setattr(jobs, "_which_running", lambda names: prog["name"])
    calls = []

    def fn(job, progress, cancelled):
        calls.append(job["id"])
        if len(calls) == 1:
            prog["name"] = "game.exe"                  # the user starts a game while it works
            progress(3, "Finding moments")             # -> Paused: back to the front of the line
            raise AssertionError("progress() should have paused this job")
        return "done later"
    kind("t", fn, steps=6)
    a, b = line.enqueue("t"), line.enqueue("t", args={"other": 1})
    line.start()
    assert wait_for(lambda: line.state()["waiting_for"] == "game.exe")
    assert [j["id"] for j in line.queue] == [a["id"], b["id"]]
    prog["name"] = ""
    assert wait_for(lambda: not line.busy())
    assert calls == [a["id"], a["id"], b["id"]]
    assert [r["status"] for r in line.results] == ["done", "done"]


def test_errors_become_plain_messages(line, kind, caplog):
    def boom(job, progress, cancelled):
        raise ValueError("bad thing")

    def plain(job, progress, cancelled):
        raise jobs.JobError("The video file is gone.")
    kind("boom", boom)
    kind("plain", plain)
    kind("after", lambda *a: "still working")
    line.enqueue("boom")
    line.enqueue("plain")
    line.enqueue("after")
    with caplog.at_level(logging.INFO, logger="broclips"):
        line.start()
        assert wait_for(lambda: len(line.results) == 3)
    boom_r, plain_r, after_r = line.results
    assert boom_r["status"] == "failed" and "ValueError: bad thing" in boom_r["msg"] and "log folder" in boom_r["msg"]
    assert plain_r["status"] == "failed" and plain_r["msg"] == "⚠️ The video file is gone."
    assert after_r["msg"] == "still working"           # one failure never stops the line
    assert "Traceback" in caplog.text


def test_saved_job_of_unknown_kind_fails_politely(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_which_running", lambda names: "")
    monkeypatch.setattr(jobs, "_unload_models", lambda: None)
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"jobs": [{"id": "x1", "kind": "from-a-newer-version", "args": {}}]}), encoding="utf-8")
    w = jobs.WorkLine(path, idle_wait=0.02)
    w.start()
    assert wait_for(lambda: w.results)
    assert w.results[-1]["status"] == "failed" and "not in this version" in w.results[-1]["msg"]
    w.stop(2)


def test_progress_shows_in_state_and_in_the_video_folder(line, kind):
    from broclips import projects
    p = projects.create("Demo")
    vdir = projects.video_dir(p["id"], "lesson-abcd")
    vdir.mkdir(parents=True)
    gate, at = threading.Event(), threading.Event()

    def fn(job, progress, cancelled):
        progress(2, "Listening", "half way", 0.5)
        at.set()
        gate.wait(5)
    kind("make6", fn, steps=6)
    line.enqueue("make6", project=p["id"], video="lesson-abcd", title="Lesson 3")
    line.start()
    assert at.wait(5)
    st = line.state()
    assert st["current"]["pct"] == 25                  # (1 step done + half of step 2) of 6
    assert st["text"] == "⏳ Lesson 3 — Listening… 25 %"
    assert line.job_for(p["id"], "lesson-abcd")["status"] == "working"
    saved = json.loads((vdir / "progress.json").read_text(encoding="utf-8"))
    assert (saved["step"], saved["of"], saved["label"], saved["detail"]) == (2, 6, "Listening", "half way")
    gate.set()
    assert wait_for(lambda: not line.busy())
    assert line.state()["text"] == "💤 Ready"


def test_which_running_finds_real_programs():
    import os
    import psutil
    me = psutil.Process(os.getpid()).name()            # e.g. python.exe
    assert jobs._which_running([me.upper()]) == me.upper()
    assert jobs._which_running([me[:-4] if me.lower().endswith(".exe") else me])   # ".exe" is optional
    assert jobs._which_running(["surely-not-running-4f7a.exe"]) == ""
    assert jobs._which_running([]) == ""


def test_unload_models_frees_the_providers_that_were_used(monkeypatch):
    from broclips import providers
    from broclips.providers import fake
    monkeypatch.setattr(providers, "LLMS", {"fake": "fake:FakeLLM"})      # never touch a real Ollama in a test
    monkeypatch.setattr(providers, "STTS", {"fake": "fake:FakeSTT"})
    called = []
    monkeypatch.setattr(fake.FakeLLM, "unload", lambda self: called.append("llm"))
    monkeypatch.setattr(fake.FakeSTT, "unload", lambda self: called.append("stt"))
    jobs._unload_models()
    assert sorted(called) == ["llm", "stt"]

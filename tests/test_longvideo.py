"""longvideo.py: jump-cut planning (silences, padding, frame edges, off), source <-> long-video times, the filter
graph of one run, and a real render in two runs joined by concat copy (CPU encoder)."""
import json
import shutil
import subprocess

import pytest

from broclips import longvideo as L

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def w(s, e):
    return {"w": "x", "s": s, "e": e}


def test_plan_cuts_long_silences_and_keeps_padding():
    words = [w(0.5, 1.0), w(1.5, 2.0), w(5.0, 5.5), w(5.8, 6.0), w(20.0, 21.0)]
    r = L.plan(words, 30.0, silence_s=1.2, fps=100)
    assert r == [[0.0, 2.25], [4.75, 6.25], [19.75, 21.25]]
    assert L.plan(words, 30.0, jump_cuts=False) == [[0.0, 30.0]]
    assert L.plan([], 30.0) == [[0.0, 30.0]]                          # nothing said: keep everything
    # a short silence at the start / end is kept; a long one is cut
    assert L.plan([w(0.8, 1.0), w(28.9, 29.5)], 30.0, silence_s=1.2, fps=100)[0][0] == 0.0
    assert L.plan([w(3.0, 4.0)], 30.0, silence_s=1.2, fps=100) == [[2.75, 4.25]]
    # paddings that touch become one range; edges sit on frame boundaries
    r = L.plan([w(1.0, 2.0), w(2.4, 3.0)], 10.0, silence_s=0.3, pad=0.25, fps=30)
    assert len(r) == 1 and all(abs(x * 30 - round(x * 30)) < 0.05 for x in r[0])


def test_mapping_and_source_to_long_video_times():
    m = L.mapping([[0.0, 2.0], [5.0, 6.0], [10.0, 14.0]])
    assert m == [[0.0, 2.0, 0.0, 2.0], [5.0, 6.0, 2.0, 3.0], [10.0, 14.0, 3.0, 7.0]]
    assert L.to_out(m, 1.0) == 1.0 and L.to_out(m, 5.5) == 2.5
    assert L.to_out(m, 8.0) == 3.0                                     # inside a cut -> where it continues
    assert L.to_out(m, 99.0) == 7.0 and L.to_out([], 3.0) == 0.0


def test_one_run_filter_graph():
    g = L.filtergraph(2, [2.0, 3.5], 2, "delogo=x=2:y=2:w=10:h=10", 30)
    assert "[0:v]delogo=x=2:y=2:w=10:h=10,fps=30,setpts=PTS-STARTPTS[v0]" in g
    assert "[1:a:0][1:a:1]amix=inputs=2:normalize=0," in g               # several sound tracks are mixed
    assert "afade=t=in:d=0.08,afade=t=out:st=3.420:d=0.08[a1]" in g       # soft cut: never a click
    assert "concat=n=2:v=1:a=1[vc][a]" in g and g.endswith("[vc]format=yuv420p[v]")


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
def test_real_long_video_in_two_runs(tmp_path, monkeypatch):
    src = tmp_path / "rec.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=12", "-f", "lavfi",
                    "-i", "sine=f=330:d=12", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(src)], check=True, creationflags=NO_WINDOW)
    monkeypatch.setattr(L, "PER_RUN", 2)                               # 3 ranges -> 2 runs -> concat copy
    info = {"width": 320, "height": 180, "fps": 10, "n_audio": 1, "dur": 12.0}
    ranges = [[0.0, 2.0], [4.0, 6.0], [8.0, 11.0]]
    steps = []
    out = L.render(src, info, ranges, tmp_path / "long.mp4", boxes=[[100, 100, 300, 200]], enc="libx264",
                   step=lambda k, n: steps.append((k, n)))
    assert out.is_file() and steps == [(1, 2), (2, 2)]
    assert not (tmp_path / "_long_parts").exists()                     # the temporary runs are cleaned up
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration", "-of", "json",
                        str(out)], capture_output=True, text=True, creationflags=NO_WINDOW)
    j = json.loads(p.stdout)
    assert float(j["format"]["duration"]) == pytest.approx(7.0, abs=0.3)
    assert sorted(s["codec_type"] for s in j["streams"]) == ["audio", "video"]

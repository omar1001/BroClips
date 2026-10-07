"""shorts.py + camera.py: the three layouts' filter graphs, caption words on the Short's own timeline (cuts, fixes),
the subtitle file (one event per spoken word, Latin / Arabic fonts, title time), layouts and sizes, filter-graph
paths, the follow camera — and one real tiny render with ffmpeg (CPU encoder, no GPU)."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from broclips import camera, shorts

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")


def graph(layout, segs=((10.0, 25.0, 1.0),), blur="delogo=x=1650:y=950:w=190:h=50", w=1920, h=1080):
    return shorts.filtergraph([list(s) for s in segs], 1, layout, w, h, 60, blur, "656", "cache/ass/x.ass",
                              "assets/fonts", shorts.out_len([list(s) for s in segs]))


def test_follow_zoom_fit_filter_graphs():
    f = graph("follow")
    assert "crop=w=608:h=1080:x=656:y=0,scale=1080:1920:flags=lanczos" in f
    z = graph("zoom")
    assert "crop=w=810:h=1080:x=656:y=0,split=2[g0][b0]" in z and "[g0]scale=1080:1440" in z
    assert "overlay=0:240" in z and "boxblur" in z
    t = graph("fit")
    assert "[f0]scale=1080:608" in t and "overlay=0:560" in t and "crop=w=" not in t
    for g in (f, z, t):
        assert g.index("delogo") < g.index("setpts")                 # the blur boxes come FIRST
        assert "ass=filename=cache/ass/x.ass:fontsdir=assets/fonts" in g
        assert "loudnorm=I=-14:TP=-1.5:LRA=11" in g and "fps=60" in g and "format=yuv420p[v]" in g
        assert "afade=t=out:st=14.650:d=0.35" in g                     # 15 s Short: 0.35 s fade at its end


def test_two_parts_get_a_soft_join_and_no_blur_when_no_boxes():
    g = graph("fit", segs=((10.0, 20.0, 1.0), (60.0, 70.0, 1.0)), blur="")
    assert "concat=n=2:v=1:a=1" in g and "delogo" not in g
    assert "afade=t=out:st=9.920:d=0.08[a0]" in g and "afade=t=in:d=0.08[a1]" in g
    assert "[0:v]setpts" in g


def test_caption_words_follow_the_shorts_timeline_and_fixes():
    words = [{"w": "one", "s": 10.2, "e": 10.5}, {"w": "two", "s": 10.6, "e": 10.9}, {"w": "gone", "s": 30.0, "e": 30.4},
             {"w": "three", "s": 60.1, "e": 60.5}, {"w": "four", "s": 60.6, "e": 61.0}]
    segs = [[10.0, 20.0, 1.0], [60.0, 70.0, 1.0]]
    cw = shorts.caption_words(words, segs)
    assert [w["w"] for w in cw] == ["one", "two", "three", "four"]            # cut bits dropped
    assert cw[0]["s"] == pytest.approx(0.2) and cw[2]["s"] == pytest.approx(10.1)
    assert [w["part"] for w in cw] == [0, 0, 1, 1] and cw[2]["src_s"] == 60.1
    fixed = shorts.caption_words(words, segs, [{"a": 10.0, "b": 11.0, "text": "uno dos tres"}])
    assert [w["w"] for w in fixed][:3] == ["uno", "dos", "tres"] and fixed[3]["w"] == "three"
    removed = shorts.caption_words(words, segs, [{"a": 60.0, "b": 61.5, "text": ""}])
    assert [w["w"] for w in removed] == ["one", "two"]


def test_subtitle_file_highlights_each_word_and_picks_fonts_by_script():
    from broclips import transcript as T
    words = [{"w": w, "s": 0.5 + i * 0.4, "e": 0.8 + i * 0.4, "part": 0} for i, w in enumerate("hello my friends".split())]
    words += [{"w": w, "s": 3.0 + i * 0.4, "e": 3.3 + i * 0.4, "part": 0} for i, w in enumerate("يا جماعة اهلا".split())]
    phr = T.phrases(words)
    ass = shorts.ass_file("follow", 1920, 1080, 10.0, phr, "My {LOUD} title")
    events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue: 0,")]
    assert len(events) == 6                                      # one event per spoken word
    assert all(shorts.CAP_NOW in e for e in events)
    assert sum(",CapLat," in e for e in events) == 3 and sum(",CapAr," in e for e in events) == 3
    hook = [ln for ln in ass.splitlines() if ln.startswith("Dialogue: 1,")][0]
    assert "0:00:04.50,HookLat" in hook and "{LOUD}" not in hook and hook.endswith("My  title")
    assert "Style: CapLat,Montserrat ExtraBold" in ass and "Style: CapAr,Lalezar" in ass
    fit = shorts.ass_file("fit", 1920, 1080, 10.0, phr, "عنوان")
    assert "0:00:10.00,HookAr" in fit                            # fit: the title stays the whole Short
    assert "Style: CapLat,Montserrat ExtraBold,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0," \
           "1,8,2,8,90,90,1194" in fit                          # fit: captions right under the picture


def test_layouts_sizes_and_paths(tmp_path):
    assert shorts.layout_for({"video_type": "screen"}) == "fit"
    assert shorts.layout_for({"video_type": "gameplay"}) == "follow"
    assert shorts.layout_for({"video_type": "gameplay", "layout": "zoom"}) == "zoom"
    assert shorts.layout_for({"layout": "zoom"}, {"layout": "fit"}) == "fit"
    assert shorts.window_w("follow", 1920, 1080) == 608 and shorts.window_w("zoom", 1280, 720) == 540
    assert shorts.window_w("follow", 1080, 1920) is None                       # a tall video uses "fit"
    assert shorts.fit_box(1920, 1080) == (1080, 608, 0, 560)
    assert shorts.fit_box(1080, 1920) == (1080, 1920, 0, 0)
    assert shorts.out_fps(59.94) == 60 and shorts.out_fps(29.97) == 30 and shorts.out_fps(0) == 30
    (tmp_path / "cache").mkdir()
    assert shorts.fg_path(tmp_path / "cache" / "a.ass", tmp_path) == "cache/a.ass"
    odd = shorts.fg_path(tmp_path / "x, y.ass", tmp_path)
    assert odd.startswith("'") and "\\:" in odd
    assert shorts.ass_time(3725.456) == "1:02:05.46"


def test_camera_path_moves_to_the_action_and_writes_an_expression():
    E = np.zeros((14, 384), np.float32)
    E[:2, 40:60] = 50                       # first the action is on the left ...
    E[2:, 320:340] = 50                     # ... then on the right (the window crosses at <= 260 px/s)
    p = camera.path(E, 608, 1920)
    assert p[0] < 300 and all(0 <= x <= 1920 - 608 for x in p)
    assert p[-1] <= 1600 and p[-1] + 608 >= 1695           # it ends where the action (1600-1700) is inside
    steps = np.diff(p)
    assert np.all(np.abs(steps) <= camera.MAX_SPEED + 1e-6)              # never faster than the speed limit
    pts = [(0.0, 300.0), (2.0, 300.0), (4.0, 800.0)]
    ex = camera.expr(pts, 1920, 608)
    assert ex.startswith("300+500*clip((t-2.000)/2.000\\,0\\,1)")
    assert camera.expr([(0.0, 301.0), (5.0, 302.0)]) == "300"            # still -> a plain (even) number
    m = camera.mask(1920, 1080, [[1650, 950, 190, 50]])
    assert not m[200, 350] and m[10, 10]


@needs_ffmpeg
def test_a_real_short_renders_with_the_cpu_encoder(tmp_path):
    src = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=8", "-f", "lavfi",
                    "-i", "sine=f=440:d=8", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(src)], check=True, creationflags=NO_WINDOW)
    info = {"width": 320, "height": 180, "fps": 10, "n_audio": 1, "dur": 8.0}
    words = [{"w": w, "s": 1.0 + i * 0.5, "e": 1.4 + i * 0.5} for i, w in enumerate("this is a tiny test short".split())]
    short = {"segments": [[1.0, 4.0, 1.0], [5.0, 7.0, 1.0]], "hook": "Tiny test", "caption_fixes": []}
    out = tmp_path / "shorts" / "short_01.mp4"
    res = shorts.render(src, info, short, words, out, tmp_path / "data" / "cache" / "ass" / "t.ass", "follow",
                        boxes=[[1600, 900, 200, 100]], enc="libx264")
    assert out.is_file() and not list(out.parent.glob("*.part*"))
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                        "-of", "json", str(out)], capture_output=True, text=True, creationflags=NO_WINDOW)
    j = json.loads(p.stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert float(j["format"]["duration"]) == pytest.approx(5.0, abs=0.25)
    assert any(s["codec_type"] == "audio" for s in j["streams"])
    assert res["layout"] == "follow" and res["dur"] == 5.0
    assert [p["text"] for p in res["phrases"]] == ["this is a tiny test", "short"] or \
        " ".join(p["text"] for p in res["phrases"]) == "this is a tiny test short"
    assert all(p["a"] < p["b"] for p in res["phrases"])

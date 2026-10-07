"""moments.py: wishes, line ids -> times (also the ids models garble), edges, merging picks of one scene, dropping
duplicates, hints first, the no-AI fallback, the AI log — with a scripted fake AI (no network, no GPU)."""
import json

from broclips import moments as M
from broclips import transcript as T


class ScriptLLM:
    """A fake Thinking AI whose answers come from a function of (system, user, schema)."""
    name, model, vision, local = "fake", "script", False, True

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def chat_json(self, system, user, schema, images=None, temperature=0.2, max_tokens=1500):
        self.calls.append({"system": system, "user": user, "schema": schema})
        return self.fn(system, user, schema)

    def unload(self):
        pass


def make_lines(n=60, every=5.0, talk=4.0):
    """n one-sentence lines: line i is said from i*every to i*every+talk."""
    words = []
    for i in range(n):
        for k, w in enumerate(f"sentence {i} has some words here.".split()):
            s = i * every + k * talk / 6
            words.append({"w": w, "s": round(s, 2), "e": round(s + talk / 6 - 0.05, 2), "p": 0.9})
    return T.make_lines(words)


def brain(fn=None, project=None, wishes="", log_path=None):
    p = {"prompt": "funny gaming talk", "video_type": "gameplay", "platforms": ["shorts", "tiktok"]}
    p.update(project or {})
    return M.Brain(ScriptLLM(fn or (lambda *a: None)), p, language="en", wishes=wishes, log_path=log_path)


def pick(start, end, score, title, payoff=None):
    return {"start": start, "end": end, "payoff": payoff or end, "type": "funny", "why": f"why {title}",
            "title": title, "hook": f"hook {title}", "score": score}


def test_parse_wishes():
    hints, instr = M.parse_wishes("5:30 the joke about the bug\nmake them funny\nminute 12 the fight; "
                                  "1:02:03 late story\nPython 3.12 tips please\n2:00 a, 2:45 b")
    assert hints == [(330, "5:30 the joke about the bug"), (720, "minute 12 the fight"), (3723, "1:02:03 late story"),
                     (120, "2:00 a"), (165, "2:45 b")]
    assert instr == "make them funny\nPython 3.12 tips please"
    assert M.parse_wishes("") == ([], "")
    assert M.parse_wishes("الدقيقة 3 القصة")[0] == [(180, "الدقيقة 3 القصة")]


def test_line_index_reads_ids_times_and_glued_times():
    lines = make_lines(150)                           # 0 .. 750 s
    by_id = {ln["id"]: i for i, ln in enumerate(lines)}
    assert M.line_index(lines, "L012", by_id) == 11
    assert M.line_index(lines, "12", by_id) == 11 and M.line_index(lines, "L12", by_id) == 11
    assert M.line_index(lines, "02:05", by_id) == 25                    # a time instead of an id
    assert lines[M.line_index(lines, "L0426", by_id)]["s"] == 265.0       # 04:26 glued into an id
    assert lines[M.line_index(lines, "L1026", by_id)]["s"] == 625.0      # 10:26 (no line L1026 exists)
    assert M.line_index(lines, "nonsense", by_id) is None
    assert M.find_quote(lines, "sentence 77 has some", near=0) == 77


def test_edges_grow_pull_back_and_keep_the_laugh():
    lines = make_lines(40)
    lines[10]["text"] = "and then it happened"            # continues the line before -> start one line earlier
    s, e, i0, i1 = M.edges(lines, 10, 11, spikes=[62], min_s=15, max_s=90)
    assert i0 == 9 and s == round(lines[9]["s"] - 0.3, 2)
    assert e - s >= 15 and i1 > 11                        # grown to the minimum length
    s2, e2, _, _ = M.edges(lines, 20, 23, spikes=[120.0, 122.0], min_s=15, max_s=90)
    assert e2 == 123.4                                    # the shout right after the payoff stays in
    s3, e3, _, _ = M.edges(lines, 0, 39, spikes=[], min_s=15, max_s=90, total_s=150)
    assert e3 <= 150


def test_find_maps_ids_merges_one_scene_and_drops_duplicates(tmp_path):
    def fn(system, user, schema):
        if "L003" in user and "THE CREATOR REMEMBERS" not in user.upper():
            return {"notes": "a funny bit", "moments": [pick("L003", "L005", 8, "T1"), pick("L006", "L007", 6, "T2"),
                                                       pick("L003", "L004", 5, "dup"), pick("L999x", "L998x", 9, "bad")]}
        return {"notes": "nothing", "moments": []}
    b = brain(fn, log_path=tmp_path / "ai_log.jsonl")
    lines = make_lines(60)
    got = M.find(b, lines, spikes=[], total_s=300)
    assert len(got) == 1                                  # T1 + T2 are one scene (0.5 s apart), "dup" overlaps
    m = got[0]
    assert m["title"] == "T1" and m["hook"] == "hook T1" and m["score"] == 8
    assert m["s"] == round(lines[2]["s"] - 0.3, 2) and m["e"] >= lines[6]["e"]
    assert m["segments"] == [[m["s"], m["e"], 1.0]] and "why T1" in m["why"] and "why T2" in m["why"]
    notes = [json.loads(x) for x in (tmp_path / "ai_log.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {n["k"] for n in notes}
    assert {"read", "think", "idea", "drop", "merge", "step"} <= kinds
    assert any("do not exist" in n["text"] for n in notes)
    # every question carries the project, the platforms and the language rule
    sys0 = b.llm.calls[0]["system"]
    assert "funny gaming talk" in sys0 and "TikTok" in sys0 and "English" in sys0 and "15-90 seconds" in sys0


def test_hints_are_asked_first_and_rank_higher():
    seen = []

    def fn(system, user, schema):
        seen.append(user)
        if "remembers a good moment around 3:20" in user:
            return {"notes": "found it", "moments": [pick("L041", "L043", 5, "hinted one")]}
        if "L003" in user:
            return {"notes": "ok", "moments": [pick("L003", "L005", 9, "great one")]}
        return {"notes": "", "moments": []}
    b = brain(fn, wishes="3:20 the bug story\nkeep it short")
    got = M.find(b, make_lines(60), spikes=[], total_s=300)
    assert "remembers a good moment around 3:20" in seen[0]
    assert [m["title"] for m in got] == ["hinted one", "great one"] and got[0]["hinted"]
    assert "keep it short" in b.llm.calls[0]["system"]          # a wish without a time = an instruction


def test_teaching_projects_get_longer_moments_and_arabic_rule():
    assert M.limits({"video_type": "screen"}) == (30, 120)
    assert M.limits({"video_type": "camera", "prompt": "Short Python lessons for beginners"}) == (30, 120)
    assert M.limits({"video_type": "gameplay", "prompt": "funny moments"}) == (15, 90)
    assert "dialect" in M.lang_rule("ar") and "English" in M.lang_rule("en")
    b = M.Brain(ScriptLLM(lambda *a: None), {"video_type": "screen", "prompt": "lessons"}, language="ar")
    assert "lesson / tutorial" in b.system(M._scan_system(b)) and "30-120 seconds" in M._scan_system(b)


def test_no_usable_answer_falls_back_to_plain_rules(tmp_path):
    b = brain(lambda *a: None, log_path=tmp_path / "log.jsonl")
    lines = make_lines(60)
    assert M.find(b, lines, spikes=[], total_s=300) == []
    got = M.fallback(b, lines, [], 300, 3)
    assert 1 <= len(got) <= 3 and all(m["src"] == "rules" and M.MIN_KEEP_S <= m["e"] - m["s"] <= 90 for m in got)
    assert all(M.overlap(a, c) <= 0.5 for a in got for c in got if a is not c)
    assert "simple rules" in (tmp_path / "log.jsonl").read_text(encoding="utf-8")


def test_clean_text_and_hook():
    assert M.clean_text("Great {LOUD} moment L045\nsecond line") == "Great moment"
    assert M.clean_text("خناقة العجول واللзقة") == "خناقة العجول واللزقة" and M.fix_script("Привет") == "Привет"
    assert M.hook_text("one two three four five six seven eight nine ten eleven") == \
        "one two three four five six seven eight nine ten"


def test_two_separate_tips_stay_two_shorts():
    """Live test 2026-10-07: two tips grown to the 30 s lesson minimum touched and were joined into one Short."""
    words = []
    for i in range(24):                                   # 4 slides of 6 lines; 3 s of silence after slides 2 and 4
        t0 = i * 2.6 + (3.0 if i >= 12 else 0.0)
        for k, w in enumerate(f"slide {i // 6} line {i} words.".split()):
            words.append({"w": w, "s": round(t0 + k * 0.4, 2), "e": round(t0 + k * 0.4 + 0.35, 2), "p": 0.9})
    lines = T.make_lines(words)

    def fn(system, user, schema):
        return {"notes": "two tips", "moments": [pick("L008", "L011", 9, "tip one"), pick("L019", "L022", 8, "tip two")]}
    b = brain(fn, project={"video_type": "screen", "prompt": "Python lessons"})
    got = M.find(b, lines, spikes=[], total_s=70)
    assert sorted(m["title"] for m in got) == ["tip one", "tip two"]
    assert all(M.overlap(a, c) < 0.5 for a in got for c in got if a is not c)

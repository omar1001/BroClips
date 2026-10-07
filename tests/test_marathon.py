"""marathon.py: pairing, the tournament (both orders, rounds + final, ranking by wins), the answer cache (a stopped
job asks nothing twice), the whole search with a deterministic fake judge, the question estimate."""
import re

from broclips import marathon as MA
from broclips import moments as M
from test_moments import ScriptLLM, make_lines


def power_judge(system, user, schema, power=lambda n: n, step=4):
    """Clip strength = power(highest 'sentence N' in it); the stronger one wins in both orders."""
    if "winner" in (schema.get("properties") or {}):
        a, b = user.split("CLIP B", 1)
        pa = max(power(int(x)) for x in re.findall(r"sentence (\d+)", a))
        pb = max(power(int(x)) for x in re.findall(r"sentence (\d+)", b))
        return {"reason": "stronger", "winner": "A" if pa > pb else "B"}
    if "best_line" in str(schema):                   # the finder: list every line pair it sees
        ids = re.findall(r"^(L\d{3})", user, re.M)
        return {"notes": "a part", "moments": [{"start": ids[k], "end": ids[k + 2], "kind": "funny", "what": "w",
                                                "best_line": "x"} for k in range(0, len(ids) - 2, step)][:8]}
    if "hook" in (schema.get("properties") or {}):
        return {"type": "funny", "why": "it is strong", "title": "named", "hook": "named hook"}
    return None


def brain(fn, tmp_path=None):
    p = {"prompt": "talk", "video_type": "gameplay", "platforms": ["tiktok"]}
    return M.Brain(ScriptLLM(fn), p, language="en", log_path=(tmp_path / "log.jsonl") if tmp_path else None)


def cand(i, lines):
    ln = lines[i]
    return {"s": ln["s"], "e": ln["e"] + 15, "i0": i, "i1": min(i + 3, len(lines) - 1), "title": f"c{i}", "rank": 0.5,
            "segments": [[ln["s"], ln["e"] + 15, 1.0]]}


def test_pairs_never_repeat_a_match():
    assert MA.pairs([0, 1, 2, 3, 4, 5], {}) == [(0, 1), (2, 3), (4, 5)]
    assert MA.pairs([0, 1, 2, 3, 4, 5], {(0, 1): 1}) == [(0, 2), (1, 3), (4, 5)]
    assert MA.pairs([3], {}) == []


def test_tournament_ranks_by_wins_in_both_orders(tmp_path):
    lines = make_lines(200)
    pool = [cand(i, lines) for i in range(0, 200, 10)]            # 20 candidates: rounds, then the best 16 meet
    b = brain(power_judge, tmp_path)
    memo = MA.Memo(b, tmp_path / "cache.json")
    out = MA.tournament(b, memo, pool, lines)
    assert len(out) == 16
    order = [c["i0"] for c in out]
    assert order == sorted(order, reverse=True)                   # the strongest clip wins
    assert out[0]["marathon"] == {"rank": 1, "won": 30, "of": 30}
    assert out[-1]["marathon"]["won"] == 0
    asked = len(b.llm.calls)
    assert asked % 2 == 0                                          # every match = 2 questions (both orders)
    log = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert "→" in log and '"k": "match"' in log and "final standings" in log
    assert "in both orders" in log


def test_a_judge_that_always_says_A_gives_ties(tmp_path):
    lines = make_lines(60)
    pool = [cand(i, lines) for i in range(0, 50, 10)]
    b = brain(lambda s, u, sc: {"reason": "first", "winner": "A"}, tmp_path)
    out = MA.tournament(b, MA.Memo(b), pool, lines)
    assert len(out) == 5 and all(c["marathon"]["won"] == 4 for c in out)       # 1 of 2 per match = toss-ups
    assert "in 0 of 10 matches" in (tmp_path / "log.jsonl").read_text(encoding="utf-8")


def test_memo_answers_from_the_file_after_a_stop(tmp_path):
    b = brain(power_judge)
    memo = MA.Memo(b, tmp_path / "marathon_cache.json")
    q = ("sys", "CLIP A (5 s):\nsentence 3\n\nCLIP B (5 s):\nsentence 9", MA.PAIR_SCHEMA, 160)
    assert memo.ask(*q)["winner"] == "B"
    assert memo.ask(*q)["winner"] == "B" and len(b.llm.calls) == 1        # asked once
    memo.save()
    b2 = brain(lambda *a: (_ for _ in ()).throw(AssertionError("asked again")))
    assert MA.Memo(b2, tmp_path / "marathon_cache.json").ask(*q)["winner"] == "B"


def test_search_lists_pools_ranks_and_names_the_winners(tmp_path):
    lines = make_lines(60)                                        # 5 minutes of talk

    def judge(system, user, schema):                              # strong clips spread over the video
        return power_judge(system, user, schema, power=lambda n: (n * 37) % 61)
    b = brain(judge, tmp_path)
    normal = [M.make_moment(lines, 30, 33, {"title": "from search", "hook": "h", "score": 7}, [], b, 300)]
    got = MA.search(b, lines, [], 300, normal, cache_path=tmp_path / "marathon_cache.json")
    assert len(got) >= MA.MIN_WINNERS
    assert all(c.get("title") and c.get("hook") and c.get("marathon") for c in got)
    assert all(M.overlap(a, c) <= 0.5 for a in got for c in got if a is not c)
    ranks = [c["marathon"]["rank"] for c in got]
    assert ranks == sorted(ranks)
    assert (tmp_path / "marathon_cache.json").is_file()
    n_first = len(b.llm.calls)
    b2 = brain(judge, tmp_path)                                  # the same search again: everything from the file
    again = MA.search(b2, lines, [], 300, normal, cache_path=tmp_path / "marathon_cache.json")
    assert len(b2.llm.calls) == 0 and [c["s"] for c in again] == [c["s"] for c in got]
    assert n_first > 20


def test_estimate_grows_with_the_video():
    short, mid, long = MA.estimate(89), MA.estimate(13 * 60 + 39), MA.estimate(3600)
    assert short < mid <= long
    assert short < 150 and 300 < mid < 800


def test_winning_neighbours_become_one_short(tmp_path):
    lines = make_lines(60)
    def judge(system, user, schema):                              # ideas 1 s apart; the strongest are the last ones
        return power_judge(system, user, schema, step=3)
    b = brain(judge, tmp_path)
    got = MA.search(b, lines, [], 300, [], cache_path=tmp_path / "c.json")
    assert 1 <= len(got) < MA.MIN_WINNERS
    assert "become one Short" in (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert all(c["e"] - c["s"] <= 90 for c in got)

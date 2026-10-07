"""Marathon mode (Omar's idea, SPEC §6.4): instead of one big answer the AI works in MANY small rounds. Port of the
generic parts of LoL Clips marathon.py (no game facts, no eyes, no score corner):

  1. find wide    in 2-minute parts the AI only LISTS candidate moments — it never judges          find_wide()
  2. pool         every idea becomes a complete candidate (the same moment only once); the
                  normal search's moments join in (they carry the creator's hints)                  build_pool()
  3. tournament   two candidates at a time — "which makes the better short for this channel?" —
                  every pair asked in BOTH orders; a few rounds against random opponents, then the
                  best 16 all meet. Ranking = wins, never a 1-10 score                              tournament()
  4. winners      finalists that won at least half of their final questions (at least 3) become
                  the moments; winners that are neighbours in the video are joined                 search()

Measured in LoL Clips (a 12B local model): it cannot SCORE (8/10 for everything) but it can COMPARE — the same winner
in both orders in ~78 % of the final's matches. The first rounds are only a rough filter; the final ranks.
Every answer is kept in <video>/marathon_cache.json, so a paused or stopped job asks nothing twice."""
import hashlib
import math
import random

from . import moments as M
from . import transcript as T
from .util import load_json, save_json

WIDE_WINDOW_S, WIDE_STRIDE_S = 120, 90     # the finder lists ideas in 2-minute parts, one every 90 s
POOL = 40                                  # at most this many candidates enter the tournament
ROUNDS = 6                                 # first rounds: everybody plays this many random opponents...
FINALISTS = 16                             # ...then the best ones all meet each other
MIN_WINNERS = 3

LIST_SCHEMA = {"type": "object", "properties": {"notes": {"type": "string"}, "moments": {
    "type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
        "start": {"type": "string"}, "end": {"type": "string"}, "kind": {"type": "string", "enum": M.KINDS},
        "what": {"type": "string"}, "best_line": {"type": "string"}},
        "required": ["start", "end", "kind", "what", "best_line"]}}}, "required": ["notes", "moments"]}
PAIR_SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"},
                                                "winner": {"type": "string", "enum": ["A", "B"]}},
               "required": ["reason", "winner"]}
NAME_SCHEMA = {"type": "object", "properties": {"type": {"type": "string", "enum": M.KINDS}, "why": {"type": "string"},
                                                "title": {"type": "string"}, "hook": {"type": "string"}},
               "required": ["type", "why", "title", "hook"]}


def _list_system(brain):
    return f"""You help a creator who makes short vertical videos ({brain.short_platforms()}) from their recordings.
The transcript is automatic speech-to-text (spelling mistakes are normal - judge the meaning). {{LOUD}} = the voice got
suddenly loud. {brain.good()}
Your ONLY job now is to LIST candidates. Do not judge strictly and do not reject: a later step compares all candidates
and throws away the weak ones. List EVERY moment in this part that could become a short - something funny, surprising,
useful, emotional or absurd, a strong reaction, a good line, a little story, a tip - even if it is short or you are
not sure. Up to 8 moments.
For each: start and end = line ids exactly as written (like L045); kind; what = one simple English sentence;
best_line = the best 3-8 words said there, copied exactly from the transcript.
notes: FIRST one simple English sentence about what happens in this part."""


def _judge_system(brain):
    return f"""You compare two candidate clips for short vertical videos ({brain.short_platforms()}). Both come from the
same recording. The transcripts are automatic speech-to-text (spelling errors are normal - judge the meaning).
Think like a viewer who never saw the video: they scroll away in 2 seconds unless the first line grabs them, and they
leave if there is no payoff (a punchline, a twist, an answer, a useful point, a reaction with a clear cause).
Do not prefer a clip just because it is shorter or longer. Which clip makes the better short for this channel?
reason: FIRST one short, simple English sentence that compares the two. winner: "A" or "B"."""


def _name_system(brain):
    return f"""You name ONE chosen moment of a creator's recording that becomes a short vertical video
({brain.short_platforms()}). The transcript is automatic speech-to-text (spelling errors are normal).
title = a catchy title, at most 8 words. hook = the text shown on top of the short, at most 8 words.
{M.lang_rule(brain.language)} why = one short, simple English sentence: why it works as a short."""


# ---------- the answer cache (a paused job continues where it stopped) ----------
class Memo:
    def __init__(self, brain, path=None):
        self.brain, self.path, self.new = brain, path, 0
        self.data = (load_json(path) if path else None) or {}

    def ask(self, system, user, schema, max_tokens, temperature=0.2):
        key = hashlib.sha1(f"{system}\0{user}\0{temperature}".encode("utf-8")).hexdigest()
        if key in self.data:
            return self.data[key]
        a = self.brain.ask(system, user, schema, max_tokens=max_tokens, temperature=temperature)
        if a is not None:
            self.data[key] = a
            self.new += 1
            if self.new >= 10:
                self.save()
        return a

    def save(self):
        if self.path and self.new:
            save_json(self.data, self.path)
            self.new = 0


def estimate(total_s, normal_questions=None):
    """About how many AI questions Marathon mode asks for a video this long (the UI warns cloud users)."""
    windows = len(range(0, int(total_s) + 1, WIDE_STRIDE_S))
    n = min(POOL, 4 + 4 * windows)
    fin = min(n, FINALISTS)
    rounds = min(ROUNDS, n - 1) if n > fin else 0
    tournament = rounds * (n // 2) * 2 + fin * (fin - 1)
    normal = normal_questions if normal_questions is not None else math.ceil(total_s / M.STRIDE_S) + 3
    return int(windows + tournament + normal + 6)


def _len(c):
    return M.length(c)


def _text(c, lines):
    out = []
    for a, b in M.pieces(c):
        out += [ln["text"] for ln in lines if a - 0.5 <= ln["s"] < b]
    return "\n".join(out)


def _name(c):
    return c.get("title") or c.get("best") or ""


def _label(c):
    return f"{T.mmss(c['s'])} «{M.clean_text(_name(c), 40)}»"


# ---------- 1. find wide ----------
def find_wide(brain, memo, lines, spikes, total_s):
    """The finder only LISTS, in 2-minute parts. -> [{"i0", "i1", "kind", "what", "best"}]"""
    by_id = {ln["id"]: i for i, ln in enumerate(lines)}
    starts = list(range(0, int(total_s) + 1, WIDE_STRIDE_S))
    found, system = [], brain.system(_list_system(brain))
    for k, t in enumerate(starts, 1):
        brain.step(f"listing ideas, part {k} of {len(starts)}", 0.25 * k / len(starts))
        win = [ln for ln in lines if t <= ln["s"] < t + WIDE_WINDOW_S]
        if len(win) < 3:
            continue
        j = memo.ask(system, "TRANSCRIPT PART:\n" + M.window_text(win, spikes), LIST_SCHEMA, 1400)
        lo, hi = win[0]["s"] - 20, win[-1]["e"] + 20
        got = 0
        for m in (j or {}).get("moments") or []:
            a = M.line_index(lines, m.get("start"), by_id)
            b = M.line_index(lines, m.get("end"), by_id, end=True)
            # ids from outside this part are wrong ids (models glue times into ids): trust the quoted words then
            if a is None or b is None or not (lo <= lines[a]["s"] <= hi and lo <= lines[b]["s"] <= hi):
                a = b = M.find_quote(lines, m.get("best_line"), near=t + WIDE_WINDOW_S / 2, lo=lo, hi=hi)
                if a is None:
                    continue
            a, b = min(a, b), max(a, b)
            while b > a and lines[b]["e"] - lines[a]["s"] > brain.max_s:
                b -= 1
            found.append({"i0": a, "i1": b, "kind": m.get("kind") or "other", "what": M.clean_text(m.get("what"), 200),
                          "best": M.clean_text(m.get("best_line"), 80)})
            got += 1
        brain.note("read", f"listing every idea in {T.mmss(t)}–{T.mmss(min(total_s, t + WIDE_WINDOW_S))} "
                           f"({len(win)} lines of talk): {got} found" if j else
                           f"no usable answer for {T.mmss(t)}–{T.mmss(min(total_s, t + WIDE_WINDOW_S))}", at=t)
    return found


# ---------- 2. pool ----------
def build_pool(brain, found, normal, lines, spikes, total_s=None):
    """Listed ideas -> complete candidates (plain edge rules). Ideas that are the SAME moment (more than half shared)
    become one candidate of at most half the longest Short; neighbours stay separate, so every bit is judged on its
    own (winning neighbours are joined afterwards). The normal search's moments join (theirs wins where both found
    the same moment). At most POOL, the most-listed first."""
    bit = max(brain.min_s, brain.max_s // 2)
    wide = []
    for f in found:
        s, e, i0, i1 = M.edges(lines, f["i0"], f["i1"], spikes, brain.min_s, brain.max_s, total_s=total_s)
        if e - s > brain.max_s or e - s < M.MIN_KEEP_S:
            continue
        wide.append({"s": s, "e": e, "i0": i0, "i1": i1, "core": [lines[f["i0"]]["s"], lines[f["i1"]]["e"]],
                     "type": f["kind"] if f["kind"] in M.KINDS else "other",
                     "whys": [f["what"]] if f["what"] else [], "best": f["best"], "votes": 1})
    scenes = []
    for c in sorted(wide, key=lambda x: x["s"]):
        p = next((p for p in scenes if M.overlap(c, p) > 0.5), None)
        if p is None:
            scenes.append(c)
            continue
        p["votes"] += 1
        p["whys"] += [w for w in c["whys"] if w not in p["whys"]]
        if max(p["e"], c["e"]) - min(p["s"], c["s"]) <= bit:
            p["s"], p["e"] = min(p["s"], c["s"]), max(p["e"], c["e"])
            p["i0"], p["i1"] = min(p["i0"], c["i0"]), max(p["i1"], c["i1"])
            p["core"] = [min(p["core"][0], c["core"][0]), max(p["core"][1], c["core"][1])]
    pool = [dict(m, src="search") for m in normal]
    extra = []
    for sc in scenes:
        if any(M.overlap(sc, m) > 0.5 for m in pool):
            continue
        extra.append({"s": sc["s"], "e": sc["e"], "i0": sc["i0"], "i1": sc["i1"], "type": sc["type"], "core": sc["core"],
                      "start": lines[sc["i0"]]["id"], "end": lines[sc["i1"]]["id"], "payoff": lines[sc["i1"]]["id"],
                      "why": " / ".join(sc["whys"][:3]), "best": sc["best"], "votes": sc["votes"], "src": "list",
                      "title": "", "hook": "", "score": 5, "rank": 0.5, "hinted": False,
                      "loud": any(sc["s"] <= k <= sc["e"] for k in spikes or ()),
                      "segments": [[sc["s"], sc["e"], 1.0]]})
    extra.sort(key=lambda c: -c["votes"])          # too many: the scenes listed most often stay
    pool += extra[: max(0, POOL - len(pool))]
    for c in pool:
        if c["src"] == "list":
            brain.note("idea", f"{c['type']} {T.mmss(c['s'])}–{T.mmss(c['e'])}: {c['why']}", at=c["s"],
                       title=c.get("best"))
    return pool


# ---------- 3. tournament ----------
class Arena:
    """Matches between texts. A match = the same question in both orders (models lean a little towards the first
    one), so a clear win is a win in both."""

    def __init__(self, memo, system, blocks, label="CLIP"):
        self.memo, self.system, self.blocks, self.label = memo, system, blocks, label
        self.res = {}                 # (i, j), i < j -> [questions won by i, questions won by j, a reason]
        self.matches = self.clear = 0

    def play(self, i, j):
        i, j = min(i, j), max(i, j)
        if (i, j) not in self.res:
            wi = wj = 0
            why = ""
            for x, y in ((i, j), (j, i)):
                a = self.memo.ask(self.system, f"{self.label} A {self.blocks[x]}\n\n{self.label} B {self.blocks[y]}",
                                  PAIR_SCHEMA, 160)
                w = (a or {}).get("winner")
                if w in ("A", "B"):
                    if (x if w == "A" else y) == i:
                        wi += 1
                    else:
                        wj += 1
                    why = why or M.clean_text(a.get("reason"), 160)
            self.matches += 1
            self.clear += wi == 2 or wj == 2
            self.res[(i, j)] = [wi, wj, why]
        return self.res[(i, j)]

    def won(self, i, among):
        """Questions `i` won against the others in `among`."""
        return sum(r[0] if a == i else r[1] for (a, b), r in self.res.items()
                   if i in (a, b) and a in among and b in among)

    def agreed(self):
        return f"the judge gave the same winner in both orders in {self.clear} of {self.matches} matches"


def pairs(order, played):
    """Neighbours in `order` meet, never twice the same two."""
    left, out = list(order), []
    while len(left) > 1:
        i = left.pop(0)
        k = next((x for x in left if (min(i, x), max(i, x)) not in played), None)
        if k is not None:
            left.remove(k)
            out.append((i, k))
    return out


def _block(c, lines):
    return f"({_len(c):.0f} s):\n{_text(c, lines)}"


def _match_note(brain, pool, i, j, wi, wj):
    a, b = pool[i], pool[j]
    if wi == wj:
        res = "a tie (each won once)"
    else:
        w = a if wi > wj else b
        res = f"{T.mmss(w['s'])}" + (" (in both orders)" if max(wi, wj) == 2 else "")
    brain.note("match", f"{_label(a)} vs {_label(b)} → {res}")


def tournament(brain, memo, pool, lines, seed=0):
    """-> the finalists, best first. Each gets c["marathon"] = {"rank", "won", "of"} and a rank score."""
    n = len(pool)
    if n < 3:
        for r, c in enumerate(sorted(pool, key=lambda c: -c.get("rank", 0)), 1):
            c["marathon"] = {"rank": r, "won": 0, "of": 0}
        return sorted(pool, key=lambda c: -c.get("rank", 0))
    system = brain.system(_judge_system(brain))
    blocks = [_block(c, lines) for c in pool]
    qual = Arena(memo, system, blocks)
    pts = [0] * n
    k_fin = min(n, FINALISTS)
    rounds = min(ROUNDS, n - 1) if n > k_fin else 0
    # first rounds: RANDOM opponents every round (LoL: neighbours in the standings gave toss-ups)
    order = list(range(n))
    rng = random.Random(seed or (n * 7919 + len(lines)))      # the same video gets the same opponents
    for r in range(rounds):
        rng.shuffle(order)
        prs = pairs(order, qual.res)
        for q, (i, j) in enumerate(prs, 1):
            brain.step(f"tournament round {r + 1} of {rounds}, match {q} of {len(prs)}",
                       0.3 + 0.3 * (r + q / len(prs)) / rounds)
            wi, wj, _ = qual.play(i, j)
            pts[min(i, j)] += wi
            pts[max(i, j)] += wj
            _match_note(brain, pool, min(i, j), max(i, j), wi, wj)
        lead = sorted(range(n), key=lambda i: -pts[i])[:3]
        brain.note("rank", f"round {r + 1} of {rounds} done ({len(prs)} matches). In front: "
                           + ", ".join(f"{T.mmss(pool[i]['s'])} ({pts[i]} wins)" for i in lead))
    rank = sorted(range(n), key=lambda i: (-pts[i], i))
    fin = rank[:k_fin] + [i for i in rank[k_fin:] if pool[i].get("hinted")]     # the creator's hints always get in
    for i in rank:
        if i not in fin:
            brain.note("drop", f"out after the first rounds ({pts[i]} wins in {2 * rounds} questions)",
                       at=pool[i]["s"], title=_name(pool[i]))
    if rounds:
        brain.note("rank", f"first rounds done: {len(fin)} of {n} candidates reach the final - {qual.agreed()}")
    # the final: everybody meets everybody
    final = Arena(memo, system, blocks)
    games = [(a, b) for x, a in enumerate(fin) for b in fin[x + 1:]]
    for q, (a, b) in enumerate(games, 1):
        brain.step(f"final, match {q} of {len(games)}", 0.6 + 0.3 * q / len(games))
        wi, wj, _ = final.play(a, b)
        _match_note(brain, pool, min(a, b), max(a, b), wi, wj)
    group = set(fin)
    best = max(1, 2 * (len(fin) - 1))
    out = sorted(fin, key=lambda i: (-final.won(i, group), -pts[i], i))
    for r, i in enumerate(out, 1):
        w = final.won(i, group)
        pool[i]["marathon"] = {"rank": r, "won": w, "of": best}
        pool[i]["rank"] = round(0.5 + 0.5 * w / best + (0.3 if pool[i].get("hinted") else 0), 3)
    brain.note("rank", f"final standings ({final.agreed()}): "
                       + " · ".join(f"{r}. {T.mmss(pool[i]['s'])} ({pool[i]['marathon']['won']} of {best})"
                                    for r, i in enumerate(out, 1)))
    return [pool[i] for i in out]


# ---------- 4. winners ----------
def _join_neighbours(brain, keep, lines):
    """Winners that follow each other in the video within brain.merge_gap are one scene -> ONE Short."""
    out = []
    for c in sorted(keep, key=lambda c: c["s"]):
        p = out[-1] if out else None
        if p and M.gap(p, c) <= brain.merge_gap and max(p["e"], c["e"]) - p["s"] <= brain.max_s:
            best, other = (p, c) if p["marathon"]["rank"] <= c["marathon"]["rank"] else (c, p)
            brain.note("merge", f"joined with its neighbour {_label(other)} - both won and they are one scene, so "
                                "they become one Short", at=p["s"], title=_name(best))
            i0, i1, e = p["i0"], max(p["i1"], c["i1"]), max(p["e"], c["e"])
            out[-1] = dict(best, s=p["s"], e=e, i0=i0, i1=i1, start=lines[i0]["id"], end=lines[i1]["id"],
                           core=[(p.get("core") or [p["s"]])[0], max((p.get("core") or [0, p["e"]])[1],
                                                                     (c.get("core") or [0, c["e"]])[1])],
                           segments=[[p["s"], e, 1.0]], loud=bool(p.get("loud") or c.get("loud")),
                           hinted=bool(p.get("hinted") or c.get("hinted")),
                           why=" + ".join(w for w in (best.get("why"), other.get("why")) if w))
        else:
            out.append(c)
    return sorted(out, key=lambda c: c["marathon"]["rank"])


def _name_it(brain, memo, c, lines):
    """A listed winner has no title yet: one small question for its title, hook and why."""
    j = memo.ask(brain.system(_name_system(brain)), f"THE MOMENT ({_len(c):.0f} s):\n{_text(c, lines)}", NAME_SCHEMA, 300)
    if j:
        c["title"] = M.clean_text(j.get("title"), 90)
        c["hook"] = M.hook_text(j.get("hook") or c["title"])
        c["why"] = M.clean_text(j.get("why"), 300) or c.get("why", "")
        if j.get("type") in M.KINDS:
            c["type"] = j["type"]
    c["title"] = c.get("title") or c.get("best") or "…"
    c["hook"] = c.get("hook") or M.hook_text(c["title"])


def search(brain, lines, spikes, total_s, normal, cache_path=None):
    """`normal` = what the normal search found (it carries the creator's hints). -> the winners, best first."""
    memo = Memo(brain, cache_path)
    brain.note("step", "🏃 MARATHON MODE: many small rounds - list every idea, then compare them two at a time")
    if memo.data:
        brain.note("step", f"continuing a marathon that was stopped: {len(memo.data)} questions are already answered")
    try:
        found = find_wide(brain, memo, lines, spikes, total_s)
        pool = build_pool(brain, found, normal, lines, spikes, total_s)
        brain.note("step", f"{len(found)} listed ideas + {len(normal)} from the normal search -> {len(pool)} different "
                           "candidates for the tournament")
        if not pool:
            return []
        ranked = tournament(brain, memo, pool, lines)
        keep = [c for c in ranked if c.get("hinted") or 2 * c["marathon"]["won"] >= c["marathon"]["of"]]
        keep = keep if len(keep) >= MIN_WINNERS else ranked[:MIN_WINNERS]
        for c in ranked:
            if c not in keep:
                brain.note("drop", f"place {c['marathon']['rank']} in the final ({c['marathon']['won']} of "
                                   f"{c['marathon']['of']} questions won) - not kept", at=c["s"], title=_name(c))
        keep = _join_neighbours(brain, keep, lines)
        for q, c in enumerate(keep, 1):
            if not c.get("title"):
                brain.step(f"naming winner {q} of {len(keep)}", 0.9 + 0.1 * q / len(keep))
                _name_it(brain, memo, c, lines)
            brain.note("keep", f"winner {q}: place {c['marathon']['rank']} ({c['marathon']['won']} of "
                               f"{c['marathon']['of']} questions won)", at=c["s"], title=c.get("title"))
    finally:
        memo.save()                    # also when the job is stopped or paused: nothing is asked twice
    kept = []
    for c in keep:
        if any(M.overlap(c, k) > 0.5 for k in kept):
            brain.note("drop", "the same moment as a better winner", at=c["s"], title=_name(c))
        else:
            kept.append(c)
    brain.note("step", f"marathon done: {len(kept)} winners out of {len(pool)} candidates "
                       f"({brain.questions} questions asked in this run)")
    return kept

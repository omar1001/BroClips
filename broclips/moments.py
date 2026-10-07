"""Finding the moments that make good Shorts — the normal search (SPEC §6.3). Port of LoL Clips brain.find_moments
(windowed questions, line-id parsing, edges that never cut the payoff, one scene = one short, talk runs, hints),
made generic: every question carries the project's own description, video type, platforms and language.

The AI only PROPOSES moments by line id; plain code then fixes the edges, drops duplicates, joins picks of the same
scene and ranks. `Brain` holds what every question about one video needs (the Thinking AI, the project, the
creator's wishes) and writes the video's "What the AI is doing" log (<video>/ai_log.jsonl), which the video page
shows live. marathon.py and texts.py use the same Brain."""
import json
import re
import time
from pathlib import Path

from . import transcript as T
from .util import log

KINDS = ["funny", "story", "reaction", "tip", "explanation", "highlight", "opinion", "other"]
WINDOW_S, STRIDE_S = 300, 210          # the AI reads 5-minute parts, one every 3.5 minutes (overlap > a moment)
HINT_BEFORE_S, HINT_AFTER_S = 120, 90  # what it reads around a time the creator named
MERGE_GAP_S = 6                        # two picks this close together are one scene -> one Short (LoL 2026-09-29)
MIN_KEEP_S = 10                        # shorter than this is never a Short
GROW_GAP_S = 1.5                       # a short pick grows towards the minimum only over pauses shorter than this
MAX_SHORT_S = 180                      # platforms' limit for a Short (3 minutes)
CONTINUATION_LATIN = {"and", "so", "but", "because", "cause", "then", "or", "also", "plus"}
CONTINUATION_AR = ("و", "بس", "ف", "عشان", "ما هو", "اصل", "أصل", "وبعدين", "لكن")
TEACH_RE = re.compile(r"\b(lessons?|tutorials?|course|teach\w*|explain\w*|learn\w*|class|lectures?|how to)\b|درس|دروس|"
                      r"شرح|تعليم|محاضر|كورس", re.I)
TYPE_TEXT = {"gameplay": "gameplay recordings with the creator's own talking (reactions, jokes, stories)",
             "camera": "the creator talking to the camera", "screen": "screen recordings where the creator explains "
             "something (lessons / tutorials)", "podcast": "a podcast or a conversation",
             "other": "recordings where someone talks"}
GOOD = {
    "gameplay": "Good moments for this channel: funny reactions, jokes, banter, rants, little stories, and big "
                "plays the creator reacts to. Calm little stories count too, even without shouting.",
    "camera": "Good moments: a strong opinion, a little story, a funny line, a surprising fact, an emotional moment, "
              "a clear piece of advice.",
    "screen": "This is a lesson / tutorial. A good moment teaches ONE thing completely: a tip, a trick, a common "
              "mistake and its fix, one idea explained clearly (the problem AND the answer). It must make sense "
              "without the rest of the lesson.",
    "podcast": "Good moments: a strong opinion, a funny exchange, a surprising story or fact, a clear piece of "
               "advice. When two people talk, keep both sides of the exchange.",
    "other": "Good moments: something funny, surprising, useful or emotional, a little story, a strong opinion, a "
             "clear tip.",
}
TEACH_EXTRA = " If it teaches something, one moment = one complete idea (the question and the answer)."
PLATFORM_NAMES = {"youtube": "YouTube (long video)", "shorts": "YouTube Shorts", "tiktok": "TikTok",
                  "reels": "Instagram Reels"}

SCAN_SCHEMA = {"type": "object", "properties": {
    "notes": {"type": "string"},
    "moments": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
        "start": {"type": "string"}, "end": {"type": "string"}, "payoff": {"type": "string"},
        "type": {"type": "string", "enum": KINDS}, "why": {"type": "string"}, "title": {"type": "string"},
        "hook": {"type": "string"}, "score": {"type": "integer"}},
        "required": ["start", "end", "payoff", "type", "why", "title", "hook", "score"]}}},
    "required": ["notes", "moments"]}           # notes first: the model says what it sees before it picks

ALONE_SCHEMA = {"type": "object", "properties": {
    "works_alone": {"type": "boolean"}, "type": {"type": "string", "enum": KINDS}, "why": {"type": "string"},
    "title": {"type": "string"}, "hook": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["works_alone", "type", "why", "title", "hook", "score"]}


# ---------- the creator's wishes ----------
TIME_RE = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,3}):(\d{2})(?![\d:])")
MINUTE_RE = re.compile(r"(?:(?<!\w)min(?:ute)?s?|(?<!\w)(?:الدقيقة|دقيقة|د))\s*(\d{1,3})\b", re.I)


def _secs(m):
    h, mm, ss = m
    return (int(h) * 3600 if h else 0) + int(mm) * 60 + int(ss)


def parse_wishes(text):
    """What the creator typed in the wishes box, one wish per line (LoL split_request) -> (hints, instructions):
    hints        [(seconds, line)]  a line with a time ("5:30 the joke about the bug"): look there first
    instructions "line\\nline"      every line without a time: added to every question about this video"""
    hints, instr, rows = [], [], []
    for line in re.split(r"[\n;]+", text or ""):
        if len(TIME_RE.findall(line)) >= 2:         # "5:30 x, 6:10 y" = two hints
            rows += re.split(r"[,،]", line)
        else:
            rows.append(line)
    for line in rows:
        line = line.strip(" \t,")
        if not line:
            continue
        times = [_secs(m) for m in TIME_RE.findall(line)]
        if times:
            hints.append((times[0], line))
            continue
        m = MINUTE_RE.search(line)
        if m:
            hints.append((int(m.group(1)) * 60, line))
            continue
        instr.append(line)
    return hints, "\n".join(instr)


def teaching(project):
    p = project or {}
    return p.get("video_type") == "screen" or bool(TEACH_RE.search(str(p.get("prompt") or "")))


def limits(project):
    """(shortest, longest) good length of a Short in seconds: 15-90, lessons 30-120 (SPEC §6.3)."""
    return (30, 120) if teaching(project) else (15, 90)


def lang_rule(code):
    code = str(code or "").split("-")[0].lower()
    if code == "ar":
        return ("Write titles and hooks in Arabic, in the same dialect and style the speaker uses (for example "
                "Egyptian colloquial Arabic if they speak it), not formal Arabic.")
    if code in ("", "auto"):
        return "Write titles and hooks in the language the speaker uses."
    return f"Write titles and hooks in {T.language_name(code)}."


_CYR_IN_AR = re.compile(r"(?<=[؀-ۿ])[Ѐ-ӿ]+|[Ѐ-ӿ]+(?=[؀-ۿ])")


def fix_script(text):
    """Models sometimes put a Cyrillic letter inside an Arabic word (live test: "اللзقة" for "اللزقة"):
    the look-alike з becomes ز, other Cyrillic letters glued to Arabic ones are dropped."""
    return _CYR_IN_AR.sub(lambda m: "ز" if m.group(0) == "з" else "", str(text or ""))


def clean_text(text, n=120):
    """A model's free text for display: first line only, markers like {LOUD} and line ids removed."""
    t = re.sub(r"\{[^}]*\}", "", fix_script(text)).strip().split("\n")[0]
    t = re.sub(r"\bL\d{2,4}\b", "", t).strip(" -–:|\"'")
    return re.sub(r"\s+", " ", t)[:n].strip()


def hook_text(text, max_words=10):
    words = clean_text(text, 160).split()
    return " ".join(words[:max_words])


# ---------- the AI context + the "What the AI is doing" log ----------
class Brain:
    """Everything the questions about ONE video share. step(detail, frac) feeds the work line's progress (it is
    also where ⏹ Stop and pause-while-programs take effect)."""

    def __init__(self, llm, project=None, language="", wishes="", log_path=None, step=None, video_type=None):
        self.llm = llm
        self.project = dict(project or {})
        if video_type:
            self.project["video_type"] = video_type
        self.language = str(language or "").split("-")[0].lower()
        self.hints, self.instructions = parse_wishes(wishes)
        self.log_path = Path(log_path) if log_path else None
        self.step = step or (lambda detail="", frac=None: None)
        self.questions = 0
        self.min_s, self.max_s = limits(self.project)
        # picks this close are one scene -> one Short. A lesson changes topic at a pause, so there only picks that
        # practically touch are joined (live test 2026-10-07: two tips 2.6 s apart became one Short)
        self.merge_gap = 1.0 if teaching(self.project) else MERGE_GAP_S

    # --- the log the video page shows ---
    def note(self, kind, text, at=None, title=None):
        """One plain-English line for the video page. Never raises: a note must not stop the work. kinds: step, you,
        listen, clean, read, think, idea, drop, merge, keep, match, rank, render, text, done, warn."""
        title = clean_text(title, 80) if title else ""
        log.info("ai: %s%s%s", f"{T.mmss(at)} " if at is not None else "", f"[{title}] " if title else "", text)
        if not self.log_path:
            return
        rec = {"k": kind, "text": str(text)[:700], "clock": time.strftime("%H:%M:%S")}
        if at is not None:
            rec["at"] = T.mmss(at)
        if title:
            rec["title"] = title
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        for _ in range(8):          # a virus scanner can hold the file for a moment right after a write (LoL)
            try:
                with open(self.log_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(line)
                return
            except PermissionError:
                time.sleep(0.06)
            except Exception:
                return

    # --- what every question carries ---
    def short_platforms(self):
        names = [PLATFORM_NAMES[p] for p in self.project.get("platforms") or [] if p in ("shorts", "tiktok", "reels")]
        return ", ".join(names) or "YouTube Shorts, TikTok"

    def channel(self):
        p = self.project
        about = str(p.get("prompt") or "").strip() or "(the creator wrote nothing - judge from the transcript)"
        plats = ", ".join(PLATFORM_NAMES[x] for x in p.get("platforms") or [] if x in PLATFORM_NAMES) or "short videos"
        return (f"ABOUT THE CREATOR'S VIDEOS (their own words): {about}\n"
                f"VIDEO TYPE: {TYPE_TEXT.get(p.get('video_type'), TYPE_TEXT['other'])}\n"
                f"THEY POST ON: {plats}\n"
                f"LANGUAGE OF THE VIDEO: {T.language_name(self.language) if self.language else 'unknown'}")

    def system(self, base):
        s = base + "\n\n" + self.channel()
        if self.instructions:
            s += ("\n\nTHE CREATOR'S OWN WISHES FOR THIS VIDEO (they know their content best; follow them whenever "
                  "they apply to what you are looking at; they may be in any language):\n" + self.instructions)
        return s

    def ask(self, system, user, schema, max_tokens=1500, temperature=0.2):
        self.questions += 1
        return self.llm.chat_json(system, user, schema, temperature=temperature, max_tokens=max_tokens)

    def good(self):
        t = self.project.get("video_type") or "other"
        text = GOOD.get(t, GOOD["other"])
        if t != "screen" and TEACH_RE.search(str(self.project.get("prompt") or "")):
            text += TEACH_EXTRA
        return text


# ---------- transcript helpers ----------
def window_text(lines, spikes=()):
    spike_set = set(spikes or ())
    out = []
    for ln in lines:
        loud = any(int(t) in spike_set for t in range(int(ln["s"]), int(ln["e"]) + 1))
        m, s = divmod(int(ln["s"]), 60)
        out.append(f"{ln['id']} {m:02d}:{s:02d} {ln['text']}{' {LOUD}' if loud else ''}")
    return "\n".join(out)


def _nearest(lines, t, end=False):
    key = (lambda i: abs(lines[i]["e"] - t)) if end else (lambda i: abs(lines[i]["s"] - t))
    return min(range(len(lines)), key=key)


def line_index(lines, ref, by_id, end=False):
    """Line id -> index. Models sometimes answer 'mm:ss' instead of an id, or glue the time into an id ('L1026' /
    ',0644' = 10:26 / 06:44 — seen in LoL Clips); those are mapped to the nearest line. None if it cannot be read."""
    if not lines:
        return None
    if ref in by_id:
        return by_id[ref]
    ref = str(ref or "").strip()
    m = re.search(r"L?(\d{1,4})$", ref)                 # "L12", "L012", "12"
    if m and ":" not in ref:
        d = m.group(1)
        # ids never carry a leading zero past 3 digits, so "L0426" is a time; "L1026" is a time unless that id exists
        if len(d) == 4 and int(d[2:]) < 60 and (d[0] == "0" or f"L{d}" not in by_id):
            return _nearest(lines, int(d[:2]) * 60 + int(d[2:]), end)
        return by_id.get(f"L{int(d):03d}")
    m = re.search(r"(\d{1,2}):(\d{2})", ref)            # "07:06", ",07:06", "L 07:06"
    if not m:
        return None
    return _nearest(lines, int(m.group(1)) * 60 + int(m.group(2)), end)


def find_quote(lines, quote, near=None, lo=None, hi=None):
    """Index of the line holding the words the model quoted (at least half of them, >= 2), the closest to `near` if
    several do; only lines starting inside [lo, hi] seconds when given. Quoted words are more reliable than ids."""
    q = T.words_of(quote)
    if len(q) < 2:
        return None
    best = None
    for i, ln in enumerate(lines):
        if (lo is not None and ln["s"] < lo) or (hi is not None and ln["s"] > hi):
            continue
        common = len(q & T.words_of(ln["text"]))
        if common >= 2 and common >= 0.5 * len(q):
            key = (common, -abs(ln["s"] - near) if near is not None else 0)
            if best is None or key > best[0]:
                best = (key, i)
    return best[1] if best else None


def _continues(text):
    """Does this line continue the one before ("and so…", "و…")? Then a moment must not start on it."""
    t = str(text or "").strip()
    if not t:
        return False
    if T.is_arabic(t.split()[0]):
        return t.startswith(CONTINUATION_AR)
    return t.split()[0].lower().strip(".,!?\"'") in CONTINUATION_LATIN


def edges(lines, i0, i1, spikes, min_s, max_s, grow_min=True, total_s=None):
    """A moment's edges by plain rules -> (s, e, i0, i1): never open on a line that continues the one before, never
    cut continuous speech, grow short ones to the minimum, keep the laugh / shout right after the payoff."""
    spike_set = set(spikes or ())
    while i0 > 0 and _continues(lines[i0]["text"]) and lines[i0]["s"] - lines[i0 - 1]["s"] < 10:
        i0 -= 1
    while (i1 + 1 < len(lines) and lines[i1 + 1]["s"] - lines[i1]["e"] < 0.35
           and lines[i1 + 1]["e"] - lines[i0]["s"] <= max_s):
        i1 += 1
    while i0 > 0 and lines[i0]["s"] - lines[i0 - 1]["e"] < 0.35 and lines[i1]["e"] - lines[i0 - 1]["s"] <= max_s:
        i0 -= 1
    # grow a short pick towards the minimum length, but only over continuous talk: a pause is often where the
    # next topic starts (live test 2026-10-07: two separate tips grown to 30 s ran into each other and became one)
    while grow_min and lines[i1]["e"] - lines[i0]["s"] < min_s:
        if (i1 + 1 < len(lines) and lines[i1 + 1]["s"] - lines[i1]["e"] < GROW_GAP_S
                and lines[i1 + 1]["e"] - lines[i0]["s"] <= max_s):
            i1 += 1
        elif i0 > 0 and lines[i0]["s"] - lines[i0 - 1]["e"] < GROW_GAP_S and lines[i1]["e"] - lines[i0 - 1]["s"] <= max_s:
            i0 -= 1
        else:
            break
    s = max(0.0, lines[i0]["s"] - 0.3)
    e = lines[i1]["e"] + 0.6
    after = [k for k in spike_set if e <= k <= e + 6]
    if after:
        e = max(after) + 1.4
    if total_s:
        e = min(e, float(total_s))
    return round(float(s), 2), round(float(e), 2), int(i0), int(i1)


def pieces(m):
    return [(a, b) for a, b, *_ in (m.get("segments") or [[m["s"], m["e"], 1.0]])]


def overlap(c, k):
    """Shared time / length of the shorter one (0-1)."""
    pc, pk = pieces(c), pieces(k)
    ov = sum(max(0.0, min(b, d) - max(a, x)) for a, b in pc for x, d in pk)
    return ov / max(0.01, min(sum(b - a for a, b in pc), sum(b - a for a, b in pk)))


def length(m):
    return sum(b - a for a, b in pieces(m))


def _rank(score, loud, hinted):
    return round(min(10, max(1, int(score or 1))) / 10 + (0.1 if loud else 0) + (0.5 if hinted else 0), 3)


def _talk_runs(lines, min_s, max_s, gap=2.5, min_wps=1.5):
    """Stretches where the creator talks almost without stopping — often a little story. A cheap signal the model
    tends to miss (LoL). -> [(i0, i1, words)] most words first"""
    if not lines:
        return []
    runs, cur = [], [0]
    for i in range(1, len(lines)):
        if lines[i]["s"] - lines[i - 1]["e"] < gap:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    out = []
    for r in runs:
        dur = lines[r[-1]]["e"] - lines[r[0]]["s"]
        nw = sum(len(lines[i]["words"]) for i in r)
        if min_s <= dur <= max_s and nw / max(dur, 0.1) >= min_wps:
            out.append((r[0], r[-1], nw))
    return sorted(out, key=lambda x: -x[2])


def _scan_system(brain):
    return f"""You help a creator turn a long recording into short vertical videos ({brain.short_platforms()}).
The transcript is automatic speech-to-text: spelling mistakes and wrong words are normal - judge the meaning.
Each line starts with its id (like L045) and its time (mm:ss). {{LOUD}} marks where the voice got suddenly loud.

Find 0 to 4 moments in this part that would work as a short ON ITS OWN, for a viewer who never saw the rest.
{brain.good()}
A good moment has a setup AND a payoff (a punchline, a twist, an answer, a key point, a reaction with a clear cause),
makes sense alone, and its first line grabs attention. It is {brain.min_s}-{brain.max_s} seconds long.
It starts at the beginning of a sentence and ends after the payoff and the reaction to it - never in the middle of a
sentence. If someone else asks or answers, keep both sides.
Reject: greetings, filler, "like and subscribe" talk, anything that needs earlier context to be understood.
Use ONLY line ids that appear in the transcript (they look like L045). Never write a time as an id.
payoff = the id of the line with the punchline / answer / key point.
title = a catchy title, at most 8 words. hook = the text shown on top of the short, at most 8 words.
{lang_rule(brain.language)} Never put markers like {{LOUD}} or line ids in them.
why = one short, simple English sentence: why this works as a short.
score = 1-10, how good it is as a short (be strict: 8 or more only for a moment people would share).
notes: FIRST write 1-2 short, simple English sentences: what happens in this part and why you pick these moments (or
why nothing here is good enough). Returning an empty list is fine."""


def _alone_system(brain):
    return f"""You judge ONE stretch of a creator's talk (automatic speech-to-text: spelling errors are normal - judge the
meaning). {brain.good()}
Would this stretch work ALONE as a {brain.min_s}-{brain.max_s} second short ({brain.short_platforms()}) for a viewer
who never saw the rest of the video? Be honest: small talk, filler and things that need earlier context do not.
why = one short, simple English sentence. score = 1-10 (8 or more only for a moment people would share).
title and hook: at most 8 words each. {lang_rule(brain.language)}"""


def _idea_note(brain, lines, by_id, c, prefix=""):
    a, b = line_index(lines, c.get("start"), by_id), line_index(lines, c.get("end"), by_id, end=True)
    where = f"{T.mmss(lines[a]['s'])}–{T.mmss(lines[b]['e'])}" if a is not None and b is not None else "?"
    brain.note("idea", f"{prefix}{c.get('type', '')} {where}: {clean_text(c.get('why'), 200)}",
               at=lines[a]["s"] if a is not None else None, title=c.get("title"))


def gap(p, c):
    """Seconds between two picks as the AI chose them (their "core", before the edges grew them) — two separate
    tips grown towards the minimum length can touch without being one scene (live test 2026-10-07)."""
    pc, cc = p.get("core") or [p["s"], p["e"]], c.get("core") or [c["s"], c["e"]]
    return cc[0] - pc[1]


def _merge_close(brain, clips, max_s):
    """One scene = one Short: picks that follow each other within brain.merge_gap become ONE moment (if <= max)."""
    out = []
    for c in sorted(clips, key=lambda x: x["s"]):
        p = out[-1] if out else None
        if p and gap(p, c) <= brain.merge_gap and max(p["e"], c["e"]) - p["s"] <= max_s:
            best, other = (p, c) if p["rank"] >= c["rank"] else (c, p)
            brain.note("merge", f"joined with the idea at {T.mmss(other['s'])} («{clean_text(other.get('title'), 50)}») "
                                "- it is the same scene, so it becomes one longer Short", at=p["s"], title=best.get("title"))
            e = max(p["e"], c["e"])
            out[-1] = dict(best, s=p["s"], e=e, i0=p["i0"], i1=max(p["i1"], c["i1"]), segments=[[p["s"], e, 1.0]],
                           core=[(p.get("core") or [p["s"]])[0], max((p.get("core") or [0, p["e"]])[1],
                                                                     (c.get("core") or [0, c["e"]])[1])],
                           loud=bool(p["loud"] or c["loud"]), hinted=bool(p.get("hinted") or c.get("hinted")),
                           why=" + ".join(x for x in (best.get("why"), other.get("why")) if x))
        else:
            out.append(c)
    return sorted(out, key=lambda x: -x["rank"])


def make_moment(lines, i0, i1, c, spikes, brain, total_s=None, src="search"):
    """An AI pick (or a rule pick) -> a complete moment dict with edges by the plain rules."""
    core = [lines[i0]["s"], lines[i1]["e"]]
    s, e, i0, i1 = edges(lines, i0, i1, spikes, brain.min_s, brain.max_s, total_s=total_s)
    loud = any(s <= k <= e for k in spikes or ())
    score = min(10, max(1, int(c.get("score") or 5)))
    title = clean_text(c.get("title"), 90)
    return {"s": s, "e": e, "i0": i0, "i1": i1, "start": lines[i0]["id"], "end": lines[i1]["id"],
            "payoff": c.get("payoff") or lines[i1]["id"], "type": c.get("type") if c.get("type") in KINDS else "other",
            "title": title, "hook": hook_text(c.get("hook") or title), "why": clean_text(c.get("why"), 300),
            "score": score, "loud": bool(loud), "hinted": bool(c.get("hinted")),
            "rank": _rank(score, loud, c.get("hinted")), "src": src, "segments": [[s, e, 1.0]], "core": core}


# ---------- the search ----------
def find(brain, lines, spikes, total_s):
    """The normal search. -> candidate moments, best first (not yet cut to the project's maximum)."""
    if not lines:
        brain.note("drop", "nothing was said in this video, so there are no moments to find")
        return []
    by_id = {ln["id"]: i for i, ln in enumerate(lines)}
    starts = [t for t in range(0, int(total_s) + 1, STRIDE_S) if t == 0 or t + 60 < total_s] or [0]
    n_steps = len(brain.hints) + len(starts) + 3
    raw, k = [], 0
    scan_sys = brain.system(_scan_system(brain))
    # 1) the creator's own hints first ("5:30 the funny story"): look closely around each one
    for ht, hint in brain.hints:
        k += 1
        brain.step(f"looking where you pointed ({T.mmss(ht)})", k / n_steps)
        win = [ln for ln in lines if ht - HINT_BEFORE_S <= ln["s"] < ht + HINT_AFTER_S]
        if len(win) < 2:
            brain.note("drop", f"you pointed here (\"{hint}\"), but almost nothing is said there", at=ht)
            continue
        brain.note("read", f"looking closely around the moment you remember: \"{hint}\"", at=ht)
        res = brain.ask(scan_sys, f"The creator remembers a good moment around {T.mmss(ht)} and wrote: \"{hint}\".\n"
                                  "Find that COMPLETE moment in this part (1-2 moments, near that time).\n\n"
                                  "TRANSCRIPT PART:\n" + window_text(win, spikes), SCAN_SCHEMA)
        if (res or {}).get("notes"):
            brain.note("think", clean_text(res["notes"], 400), at=ht)
        for c in (res or {}).get("moments") or []:
            c["hinted"] = True
            _idea_note(brain, lines, by_id, c, "(your hint) ")
            raw.append(c)
    # 2) the whole video in overlapping parts
    for t in starts:
        k += 1
        brain.step(f"reading part {starts.index(t) + 1} of {len(starts)}", k / n_steps)
        win = [ln for ln in lines if t <= ln["s"] < t + WINDOW_S]
        if len(win) < 3:
            continue
        brain.note("read", f"reading {T.mmss(t)}–{T.mmss(min(total_s, t + WINDOW_S))} ({len(win)} lines of talk)", at=t)
        res = brain.ask(scan_sys, "TRANSCRIPT PART:\n" + window_text(win, spikes), SCAN_SCHEMA)
        if res is None:
            brain.note("drop", "no usable answer for this part", at=t)
            continue
        if res.get("notes"):
            brain.note("think", clean_text(res["notes"], 400), at=t)
        for c in res.get("moments") or []:
            _idea_note(brain, lines, by_id, c)
            raw.append(c)
    # 3) long stretches of non-stop talk the scan did not cover: asked one by one (stories hide there)
    covered = []
    for c in raw:
        a, b = line_index(lines, c.get("start"), by_id), line_index(lines, c.get("end"), by_id, end=True)
        if a is not None and b is not None:
            covered.append((lines[min(a, b)]["s"], lines[max(a, b)]["e"]))
    asked = 0
    for i0, i1, _nw in _talk_runs(lines, max(20, brain.min_s), brain.max_s):
        if asked >= 3:
            break
        s, e = lines[i0]["s"], lines[i1]["e"]
        if any(min(e, b) - max(s, a) > 0.5 * (e - s) for a, b in covered):
            continue
        asked += 1
        k += 1
        brain.step("checking a long stretch of talk", min(0.99, k / n_steps))
        seg = "\n".join(f"{ln['id']} {ln['text']}" for ln in lines[i0:i1 + 1])
        j = brain.ask(brain.system(_alone_system(brain)), "STRETCH:\n" + seg, ALONE_SCHEMA, max_tokens=500)
        if j and j.get("works_alone"):
            raw.append(dict(j, start=lines[i0]["id"], end=lines[i1]["id"], payoff=lines[i1]["id"]))
            brain.note("idea", f"you talked for {e - s:.0f} s without stopping - it works on its own: "
                               f"{clean_text(j.get('why'), 200)}", at=s, title=j.get("title"))
        elif j:
            brain.note("drop", f"you talked for {e - s:.0f} s without stopping, but it does not work on its own: "
                               f"{clean_text(j.get('why'), 200)}", at=s)
    # 4) ids -> times, edges by plain rules, checks
    clips = []
    for c in raw:
        i0 = line_index(lines, c.get("start"), by_id)
        i1 = line_index(lines, c.get("end"), by_id, end=True)
        title = c.get("title")
        if i0 is None or i1 is None:
            brain.note("drop", f"the AI named lines that do not exist ({c.get('start')}–{c.get('end')})", title=title)
            continue
        i0, i1 = min(i0, i1), max(i0, i1)
        ip = line_index(lines, c.get("payoff"), by_id, end=True)
        if ip is not None and i0 <= ip <= i1 + 3:          # the payoff is always inside
            i1 = max(i1, ip)
        m = make_moment(lines, i0, i1, c, spikes, brain, total_s)
        if m["e"] - m["s"] < MIN_KEEP_S:
            brain.note("drop", f"too short ({m['e'] - m['s']:.0f} s)", at=m["s"], title=title)
            continue
        if m["e"] - m["s"] > min(MAX_SHORT_S, brain.max_s + 20):
            brain.note("drop", f"too long ({m['e'] - m['s']:.0f} s)", at=m["s"], title=title)
            continue
        clips.append(m)
    # 5) the same moment twice -> the better one; picks of one scene -> one Short
    clips.sort(key=lambda c: -c["rank"])
    kept, repeats = [], []
    for c in clips:
        if any(overlap(c, k2) > 0.5 for k2 in kept):
            repeats.append(T.mmss(c["s"]))
        else:
            kept.append(c)
    if repeats:
        brain.note("drop", f"{len(repeats)} idea(s) were the same moment as a better idea "
                           f"(at {', '.join(sorted(set(repeats)))})")
    kept = _merge_close(brain, kept, brain.max_s)
    brain.note("step", f"{len(raw)} ideas -> {len(kept)} different moments after the checks")
    return kept


def _rule_pieces(lines, min_s, max_s, gap=1.5):
    """Stretches of talk (split at pauses >= gap) cut into pieces of about the middle length -> [(i0, i1, words)]."""
    target = (min_s + max_s) / 2
    runs, cur = [], [0]
    for i in range(1, len(lines)):
        if lines[i]["s"] - lines[i - 1]["e"] < gap:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    out = []
    for r in runs:
        a = r[0]
        for i in r:
            if lines[i]["e"] - lines[a]["s"] >= target or i == r[-1]:
                if lines[i]["e"] - lines[a]["s"] >= MIN_KEEP_S:
                    out.append((a, i, sum(len(lines[k]["words"]) for k in range(a, i + 1))))
                a = i + 1
                if a > r[-1]:
                    break
    return sorted(out, key=lambda x: -x[2])


def fallback(brain, lines, spikes, total_s, n):
    """No usable answer from the AI at all: pick the stretches with the most talk by plain rules, so the user still
    gets Shorts (and the demo AI works). Titles = the first words said."""
    out = []
    for i0, i1, _nw in _rule_pieces(lines, brain.min_s, brain.max_s):
        first = " ".join(lines[i0]["text"].split()[:7])
        c = {"title": first, "hook": first, "type": "other", "score": 5,
             "why": "Picked by simple rules: a long stretch of non-stop talk (the AI gave no usable answer)."}
        m = make_moment(lines, i0, i1, c, spikes, brain, total_s, src="rules")
        if m["e"] - m["s"] >= MIN_KEEP_S and not any(overlap(m, o) > 0.5 for o in out):
            out.append(m)
        if len(out) >= n:
            break
    if out:
        brain.note("warn", f"the AI gave no usable moments, so {len(out)} were picked by simple rules "
                           "(the longest stretches of talk)")
    return out

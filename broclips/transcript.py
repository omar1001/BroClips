"""Words (from the Listening AI) -> numbered LINES (what the Thinking AI reads) and caption PHRASES (what the Shorts
show), plus a filter for the sentences Whisper invents in silence. Any language; Arabic is handled (letter variants,
right-to-left is libass' job). Ported from LoL Clips asr.make_lines + render._caption_chunks and generalised.

A word is {"w": "hello", "s": 1.23, "e": 1.56, "p": 0.98} (times in seconds, p = how sure the listener was)."""
import math
import re

END_PUNCT = re.compile(r"[.!?؟…]+[\"'”’»)\]]*$")         # a sentence ends after this word
SOFT_PUNCT = re.compile(r"[,،;؛:]+[\"'”’»)\]]*$")        # a good place for a caption break
TRAIL = re.compile(r"[.,،;؛:…]+$")                       # never shown at the end of a caption word (? and ! stay)
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
BRACKETED = re.compile(r"^\s*([\[(（【♪*].*[\])）】♪*]|♪+)\s*$")   # "[Music]", "(applause)", "♪" — not speech
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_AR = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})

MIN_WORD_S = 0.12           # Whisper can return zero-length words: every caption word is shown at least this long
LANG_NAMES = {"en": "English", "ar": "Arabic", "fr": "French", "es": "Spanish", "de": "German", "tr": "Turkish",
              "pt": "Portuguese", "it": "Italian", "ru": "Russian", "hi": "Hindi", "ur": "Urdu", "id": "Indonesian",
              "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "nl": "Dutch", "pl": "Polish", "fa": "Persian"}


# ---------- text helpers ----------
def is_arabic(text):
    """True if the text contains Arabic letters (captions then use the Arabic font)."""
    return bool(ARABIC.search(str(text or "")))


def norm(text):
    """Comparable text: lower case, Arabic letter variants unified, diacritics and punctuation dropped."""
    t = _DIACRITICS.sub("", str(text or "")).translate(_AR).lower().replace("'", "").replace("’", "")
    return " ".join(re.findall(r"\w+", t))


def words_of(text):
    return set(norm(text).split())


def language_name(code):
    code = str(code or "").split("-")[0].lower()
    return LANG_NAMES.get(code, code or "the language of the video")


def mmss(sec):
    sec = max(0.0, float(sec or 0))
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def display(word):
    """A word as a caption shows it: no full stop / comma at its end (they look like noise on a Short)."""
    w = str(word or "").strip()
    return w if re.fullmatch(r"\d+([.,]\d+)*", w) else TRAIL.sub("", w)        # "3.5" and "1,000" stay


def clean_words(words):
    """Plain, sorted words with sane times (floats, end >= start, empty words dropped)."""
    out = []
    for w in words or []:
        t = str(w.get("w") or "").strip()
        if not t:
            continue
        s = float(w.get("s") or 0.0)
        e = max(s, float(w.get("e") if w.get("e") is not None else s))
        out.append({"w": t, "s": round(s, 3), "e": round(e, 3), "p": round(float(w.get("p", 1.0)), 3)})
    out.sort(key=lambda x: x["s"])
    return out


# ---------- lines for the AI ----------
def make_lines(words, pause=0.7, max_len=12.0):
    """Group words into short lines: break on a pause, at a sentence end, or when a line gets too long (LoL)."""
    lines, cur = [], []
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] >= pause or w["e"] - cur[0]["s"] > max_len):
            lines.append(cur)
            cur = []
        cur.append(w)
        if END_PUNCT.search(w["w"]):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return [{"id": f"L{i + 1:03d}", "s": ln[0]["s"], "e": ln[-1]["e"],
             "text": " ".join(w["w"] for w in ln), "words": ln} for i, ln in enumerate(lines)]


# ---------- what Whisper invents in silence ----------
# Stock sentences the speech model was trained on (YouTube subtitles) and "hears" in quiet stretches. kind:
# "always" = never real speech in practice; "tail" = drop the rest of that line too ("Subtitles by <name>");
# "maybe" = a creator may really say it, so it goes only when it is quiet, unsure, alone or repeated.
STOCK = [
    ("amara org", "always"), ("نانسي قنقر", "always"), ("subtitles by", "tail"), ("subtitles made by", "tail"),
    ("captions by", "tail"), ("transcribed by", "tail"), ("transcription by", "tail"), ("translated by", "tail"),
    ("sous titres réalisés par", "tail"), ("subtítulos realizados por", "tail"),
    ("thanks for watching", "maybe"), ("thank you for watching", "maybe"), ("thank you so much for watching", "maybe"),
    ("please subscribe", "maybe"), ("like and subscribe", "maybe"), ("subscribe to my channel", "maybe"),
    ("subscribe to the channel", "maybe"), ("dont forget to subscribe", "maybe"), ("see you in the next video", "maybe"),
    ("merci davoir regardé", "maybe"), ("gracias por ver", "maybe"),
    ("اشتركوا في القناة", "maybe"), ("اشترك في القناة", "maybe"), ("اشتركو في القناة", "maybe"),
    ("لا تنسوا الاشتراك في القناة", "maybe"), ("لا تنسى الاشتراك في القناة", "maybe"),
    ("شكرا للمشاهدة", "maybe"), ("شكرا على المشاهدة", "maybe"), ("شكرا لكم على المشاهدة", "maybe"),
    ("شكرا لمشاهدتكم", "maybe"),
]
_STOCK = [(norm(p).split(), kind, p) for p, kind in STOCK]


def _tok_eq(tok, want):
    return tok == want or (len(tok) > 2 and tok[0] in "وف" and tok[1:] == want and is_arabic(tok))


def _find_stock(words):
    """[(first word index, last word index, phrase, kind)] of stock sentences in the word list."""
    flat = []                                       # (token, word index)
    for i, w in enumerate(words):
        flat += [(t, i) for t in norm(w["w"]).split()]
    hits, k = [], 0
    while k < len(flat):
        hit = None
        for toks, kind, phrase in _STOCK:
            if k + len(toks) <= len(flat) and all(_tok_eq(flat[k + j][0], t) for j, t in enumerate(toks)):
                if hit is None or len(toks) > len(hit[0]):
                    hit = (toks, kind, phrase)
        if hit is None:
            k += 1
            continue
        a, b = flat[k][1], flat[k + len(hit[0]) - 1][1]
        if hit[1] == "tail":                        # "Subtitles by the Amara.org community": the rest of the line
            while b + 1 < len(words) and words[b + 1]["s"] - words[b]["e"] < 0.7 and b - a < 12:
                b += 1
        hits.append((a, b, hit[2], hit[1]))
        k += len(hit[0])
        while k < len(flat) and flat[k][1] <= b:
            k += 1
    return hits


def drop_hallucinations(words, per_s=None, quiet_db=10.0):
    """Remove the sentences Whisper typically invents in silence ("Thanks for watching", "اشتركوا في القناة",
    "Subtitles by…") and non-speech marks ("[Music]"). A stock sentence that a creator may really say is dropped only
    when it is QUIET (per_s = loudness per second: >= quiet_db under the usual speech level), UNSURE (mean word
    confidence < 0.5), ALONE (>= 1 s of silence on both sides — only used when the loudness is unknown) or REPEATED
    (3+ times, or twice in a row). -> (kept words, dropped [{"text", "s", "e", "why"}])"""
    words = list(words or [])
    drop, dropped = set(), []
    for i, w in enumerate(words):
        if BRACKETED.match(w["w"]):
            drop.add(i)
            dropped.append({"text": w["w"], "s": w["s"], "e": w["e"], "why": "a sound mark, not speech"})
    hits = [h for h in _find_stock(words) if not any(i in drop for i in range(h[0], h[1] + 1))]
    level = None
    if per_s:
        secs = sorted({int(w["s"]) for w in words if 0 <= int(w["s"]) < len(per_s)})
        vals = sorted(per_s[s] for s in secs)
        level = vals[len(vals) // 2] if vals else None
    count = {}
    for a, b, phrase, kind in hits:
        count[phrase] = count.get(phrase, 0) + 1
    for n, (a, b, phrase, kind) in enumerate(hits):
        seg = words[a:b + 1]
        why = ""
        if kind in ("always", "tail"):
            why = "a sentence speech-to-text invents (it is never really said)"
        else:
            conf = sum(w.get("p", 1.0) for w in seg) / len(seg)
            gap_before = seg[0]["s"] - words[a - 1]["e"] if a > 0 else 99.0
            gap_after = words[b + 1]["s"] - seg[-1]["e"] if b + 1 < len(words) else 99.0
            prev_hit = hits[n - 1] if n else None
            next_hit = hits[n + 1] if n + 1 < len(hits) else None
            twice = ((prev_hit and prev_hit[2] == phrase and seg[0]["s"] - words[prev_hit[1]]["e"] < 2.0)
                     or (next_hit and next_hit[2] == phrase and words[next_hit[0]]["s"] - seg[-1]["e"] < 2.0))
            loud = None
            if level is not None:          # past the end of the measured loudness there was no sound (-70)
                secs = [per_s[s] if s < len(per_s) else -70.0 for s in range(int(seg[0]["s"]), int(seg[-1]["e"]) + 1)
                        if s >= 0]
                loud = sum(secs) / len(secs) if secs else None
            if count[phrase] >= 3 or twice:
                why = "repeated — speech-to-text loops on this sentence"
            elif conf < 0.5:
                why = "the listener was not sure about it"
            elif loud is not None and loud < level - quiet_db:
                why = "it was said into silence (the sound there is very quiet)"
            elif loud is None and gap_before >= 1.0 and gap_after >= 1.0:
                why = "it stands alone between silences"
        if why:
            drop.update(range(a, b + 1))
            dropped.append({"text": " ".join(w["w"] for w in seg), "s": seg[0]["s"], "e": seg[-1]["e"], "why": why})
    return [w for i, w in enumerate(words) if i not in drop], sorted(dropped, key=lambda d: d["s"])


# ---------- caption phrases for the Shorts ----------
def fix_times(words, min_dur=MIN_WORD_S):
    """Copies of the words with usable times: words that start at (almost) the same moment are spread over the time
    they share, and every word lasts at least `min_dur` when the next word leaves room (else up to the next word)."""
    ws = [dict(w) for w in words]
    i = 0
    while i < len(ws):                      # groups of words starting within 50 ms of each other
        j = i + 1
        while j < len(ws) and ws[j]["s"] - ws[j - 1]["s"] < 0.05:
            j += 1
        if j - i > 1:
            end = max(w["e"] for w in ws[i:j])
            if j < len(ws):
                end = min(max(end, ws[i]["s"] + min_dur * (j - i)), ws[j]["s"])
            else:
                end = max(end, ws[i]["s"] + min_dur * (j - i))
            step = max(0.01, (end - ws[i]["s"]) / (j - i))
            for k in range(i, j):
                ws[k]["s"] = round(ws[i]["s"] + (k - i) * step, 3)
                ws[k]["e"] = round(ws[k]["s"] + step, 3)
        i = j
    for k, w in enumerate(ws):
        nxt = ws[k + 1]["s"] if k + 1 < len(ws) else None
        if w["e"] - w["s"] < min_dur:
            want = w["s"] + min_dur
            w["e"] = round(want if nxt is None else max(w["e"], min(want, nxt)), 3)
    return ws


def _soft_break(run, i):
    return bool(SOFT_PUNCT.search(run[i]["raw"])) or (i + 1 < len(run) and run[i + 1]["s"] - run[i]["e"] >= 0.25)


def _split(run, max_words, max_s):
    """One stretch of speech without a hard break -> balanced phrases (7 words -> 4 + 3, not 6 + 1), preferring to
    break after a comma or a small pause, keeping each phrase short enough to read."""
    n = len(run)
    if n <= max_words and run[-1]["e"] - run[0]["s"] <= max_s:
        return [run]
    target = n / math.ceil(n / max_words)
    inf = float("inf")
    best, back = [0.0] + [inf] * n, [0] * (n + 1)
    for i in range(1, n + 1):
        for size in range(1, min(max_words, i) + 1):
            j = i - size
            if best[j] == inf:
                continue
            dur = run[i - 1]["e"] - run[j]["s"]
            c = 1.0 + 0.25 * (size - target) ** 2
            if size == 1:
                c += 3.0
            if dur > max_s:
                c += 4.0 * (dur - max_s)
            if i < n and not _soft_break(run, i - 1):
                c += 0.8
            if best[j] + c < best[i]:
                best[i], back[i] = best[j] + c, j
    cuts, i = [], n
    while i > 0:
        cuts.append((back[i], i))
        i = back[i]
    return [run[a:b] for a, b in reversed(cuts)]


def phrases(words, max_words=6, max_s=3.0, pause=0.6):
    """Caption phrases: 2-6 words, broken at sentence ends, pauses and cuts, never across a cut (`part`), balanced,
    no trailing full stops, every word visible long enough.
    words: [{"w", "s", "e", "part"?}] in the Short's own time. -> [{"s", "e", "part", "words": [{"w", "s", "e"}]}]"""
    src = [w for w in sorted(words, key=lambda x: x["s"])
           if display(w.get("w")) and not BRACKETED.match(str(w.get("w")))]
    ws = fix_times([dict(w, raw=str(w["w"]).strip(), w=display(w["w"])) for w in src])
    runs, cur = [], []
    for w in ws:                            # hard breaks: a cut, a pause, the end of a sentence
        if cur and (w.get("part", 0) != cur[-1].get("part", 0) or w["s"] - cur[-1]["e"] >= pause
                    or END_PUNCT.search(cur[-1]["raw"])):
            runs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        for chunk in _split(run, max_words, max_s):
            out.append({"s": chunk[0]["s"], "e": chunk[-1]["e"], "part": chunk[0].get("part", 0),
                        "words": [{k: v for k, v in w.items() if k != "raw"} for w in chunk]})
    return out

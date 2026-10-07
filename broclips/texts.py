"""Titles, descriptions and hashtags per platform, and YouTube chapters for the long video (SPEC §6.6). Port of LoL
Clips brain.packaging, made generic (project prompt, platforms, language).

One question writes the packaging (3 titles + a description + hashtags for the long video; a title, a caption and
hashtags per Short); a second one splits the long video into chapters by topic (line ids -> times in the long video
through long.json). Code then builds the ready-to-paste text for every platform, which the video page shows with
📋 copy buttons. No usable answer -> the texts are built from what is known (the Shorts' own titles)."""
import re
from datetime import datetime

from . import longvideo
from . import moments as M
from . import transcript as T

PACK_SCHEMA = {"type": "object", "properties": {
    "long_titles": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    "long_description": {"type": "string"},
    "long_hashtags": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
    "shorts": {"type": "array", "maxItems": 30, "items": {"type": "object", "properties": {
        "n": {"type": "integer"}, "title": {"type": "string"}, "caption": {"type": "string"},
        "hashtags": {"type": "array", "maxItems": 6, "items": {"type": "string"}}},
        "required": ["n", "title", "caption", "hashtags"]}}},
    "required": ["long_titles", "long_description", "long_hashtags", "shorts"]}
CHAPTER_SCHEMA = {"type": "object", "properties": {"chapters": {"type": "array", "maxItems": 10, "items": {
    "type": "object", "properties": {"start": {"type": "string"}, "name": {"type": "string"}},
    "required": ["start", "name"]}}}, "required": ["chapters"]}
CHAPTER_MIN_S = 10          # YouTube: at least 3 chapters, each at least 10 seconds, the first at 0:00
MAX_PROMPT_CHARS = 24000    # transcript text per question (keeps a 16k-token local model comfortable)


def _pack_system(brain):
    return f"""You write the texts a creator needs to post their videos. {M.lang_rule(brain.language)}
Use the creator's own tone. Be honest: never promise or invent anything that is not in the video.
Titles: catchy, at most 70 characters, no quotation marks. Hashtags: words without spaces (join words), with or
without #. Never put line ids or markers like {{LOUD}} in a text."""


def _chapter_system(brain):
    return f"""You split a video into chapters for YouTube. The transcript is automatic speech-to-text (spelling mistakes
are normal - judge the meaning). Each line starts with its id (like L045) and its time (mm:ss).
Make 3 to 8 chapters that follow the topics of the video, in time order. start = the id of the line where a chapter
starts (the first chapter starts at the first line). name = 2-5 words. {M.lang_rule(brain.language)}"""


def hhmmss(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def tags(items, n=10):
    """['python tips', '#coding', 'برمجة'] -> ['#pythontips', '#coding', '#برمجة'] (unique, at most n)."""
    out = []
    for t in items or []:
        t = re.sub(r"[^\w]+", "", str(t or "").strip().lstrip("#").replace(" ", ""), flags=re.UNICODE)
        if t and t.lower() not in {x[1:].lower() for x in out}:
            out.append("#" + t)
    return out[:n]


def _multi(text, n=2000):
    """A model's longer text (paragraphs kept), without markers like {LOUD} or line ids."""
    t = re.sub(r"\{[^}]*\}", "", M.fix_script(text))
    t = re.sub(r"\bL\d{2,4}\b", "", t)
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.strip().splitlines()).strip()[:n]


def _sample(lines, budget=MAX_PROMPT_CHARS, with_ids=False):
    rows = [(f"{ln['id']} {T.mmss(ln['s'])} " if with_ids else "") + ln["text"] for ln in lines]
    total = sum(len(r) + 1 for r in rows)
    if total <= budget:
        return "\n".join(rows)
    k = total / budget
    return "\n".join(rows[int(i * k)] for i in range(int(len(rows) / k)))


def chapters(brain, lines, rmap):
    """[{"t": seconds in the long video, "name"}] or [] (YouTube needs 3+ chapters of 10+ seconds)."""
    if not lines or not rmap or rmap[-1][3] < 3 * CHAPTER_MIN_S:
        return []
    by_id = {ln["id"]: i for i, ln in enumerate(lines)}
    j = brain.ask(brain.system(_chapter_system(brain)), "TRANSCRIPT:\n" + _sample(lines, with_ids=True),
                  CHAPTER_SCHEMA, max_tokens=700)
    got = []
    for c in (j or {}).get("chapters") or []:
        i = M.line_index(lines, c.get("start"), by_id)
        name = M.clean_text(c.get("name"), 60)
        if i is not None and name:
            got.append((longvideo.to_out(rmap, lines[i]["s"]), name))
    got.sort()
    out = []
    for t, name in got:
        t = 0.0 if not out else t
        if out and (t - out[-1]["t"] < CHAPTER_MIN_S or name == out[-1]["name"]):
            continue
        out.append({"t": round(t, 2), "name": name})
    if out and rmap[-1][3] - out[-1]["t"] < CHAPTER_MIN_S:
        out.pop()
    if len(out) < 3:
        brain.note("text", "no YouTube chapters: the video did not split into 3 or more parts of 10 s or more")
        return []
    return out


def write(brain, lines, shorts_list, long_info=None, video_name="", platforms=()):
    """-> the texts.json content. shorts_list = the Shorts' .json dicts (with "n"); long_info = long.json or None."""
    brain.note("step", "writing titles, descriptions and hashtags")
    items = "\n".join(f"{s['n']}. [{s.get('type') or 'moment'}] title: {s.get('title') or ''} | on screen: "
                      f"{s.get('hook') or ''} | why: {s.get('why') or ''}" for s in shorts_list) or "(no shorts)"
    dur = (long_info or {}).get("dur")
    user = (f"THE LONG VIDEO: \"{video_name}\"" + (f" ({hhmmss(dur)} long)" if dur else "") + "\n\n"
            f"THE SHORTS MADE FROM IT:\n{items}\n\nSAMPLE OF WHAT IS SAID:\n{_sample(lines, 6000)}\n\n"
            "Write: long_titles = 3 different titles for the long YouTube video; long_description = 2-4 sentences "
            "for YouTube (what the viewer gets); long_hashtags = 5-10 hashtags; shorts = for EACH short number above: "
            "a title (at most 70 characters), a caption (1-2 sentences to post with it on TikTok / Reels / Shorts) and "
            "3-6 hashtags.")
    j = brain.ask(brain.system(_pack_system(brain)), user, PACK_SCHEMA, max_tokens=900 + 160 * len(shorts_list))
    if j is None:
        brain.note("warn", "no usable answer for the texts - they are made from the Shorts' own titles")
        j = {}
    long_tags = tags(j.get("long_hashtags"), 10)
    titles = [M.clean_text(t, 100) for t in j.get("long_titles") or [] if M.clean_text(t, 100)][:3]
    if not titles:
        titles = [video_name] + [s.get("title") for s in shorts_list[:2] if s.get("title")]
    chaps = chapters(brain, lines, (long_info or {}).get("ranges")) if long_info else []
    chap_text = "\n".join(f"{hhmmss(c['t'])} {c['name']}" for c in chaps)
    desc = _multi(j.get("long_description"))
    if not desc:
        desc = " ".join(s.get("title") or "" for s in shorts_list[:3]).strip()
    full = "\n\n".join(x for x in (desc, chap_text, " ".join(long_tags)) if x)
    by_n = {}
    for s in j.get("shorts") or []:
        try:
            by_n[int(s.get("n"))] = s
        except (TypeError, ValueError):
            continue
    out_shorts = {}
    for s in shorts_list:
        a = by_n.get(int(s["n"])) or {}
        title = M.clean_text(a.get("title"), 100) or s.get("title") or s.get("hook") or video_name
        caption = M.clean_text(a.get("caption"), 600) or s.get("hook") or title
        ht = tags(a.get("hashtags"), 6) or long_tags[:4]
        out_shorts[str(s["n"])] = {
            "title": title, "caption": caption, "hashtags": ht,
            "copy": {"shorts": {"title": title[:100], "description": f"{caption}\n\n{' '.join(ht + ['#Shorts'])}"},
                     "tiktok": f"{caption} {' '.join(ht[:5])}".strip(),
                     "reels": f"{caption}\n\n{' '.join(ht)}".strip()}}
    brain.note("text", f"texts written: {len(titles)} titles for the long video, {len(chaps)} chapters, "
                       f"texts for {len(out_shorts)} Shorts")
    return {"made": datetime.now().isoformat(timespec="seconds"), "language": brain.language,
            "platforms": list(platforms or []),
            "long": {"titles": titles, "description": desc, "hashtags": long_tags, "chapters": chaps,
                     "chapters_text": chap_text, "copy": {"description": full}, "has_video": bool(long_info)},
            "shorts": out_shorts}

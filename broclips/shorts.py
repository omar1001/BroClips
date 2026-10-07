"""Vertical Shorts, 1080x1920 (SPEC §7.1). Port and generalisation of LoL Clips render.short, _caption_chunks,
_caption_lines, _caption_words, _out_range, the ASS header, layout C and segments.py.

A Short is a list of pieces of the recording, [[start, end, speed], ...] (source seconds); its own timeline skips
what is cut out. Layouts:
  follow  a 9:16 window of the frame follows the action (camera.py)                — gameplay, talking heads
  zoom    a 3:4 window (810x1080 of a 1080p frame) that follows the action, shown x1.33 on a blurred copy of
          itself (LoL "layout C")                                                   — gameplay
  fit     the whole frame, full width, on a blurred darker copy; the title above it, captions below it
          (nothing is ever cut off)                                                 — screen recordings, lessons
The always-blur boxes (Settings) are applied to every piece FIRST, so no crop can ever show them.
Captions: phrases of 2-6 words (transcript.phrases), the word being said in yellow (one subtitle event per word —
libass keeps right-to-left order and Arabic letter joining across the colour change, checked in LoL Clips), Latin in
Montserrat ExtraBold (the static file: libass cannot pick ExtraBold out of the variable font), Arabic in Lalezar,
chosen per caption by its script. Texts stay out of the phone apps' covered bands (top ~250 px, bottom ~480 px).
ffmpeg runs in the data folder so the subtitle file and the fonts folder are short relative paths in the filter
graph (a "C:" there needs fragile escaping — LoL lesson). Every program runs without a window."""
import os
import re
import subprocess
import threading
from pathlib import Path

from . import camera, config, media
from . import transcript as T
from .util import NO_WINDOW, finish, load_json, log, part_path, save_json

LAYOUTS = ("follow", "zoom", "fit")
AUTO_LAYOUT = {"gameplay": "follow", "camera": "follow", "podcast": "follow", "screen": "fit", "other": "fit"}
OUT_W, OUT_H = 1080, 1920
HOOK_TOP = 262                  # top edge of the title (the apps cover ~110-250 px at the top)
CAPTION_BOTTOM = 480            # captions end this far above the bottom (the apps cover ~300-480 px there)
ZOOM_Y, ZOOM_H = 240, 1440      # layout "zoom": the 3:4 window shown 1080x1440 at y 240
FIT_Y = 560                     # layout "fit": a wide picture's top edge (title above it, two caption lines below)
TITLE_S = 4.5                   # follow / zoom: the title is on top for the first seconds (fit: the whole Short)
LATIN_FONT, ARABIC_FONT = "Montserrat ExtraBold", "Lalezar"
CAP_BASE, CAP_NOW = "&H00FFFFFF&", "&H0000FFFF&"     # caption words white, the word being said yellow
LOCK = threading.RLock()        # read-modify-write of a short's .json (the server and the work line both write)

ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ---------- pieces of the recording (LoL segments.py) ----------
MIN_PIECE = 0.05


def of(clip):
    return [[float(a), float(b), float(sp)] for a, b, sp in (clip.get("segments") or [[clip["s"], clip["e"], 1.0]])]


def normalize(segs):
    segs = sorted([[float(a), float(b), float(sp)] for a, b, sp in segs if b - a > MIN_PIECE], key=lambda x: x[0])
    out = []
    for a, b, sp in segs:
        if out and a < out[-1][1]:            # overlaps the previous piece: keep only the new seconds
            a = out[-1][1]
            if b - a <= MIN_PIECE:
                continue
        if out and abs(out[-1][1] - a) < 1e-3 and abs(out[-1][2] - sp) < 1e-6:
            out[-1][1] = b
        else:
            out.append([a, b, sp])
    return out


def out_len(segs):
    return sum((b - a) / sp for a, b, sp in segs)


def src_to_out(segs, x):
    """Source time -> time in the Short, or None if that moment was cut out."""
    t = 0.0
    for a, b, sp in segs:
        if a <= x < b:
            return t + (x - a) / sp
        t += (b - a) / sp
    return None


def out_range(segs, a, b):
    """Source range [a, b] -> (start, end) in the Short's own time (cut bits skipped), or None."""
    t, lo, hi = 0.0, None, None
    for x, y, sp in segs:
        ia, ib = max(a, x), min(b, y)
        if ib > ia:
            oa, ob = t + (ia - x) / sp, t + (ib - x) / sp
            lo, hi = (oa if lo is None else lo), ob
        t += (y - x) / sp
    return (lo, hi) if lo is not None else None


def part_starts(segs):
    """Indexes of pieces that start a new part (the recording jumps before them) — captions never cross those."""
    return [k for k in range(1, len(segs)) if segs[k][0] - segs[k - 1][1] > 0.5]


# ---------- which layout ----------
def layout_for(project, video=None):
    """The video's own choice, else the project's, else the best one for the video type."""
    for v in ((video or {}).get("layout"), (project or {}).get("layout")):
        if v in LAYOUTS:
            return v
    return AUTO_LAYOUT.get((project or {}).get("video_type"), "fit")


def window_w(layout, w, h):
    """Width of the moving window in video pixels (follow: 9:16, zoom: 3:4), or None when the frame is too narrow
    for it (then the Short uses "fit")."""
    ratio = {"follow": 9 / 16, "zoom": 3 / 4}.get(layout)
    if not ratio:
        return None
    cw = int(round(h * ratio / 2)) * 2
    return cw if cw < w - 8 else None


def fit_box(w, h):
    """layout fit: (picture width, height, x, y) inside 1080x1920, and where the captions go."""
    if w >= h:
        pw, ph = OUT_W, int(round(OUT_W * h / w / 2)) * 2
    else:
        ph = OUT_H
        pw = int(round(OUT_H * w / h / 2)) * 2
        if pw > OUT_W:
            pw, ph = OUT_W, int(round(OUT_W * h / w / 2)) * 2
    py = FIT_Y if ph <= 660 else max(0, (OUT_H - ph) // 2)
    return pw, ph, (OUT_W - pw) // 2, py


def out_fps(fps):
    fps = float(fps or 0)
    if fps <= 1:
        return 30
    return 60 if fps >= 50 else max(10, int(round(fps)))


# ---------- captions ----------
def caption_words(words, segs, fixes=None):
    """The transcript words of this Short on its own timeline: cut bits dropped, `part` = which piece of the
    recording it belongs to (a jump starts a new part), the creator's caption fixes applied (a fix replaces the words
    of one phrase; an empty fix removes the caption there). Each word keeps its source times (src_s, src_e)."""
    ws = [dict(w) for w in words]
    for fx in fixes or []:
        try:
            a, b = float(fx["a"]), float(fx["b"])
        except (KeyError, TypeError, ValueError):
            continue
        toks = str(fx.get("text") or "").split()
        old = [w for w in ws if a <= w["s"] < b and src_to_out(segs, w["s"]) is not None]
        rest = [w for w in ws if not a <= w["s"] < b]
        new = []
        if toks:
            if not old:
                old = [{"s": a, "e": b}]
            slots = {}
            for k, t in enumerate(toks):     # the new words take the time slots of the words that were there
                slots.setdefault(min(len(old) - 1, k * len(old) // len(toks)), []).append(t)
            for j, ts_ in slots.items():
                w0 = old[j]
                step = max(0.05, (w0["e"] - w0["s"]) / len(ts_))
                new += [{"w": t, "s": w0["s"] + q * step, "e": w0["s"] + (q + 1) * step, "p": 1.0}
                        for q, t in enumerate(ts_)]
        ws = sorted(rest + new, key=lambda w: w["s"])
    jumps = part_starts(segs)
    out = []
    for w in ws:
        a = src_to_out(segs, w["s"])
        b = src_to_out(segs, max(w["s"], w["e"] - 0.01))
        if a is None or b is None:
            continue
        k = next(i for i, (x, y, _) in enumerate(segs) if x <= w["s"] < y)
        out.append({"w": w["w"], "s": round(a, 3), "e": round(max(b, a + 0.05), 3),
                    "part": sum(1 for j in jumps if j <= k), "src_s": w["s"], "src_e": w["e"]})
    return out


def _esc(text):
    text = re.sub(r"\{[^}]*\}", "", str(text or ""))
    return text.replace("\\", "").replace("{", "").replace("}", "").replace("\n", " ").strip()


def ass_time(sec):
    sec = max(0.0, float(sec))
    cs = int(round(sec * 100))
    h, rest = divmod(cs, 360000)
    m, rest = divmod(rest, 6000)
    s, c = divmod(rest, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _part_ends(segs):
    """{part: end time in the Short} — a caption is never shown past the cut that ends its part."""
    jumps = set(part_starts(segs))
    ends, t, part = {}, 0.0, 0
    for k, (a, b, sp) in enumerate(segs):
        if k in jumps:
            part += 1
        t += (b - a) / sp
        ends[part] = t
    return ends


def caption_events(phr, dur, part_ends=None):
    """One subtitle event per spoken word: the whole phrase on screen, the word being said yellow."""
    out = []
    for i, p in enumerate(phr):
        a = p["s"]
        nxt = phr[i + 1]["s"] if i + 1 < len(phr) else None
        b = p["words"][-1]["e"] + 0.6
        if nxt is not None:
            b = min(b, nxt)
        b = min(b, dur, (part_ends or {}).get(p.get("part", 0), dur))
        if b - a < 0.3 and (nxt is None or nxt - a >= 0.3):
            b = min(a + 0.3, dur)
        toks = [_esc(w["w"]) for w in p["words"]]
        style = "CapAr" if any(T.is_arabic(t) for t in toks) else "CapLat"
        for k, w in enumerate(p["words"]):
            t0 = a if k == 0 else max(a, w["s"])
            t1 = b if k + 1 == len(p["words"]) else min(b, max(p["words"][k + 1]["s"], t0 + 0.05))
            if t1 - t0 < 0.02 or not toks[k]:
                continue
            txt = " ".join(f"{{\\c{CAP_NOW}}}{t}{{\\c{CAP_BASE}}}" if j == k else t for j, t in enumerate(toks) if t)
            out.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},{style},,0,0,0,,{txt}")
    return out


def ass_file(layout, w, h, dur, phr, hook, part_ends=None):
    """The whole subtitle file of a Short (title + captions)."""
    if layout == "fit":
        _, ph, _, py = fit_box(w, h)
        below = py + ph + 26                  # two caption lines (~240 px in Arabic) must end above the covered band
        cap_align, cap_v = (8, below) if below + 210 <= OUT_H - CAPTION_BOTTOM else (2, CAPTION_BOTTOM)
    else:
        cap_align, cap_v = 2, CAPTION_BOTTOM
    sty = ("Style: {n},{f},{s},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,{o},2,{al},"
           "{m},{m},{v},-1")
    styles = "\n".join([
        sty.format(n="CapLat", f=LATIN_FONT, s=88, o=8, al=cap_align, m=90, v=cap_v),
        sty.format(n="CapAr", f=ARABIC_FONT, s=104, o=8, al=cap_align, m=90, v=cap_v),
        sty.format(n="HookLat", f=LATIN_FONT, s=80, o=7, al=8, m=70, v=HOOK_TOP),
        sty.format(n="HookAr", f=ARABIC_FONT, s=94, o=7, al=8, m=70, v=HOOK_TOP),
    ])
    lines = [ASS_HEAD.format(w=OUT_W, h=OUT_H, styles=styles)]
    hook = _esc(hook)
    if hook:
        end = dur if layout == "fit" else min(dur, TITLE_S)
        style = "HookAr" if T.is_arabic(hook) else "HookLat"
        fade = "" if layout == "fit" else "{\\fad(0,350)}"
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(end)},{style},,0,0,0,,{fade}{hook}")
    lines += caption_events(phr, dur, part_ends)
    return "\n".join(lines) + "\n"


# ---------- the ffmpeg filter graph ----------
def _atempo(sp):
    if abs(sp - 1) < 1e-6:
        return "anull"
    parts, left = [], sp
    while left > 2.0:                     # atempo accepts at most 2.0 per filter
        parts.append("atempo=2.0")
        left /= 2.0
    parts.append(f"atempo={left:.4f}")
    return ",".join(parts)


BG_BLUR = "scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=10:2,scale=1080:1920"


def filtergraph(segs, n_audio, layout, w, h, fps, blur, xexpr, ass, fonts, dur):
    """The -filter_complex text of a Short (inputs: one `-ss a -t d -i src` per piece)."""
    fc = []
    jumps = set(part_starts(segs))
    for k, (a, b, sp) in enumerate(segs):
        d = (b - a) / sp
        fc.append(f"[{k}:v]{blur + ',' if blur else ''}setpts=(PTS-STARTPTS)/{sp:g}[v{k}]")
        a_in = ("".join(f"[{k}:a:{j}]" for j in range(n_audio)) + f"amix=inputs={n_audio}:normalize=0,"
                if n_audio > 1 else f"[{k}:a]")
        fx = (",afade=t=in:d=0.08" if k in jumps else "") + \
             (f",afade=t=out:st={max(0.0, d - 0.08):.3f}:d=0.08" if k + 1 in jumps else "")
        fc.append(f"{a_in}asetpts=PTS-STARTPTS,aresample=48000,{_atempo(sp)}{fx}[a{k}]")
    fc.append("".join(f"[v{k}][a{k}]" for k in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=1[vc][ac]")
    if layout == "follow":
        cw = window_w("follow", w, h)
        fc.append(f"[vc]crop=w={cw}:h={h}:x={xexpr}:y=0,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[vv]")
    elif layout == "zoom":
        cw = window_w("zoom", w, h)
        fc.append(f"[vc]crop=w={cw}:h={h}:x={xexpr}:y=0,split=2[g0][b0]")
        fc.append(f"[g0]scale={OUT_W}:{ZOOM_H}:flags=lanczos[fg]")
        fc.append(f"[b0]{BG_BLUR},eq=brightness=-0.12:saturation=1.2[bg]")
        fc.append(f"[bg][fg]overlay=0:{ZOOM_Y},setsar=1[vv]")
    else:
        pw, ph, px, py = fit_box(w, h)
        fc.append("[vc]split=2[f0][b0]")
        fc.append(f"[f0]scale={pw}:{ph}:flags=lanczos[fg]")
        fc.append(f"[b0]{BG_BLUR},eq=brightness=-0.25:saturation=1.1[bg]")
        fc.append(f"[bg][fg]overlay={px}:{py},setsar=1[vv]")
    fc.append(f"[vv]fps={fps},ass=filename={ass}:fontsdir={fonts},format=yuv420p[v]")
    # soft start / end so the sound never clicks in or out mid-word; -14 LUFS like the platforms
    fc.append(f"[ac]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,afade=t=in:d=0.12,"
              f"afade=t=out:st={max(0.0, dur - 0.35):.3f}:d=0.35[a]")
    return ";".join(fc)


def venc(enc, fps, long=False):
    """Encoder settings: NVIDIA (NVENC, -cq 21 / long 23) or the CPU (libx264, -crf 20 / long 21)."""
    gop = str(int(2 * max(10, fps)))
    if enc == "h264_nvenc":
        rate = ["-b:v", "10M", "-maxrate", "16M", "-bufsize", "24M"] if long else \
            ["-b:v", "7M", "-maxrate", "10M", "-bufsize", "14M"]
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23" if long else "21", *rate,
                "-profile:v", "high", "-g", gop]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21" if long else "20", "-profile:v", "high",
            "-g", gop]


AENC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]


def fg_path(p, cwd):
    """A path for inside a filter graph: relative to ffmpeg's working folder when possible (no 'C:' to escape)."""
    try:
        rel = Path(os.path.relpath(str(p), str(cwd))).as_posix()
        if not rel.startswith("..") and re.fullmatch(r"[A-Za-z0-9_./-]+", rel):
            return rel
    except ValueError:                       # another drive
        pass
    # quoted: the filter-graph level keeps everything inside '…', the option level then reads "\:" as ":"
    return "'" + Path(p).resolve().as_posix().replace(":", "\\:") + "'"


def run_ffmpeg(cmd, cwd=None, tick=None):
    """Run ffmpeg without a window. tick() is called about every second while it works (it is where Stop / pause
    take effect: if it raises, ffmpeg is stopped and the exception goes on)."""
    try:
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                             creationflags=NO_WINDOW)
    except FileNotFoundError:
        raise RuntimeError(media.FFMPEG_MISSING) from None
    while True:
        try:
            _, err = p.communicate(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if tick:
                try:
                    tick()
                except BaseException:
                    p.kill()
                    p.communicate()
                    raise
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({p.returncode}): {(err or '')[-1500:]}")


def encode(cmd_for, enc, cwd, tick):
    """Run an encode; if the NVIDIA encoder fails (busy / old driver), once more on the CPU."""
    try:
        run_ffmpeg(cmd_for(enc), cwd, tick)
        return enc
    except RuntimeError as ex:
        msg = str(ex).lower()
        if enc != "h264_nvenc" or ("nvenc" not in msg and "cuda" not in msg):
            raise
        log.warning("NVIDIA encoder failed (%s) - using the CPU encoder", str(ex)[-300:])
        run_ffmpeg(cmd_for("libx264"), cwd, tick)
        return "libx264"


def render(src, info, short, words, out_mp4, ass_path, layout, boxes=None, enc=None, tick=None):
    """Make one vertical Short. info = media.probe(src). short = {"segments", "hook", "caption_fixes"}.
    -> what the editor needs: {"phrases": [{"a", "b", "text"}] (source times), "dur", "layout"}"""
    segs = normalize(of(short))
    if not segs:
        raise ValueError("This Short has no time left in it.")
    dur = out_len(segs)
    w, h = int(info.get("width") or 1920), int(info.get("height") or 1080)
    n_audio = max(1, int(info.get("n_audio") or 1))
    fps = out_fps(info.get("fps"))
    if layout not in LAYOUTS or (layout != "fit" and not window_w(layout, w, h)):
        layout = "fit"
    xexpr = "0"
    if layout != "fit":
        cw = window_w(layout, w, h)
        xexpr = camera.crop_x(src, segs, cw, w, h, boxes)
    cwords = caption_words(words, segs, short.get("caption_fixes"))
    phr = T.phrases(cwords)
    cwd = config.data_dir()
    fonts = Path(config.assets_dir()) / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    ass_path = Path(ass_path)
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(ass_file(layout, w, h, dur, phr, short.get("hook") or "", _part_ends(segs)),
                        encoding="utf-8-sig")
    blur = media.blur_filter(boxes, w, h)
    fc = filtergraph(segs, n_audio, layout, w, h, fps, blur, xexpr, fg_path(ass_path, cwd), fg_path(fonts, cwd), dur)
    ins = []
    for a, b, _ in segs:
        ins += ["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(src)]
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path(out_mp4)

    def cmd(e):
        return ["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                *venc(e, fps), *AENC, "-movflags", "+faststart", str(tmp)]
    try:
        encode(cmd, enc or media.encoder(), cwd, tick)
        finish(tmp, out_mp4)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    edit = []
    for k, p in enumerate(phr):
        a = p["words"][0]["src_s"]
        b = p["words"][-1]["src_e"] + 0.01
        if k + 1 < len(phr):
            b = min(b, phr[k + 1]["words"][0]["src_s"])
        edit.append({"a": round(a, 3), "b": round(max(b, a + 0.01), 3), "t": round(p["s"], 2),
                     "text": " ".join(w["w"] for w in p["words"])})
    return {"phrases": edit, "dur": round(dur, 2), "layout": layout}


# ---------- Shorts on disk ----------
def paths(vdir, n, folder="shorts"):
    base = Path(vdir) / folder / f"short_{int(n):02d}"
    return base.with_suffix(".mp4"), base.with_suffix(".json")


def load(vdir, n, folder="shorts"):
    return load_json(paths(vdir, n, folder)[1])


def save(vdir, n, data, folder="shorts"):
    with LOCK:
        save_json(data, paths(vdir, n, folder)[1])


def listing(vdir, folder="shorts"):
    """Every Short of a video (its .json + file facts), in number order."""
    out = []
    d = Path(vdir) / folder
    if not d.is_dir():
        return out
    for f in sorted(d.glob("short_*.json")):
        m = re.fullmatch(r"short_(\d+)\.json", f.name)
        data = load_json(f)
        if not m or not isinstance(data, dict):
            continue
        mp4 = f.with_suffix(".mp4")
        data = dict(data, n=int(m.group(1)), has_file=mp4.is_file())
        if data["has_file"]:
            data["mtime"] = int(mp4.stat().st_mtime)
        out.append(data)
    return out


def from_moment(m, n, layout):
    """A Short's .json, made from a found moment."""
    segs = normalize(of(m))
    return {"n": n, "segments": [[round(a, 3), round(b, 3), sp] for a, b, sp in segs],
            "s": round(segs[0][0], 3), "e": round(segs[-1][1], 3), "title": m.get("title") or "",
            "hook": m.get("hook") or m.get("title") or "", "why": m.get("why") or "", "type": m.get("type") or "",
            "score": m.get("score"), "rank": m.get("rank"), "marathon": m.get("marathon"),
            "hinted": bool(m.get("hinted")), "src": m.get("src") or "", "layout": layout, "mark": "",
            "caption_fixes": [], "rev": 0, "version": 0, "dirty": False}

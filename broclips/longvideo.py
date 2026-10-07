"""The clean long video for YouTube (SPEC §7.2), from the parts idea of LoL Clips render.full_video: keep every
stretch with speech, cut silences longer than the project's setting (default 1.2 s, keeping 0.25 s on each side),
join with short sound fades so a cut never clicks. "Jump cuts: off" keeps everything.

The kept ranges are encoded a few at a time (one ffmpeg run joins up to PER_RUN of them with the concat filter, each
read with its own fast seek), then all runs are joined without re-encoding the picture (concat copy) and the sound is
levelled to -14 LUFS. A stopped / paused job keeps the finished runs and continues with the next one.
long.json keeps the kept ranges with their times in the source AND in the long video -> YouTube chapters."""
import hashlib
import json
import math
import shutil
from pathlib import Path

from . import media, shorts
from .util import finish, load_json, log, part_path, save_json

PAD_S = 0.25            # kept before / after the words around a cut
FADE_S = 0.08           # sound fade at every cut
PER_RUN = 8             # kept ranges per ffmpeg run


def plan(words, total_s, silence_s=1.2, pad=PAD_S, jump_cuts=True, fps=30):
    """Which parts of the recording the long video keeps -> [[start, end], ...] (source seconds, on frame
    boundaries). Silences (between words) longer than silence_s are cut, `pad` seconds stay on each side."""
    total_s = float(total_s)
    if not jump_cuts or not words or total_s <= 0:
        return [[0.0, round(total_s, 3)]]
    ws = sorted(words, key=lambda w: w["s"])
    ranges = []
    a = 0.0 if ws[0]["s"] <= silence_s else max(0.0, ws[0]["s"] - pad)
    prev_e = ws[0]["e"]
    for w in ws[1:]:
        if w["s"] - prev_e > silence_s:
            ranges.append([a, min(total_s, prev_e + pad)])
            a = max(0.0, w["s"] - pad)
        prev_e = max(prev_e, w["e"])
    ranges.append([a, total_s if total_s - prev_e <= silence_s else min(total_s, prev_e + pad)])
    fps = max(1.0, float(fps or 30))
    out = []
    for a, b in ranges:
        a, b = math.floor(a * fps) / fps, min(total_s, math.ceil(b * fps) / fps)
        if out and a <= out[-1][1] + 1e-6:          # paddings touch: one range
            out[-1][1] = max(out[-1][1], b)
        elif b - a >= 0.1:
            out.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in out]


def mapping(ranges):
    """[[a, b, out_a, out_b], ...]: where each kept range sits in the long video."""
    out, t = [], 0.0
    for a, b in ranges:
        out.append([a, b, round(t, 3), round(t + (b - a), 3)])
        t += b - a
    return out


def to_out(rmap, t):
    """Source time -> time in the long video (a time inside a cut -> where the video continues after it)."""
    for a, b, oa, ob in rmap:
        if t < a:
            return oa
        if t <= b:
            return round(oa + (t - a), 3)
    return rmap[-1][3] if rmap else 0.0


def filtergraph(n, durs, n_audio, blur, fps):
    """-filter_complex for one run of `n` kept ranges (inputs: one `-ss a -t d -i src` per range)."""
    fc = []
    for k in range(n):
        d = durs[k]
        fc.append(f"[{k}:v]{blur + ',' if blur else ''}fps={fps},setpts=PTS-STARTPTS[v{k}]")
        a_in = ("".join(f"[{k}:a:{j}]" for j in range(n_audio)) + f"amix=inputs={n_audio}:normalize=0,"
                if n_audio > 1 else f"[{k}:a]")
        fc.append(f"{a_in}aresample=48000,asetpts=PTS-STARTPTS,afade=t=in:d={FADE_S},"
                  f"afade=t=out:st={max(0.0, d - FADE_S):.3f}:d={FADE_S}[a{k}]")
    fc.append("".join(f"[v{k}][a{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=1[vc][a]")
    fc.append("[vc]format=yuv420p[v]")
    return ";".join(fc)


def render(src, info, ranges, out_mp4, boxes=None, enc=None, tick=None, step=None):
    """Encode the long video. step(done, total) after every run; tick() while ffmpeg works (Stop / pause)."""
    out_mp4 = Path(out_mp4)
    work = out_mp4.parent / "_long_parts"
    w, h = int(info.get("width") or 1920), int(info.get("height") or 1080)
    fps = shorts.out_fps(info.get("fps"))
    n_audio = max(1, int(info.get("n_audio") or 1))
    blur = media.blur_filter(boxes, w, h)
    key = hashlib.sha1(json.dumps([ranges, blur, fps, str(src)]).encode("utf-8")).hexdigest()[:12]
    saved = load_json(work / "plan.json") or {}
    if saved.get("key") != key:                       # another plan: the old runs are useless
        shutil.rmtree(work, ignore_errors=True)
        saved = {"key": key}
    enc = saved.get("enc") or enc or media.encoder()  # all runs must come from the same encoder (concat copy)
    work.mkdir(parents=True, exist_ok=True)
    save_json(dict(saved, enc=enc), work / "plan.json")
    groups = [ranges[i:i + PER_RUN] for i in range(0, len(ranges), PER_RUN)]
    names = []
    for gi, group in enumerate(groups):
        part = work / f"p{gi:04d}.mov"
        names.append(part.name)
        if part.exists():
            continue                                 # made before a pause
        ins = []
        for a, b in group:
            ins += ["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(src)]
        fc = filtergraph(len(group), [b - a for a, b in group], n_audio, blur, fps)
        tmp = part_path(part)

        def cmd(e, ins=ins, fc=fc, tmp=tmp):
            return ["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                    *shorts.venc(e, fps, long=True), "-c:a", "pcm_s16le", "-f", "mov", str(tmp)]
        used = shorts.encode(cmd, enc, None, tick)
        if used != enc:                              # the GPU encoder failed: every run again on the CPU
            if gi:
                shutil.rmtree(work, ignore_errors=True)
                return render(src, info, ranges, out_mp4, boxes, used, tick, step)
            enc = used
            save_json({"key": key, "enc": enc}, work / "plan.json")
        finish(tmp, part)
        if step:
            step(gi + 1, len(groups))
    (work / "list.txt").write_text("".join(f"file '{n}'\n" for n in names), encoding="utf-8")
    tmp = part_path(out_mp4)
    try:
        shorts.run_ffmpeg(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"),
                           "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", *shorts.AENC,
                           "-movflags", "+faststart", str(tmp)], None, tick)
        finish(tmp, out_mp4)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    for _ in range(5):                                # a fresh file can be held a moment by a virus scanner
        shutil.rmtree(work, ignore_errors=True)
        if not work.exists():
            break
    log.info("long video: %d kept ranges, %.1f of %.1f min", len(ranges), sum(b - a for a, b in ranges) / 60,
             float(info.get("dur") or 0) / 60)
    return out_mp4

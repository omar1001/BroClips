"""Video and audio helpers on top of ffmpeg / ffprobe (SPEC §3, §6 step 1): probe a file, 16 kHz audio for listening,
single frames, loudness spikes (a hint for exciting moments), the H.264 encoder to use, the always-blur filter, and
audio pieces for cloud listening. Source videos are only ever READ (SPEC §2.1): ffmpeg gets them as an input only.
Every program runs through util.run / util.run_bytes, i.e. without a console window (SPEC §2.4)."""
import json
import re
import wave
from pathlib import Path

import numpy as np

from . import config
from .util import finish, log, part_path, run, run_bytes

FFMPEG_MISSING = ("ffmpeg is not installed (or not on PATH). Run install.bat again, or install it with: "
                  "winget install --id Gyan.FFmpeg -e")


def _run(cmd, **kw):
    try:
        return run(cmd, **kw)
    except FileNotFoundError:
        raise RuntimeError(FFMPEG_MISSING) from None


def _run_bytes(cmd, **kw):
    try:
        return run_bytes(cmd, **kw)
    except FileNotFoundError:
        raise RuntimeError(FFMPEG_MISSING) from None


def _rate(text):
    """"60000/1001" -> 59.94; "0/0" or junk -> 0.0"""
    try:
        a, _, b = str(text or "").partition("/")
        return float(a) / float(b or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path):
    """-> {"dur", "fps", "width", "height", "n_audio", "has_video", "codec"} (codec = the video codec name).
    Cover pictures inside audio files do not count as video."""
    try:
        p = _run(["ffprobe", "-v", "error", "-show_entries",
                  "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,"
                  "duration:stream_disposition=attached_pic", "-of", "json", str(path)])
    except RuntimeError as ex:
        if str(ex) == FFMPEG_MISSING:
            raise
        raise RuntimeError(f"Cannot read {Path(path).name} as a video or audio file. ({str(ex)[-300:]})") from None
    j = json.loads(p.stdout or "{}")
    streams = j.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    n_audio = sum(1 for s in streams if s.get("codec_type") == "audio")
    try:
        dur = float((j.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        dur = max([_float(s.get("duration")) for s in streams] or [0.0])
    fps = 0.0
    if video:
        fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    return {"dur": round(dur, 3), "fps": round(fps, 3),
            "width": int(video.get("width") or 0) if video else 0,
            "height": int(video.get("height") or 0) if video else 0,
            "n_audio": n_audio, "has_video": video is not None,
            "codec": (video or {}).get("codec_name", "")}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def extract_audio16k(src, out_wav, n_audio=None):
    """The sound of `src` as a 16 kHz mono WAV for listening. Several audio tracks (e.g. game + microphone) are
    mixed (amix, levels kept) so nothing said is lost. Written to a .part file first, renamed when complete."""
    n = probe(src)["n_audio"] if n_audio is None else int(n_audio)
    if n < 1:
        raise RuntimeError(f"{Path(src).name} has no sound, so there is nothing to listen to.")
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path(out_wav)
    if n == 1:
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", str(tmp)]
    else:
        ins = "".join(f"[0:a:{i}]" for i in range(n))
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
               f"{ins}amix=inputs={n}:normalize=0,aresample=16000[a]", "-map", "[a]", "-ac", "1",
               "-c:a", "pcm_s16le", str(tmp)]
    _run(cmd)
    finish(tmp, out_wav)
    return out_wav


def frame_jpg(src, t, width=None, vf=None, q=3):
    """One frame at `t` seconds as JPEG bytes (optionally filtered by `vf` first, then scaled to `width`, keeping the
    shape), or None when there is no frame there."""
    filters = [f for f in (vf, f"scale={int(width)}:-2" if width else None) if f]
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, float(t)):.3f}", "-i", str(src), "-frames:v", "1"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-q:v", str(int(q)), "-f", "image2pipe", "-c:v", "mjpeg", "-"]
    data = _run_bytes(cmd, timeout=120)
    return data if data[:2] == b"\xff\xd8" else None


def frame_rgb(src, t, w, h):
    """One frame at `t` seconds scaled to w×h as a numpy uint8 array (h, w, 3), or None when there is no frame."""
    w, h = int(w), int(h)
    data = _run_bytes(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, float(t)):.3f}", "-i", str(src),
                       "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                      timeout=120)
    if len(data) < w * h * 3:
        return None
    return np.frombuffer(data[:w * h * 3], np.uint8).reshape(h, w, 3).copy()


def loudness_spikes(wav, jump_lu=8.0):
    """Per-second momentary loudness (LUFS) and the seconds that jump >= jump_lu above the rolling 30 s median —
    shouting, laughing, a sudden noise (ported from LoL Clips audio.loudness_spikes). -> (per_second, spikes)"""
    p = _run(["ffmpeg", "-v", "info", "-nostats", "-i", str(wav), "-af",
              "ebur128=metadata=1,ametadata=print:key=lavfi.r128.M", "-f", "null", "-"])
    times, vals, t = [], [], None
    for line in p.stderr.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r"lavfi\.r128\.M=(-?[\d.]+|-inf)", line)
        if m and t is not None:
            v = m.group(1)
            times.append(t)
            vals.append(-70.0 if v == "-inf" else max(-70.0, float(v)))
    if not times:
        return [], []
    n = int(times[-1]) + 1
    per_s = np.full(n, -70.0)
    for tt, v in zip(times, vals):
        i = int(tt)
        per_s[i] = max(per_s[i], v)
    spikes = []
    for i in range(n):
        lo, hi = max(0, i - 15), min(n, i + 15)
        med = np.median(per_s[lo:hi])
        if per_s[i] - med >= jump_lu and per_s[i] > -35:
            spikes.append(i)
    return [float(x) for x in per_s], spikes


_ENCODER = {}


def encoder(choice=None):
    """The H.264 encoder for renders: "h264_nvenc" (NVIDIA GPU, fast) when a 1-frame test encode works, else
    "libx264" (CPU, works everywhere). `choice` (default: Settings "encoder") = auto | nvenc | x264. The test runs
    once per app run."""
    choice = str(choice or config.settings().get("encoder") or "auto").lower()
    if choice in ("x264", "libx264", "cpu"):
        return "libx264"
    if choice in ("nvenc", "h264_nvenc", "gpu"):
        return "h264_nvenc"
    if "auto" not in _ENCODER:
        try:
            ok = _run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1:r=10",
                       "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                      check=False, timeout=60).returncode == 0
        except Exception:
            ok = False
        _ENCODER["auto"] = "h264_nvenc" if ok else "libx264"
        log.info("video encoder: %s", _ENCODER["auto"])
    return _ENCODER["auto"]


def blur_filter(boxes, width=1920, height=1080):
    """ffmpeg filter that hides the always-blur boxes (SPEC §5: [x, y, w, h] in 1920×1080 coordinates) on a
    width×height frame: one `delogo` per box (it repaints the box from the pixels around it), as LoL Clips does for
    the Windows watermark. delogo refuses a box touching the frame edge (measured with ffmpeg 2026-06: it needs
    x >= 1, y >= 1, x+w <= W-1, y+h <= H-1) and takes no expressions, so boxes are scaled here and kept 2 px inside.
    "" when there is nothing to hide."""
    sx, sy = int(width) / 1920.0, int(height) / 1080.0
    out = []
    for b in boxes or []:
        try:
            x, y, w, h = (float(v) for v in list(b)[:4])
        except (TypeError, ValueError):
            continue
        x0, y0 = max(2, round(x * sx)), max(2, round(y * sy))
        x1, y1 = min(int(width) - 2, round((x + w) * sx)), min(int(height) - 2, round((y + h) * sy))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            out.append(f"delogo=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}")
    return ",".join(out)


# ---------- audio pieces for cloud listening ----------
_CODECS = {"mp3": ["-c:a", "libmp3lame", "-b:a", "48k"],
           "ogg": ["-c:a", "libopus", "-b:a", "32k"],
           "wav": ["-c:a", "pcm_s16le"]}


def _wav_info(wav):
    with wave.open(str(wav), "rb") as w:
        return w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()


def _quiet_point(wav, lo, hi, sr, ch, sw):
    """The middle of the quietest 0.3 s between lo and hi seconds (the latest one on a tie), so a cut there does
    not split a word. Falls back to hi."""
    if sw != 2 or hi - lo < 0.5:
        return hi
    with wave.open(str(wav), "rb") as w:
        w.setpos(int(lo * sr))
        raw = w.readframes(int((hi - lo) * sr))
    x = np.frombuffer(raw[: len(raw) // (2 * ch) * 2 * ch], np.int16).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    hop = max(1, int(0.1 * sr))
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    if len(frames) < 3:
        return hi
    energy = np.convolve((frames ** 2).mean(axis=1), np.ones(3), "valid")
    k = len(energy) - 1 - int(np.argmin(energy[::-1]))
    return lo + (k + 1.5) * hop / sr


def split_audio(wav, out_dir, piece_s=600.0, fmt="mp3", search_s=20.0):
    """Cut a WAV into pieces of at most `piece_s` seconds for services with a size or length limit (16 kHz mono,
    mp3 48 kbit/s by default). Each cut is put in the quietest moment of the last `search_s` seconds before the
    limit, so no word is cut in half. -> [(path, offset_s, dur_s)] — add offset_s to the times a piece gives."""
    try:
        sr, n, ch, sw = _wav_info(wav)
        total = n / float(sr)
    except (wave.Error, EOFError):
        sr = ch = sw = 0
        total = probe(wav)["dur"]
    cuts = [0.0]
    while total - cuts[-1] > piece_s:
        target = cuts[-1] + piece_s
        lo = max(cuts[-1] + piece_s / 2, target - search_s)
        cuts.append(_quiet_point(wav, lo, target, sr, ch, sw) if sr else target)
    cuts.append(total)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        out = out_dir / f"piece_{i:03d}.{fmt}"
        _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(wav),
              "-vn", "-ac", "1", "-ar", "16000", *_CODECS[fmt], str(out)])
        pieces.append((str(out), round(a, 3), round(b - a, 3)))
    return pieces

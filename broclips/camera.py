"""Follow-the-action camera for the "follow" and "zoom" layouts (port of LoL Clips camera.py, without the game masks).
A vertical Short shows a window of the wide frame; this module moves that window, smoothly, to where things happen.

No AI: the action is where the picture CHANGES between two neighbouring frames (a moving person, a mouse, a fight).
A camera pan changes every column the same way and is subtracted; the always-blur boxes (Settings) are masked out.
Measured in LoL Clips on gameplay: a fixed middle window held 82-88 % of the movement, a following window 91-92 %.
The path is planned per second (dynamic programming: as much movement inside as possible, few moves), softened, with
a dead band (small differences do not start a move) and a speed limit, and written as an ffmpeg crop x-expression of
t. A cut to another part of the video is a jump. Any trouble -> the still window (a render never fails because of
the camera)."""
import hashlib
from pathlib import Path

import numpy as np

from . import config
from .util import NO_WINDOW, log

AW, FPS = 384, 4                    # analysis width (height keeps the shape) and samples per second
STAY = 1.5                          # how much the window prefers to stay where it is (0 = jumps to every movement)
DEADBAND = 70                       # (in 1920-wide pixels) smaller differences do not move the window
MAX_SPEED = 260                     # (in 1920-wide pixels) the window moves at most this far per second
_MEM = {}                           # cache key -> {second: movement per column}


def _key(src):
    p = Path(src)
    st = p.stat()
    return hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")).hexdigest()[:16]


def _store(key):
    return config.cache_dir() / "cam" / f"{key}.npz"


def _rows(key):
    """{second: movement per column} for this video, kept on disk (an edit renders the same seconds again)."""
    if key not in _MEM:
        _MEM[key] = {}
        f = _store(key)
        if f.exists():
            try:
                z = np.load(f)
                _MEM[key] = {int(s): r for s, r in zip(z["secs"], z["rows"])}
            except Exception:
                pass
    return _MEM[key]


def _save(key):
    rows = _rows(key)
    if not rows:
        return
    f = _store(key)
    f.parent.mkdir(parents=True, exist_ok=True)
    secs = sorted(rows)
    tmp = f.with_name(f.stem + ".part.npz")
    try:
        np.savez_compressed(tmp, secs=np.array(secs, np.int32), rows=np.stack([rows[s] for s in secs]).astype(np.float32))
        tmp.replace(f)
    except OSError:
        pass                                  # only a cache


def _size(w, h):
    ah = max(2, int(round(AW * h / max(1, w) / 2)) * 2)
    return AW, ah


def mask(w, h, boxes):
    """Where movement counts (analysis size): everywhere except the always-blur boxes (1920x1080 coordinates)."""
    aw, ah = _size(w, h)
    m = np.ones((ah, aw), bool)
    for b in boxes or []:
        try:
            x, y, bw, bh = (float(v) for v in list(b)[:4])
        except (TypeError, ValueError):
            continue
        x0, x1 = int(x / 1920 * aw), int(np.ceil((x + bw) / 1920 * aw))
        y0, y1 = int(y / 1080 * ah), int(np.ceil((y + bh) / 1080 * ah))
        m[max(0, y0 - 1):min(ah, y1 + 1), max(0, x0 - 1):min(aw, x1 + 1)] = False
    return m


def _measure(src, key, s0, n, w, h, boxes):
    """Movement per column for the seconds s0 .. s0+n-1 of the video."""
    import subprocess
    aw, ah = _size(w, h)
    cmd = ["ffmpeg", "-v", "error", "-ss", str(s0), "-t", str(n), "-i", str(src), "-an", "-vf",
           f"scale={aw}:{ah},format=gray,tblend=all_mode=difference,fps={FPS}", "-f", "rawvideo", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW, timeout=600).stdout
    fr = np.frombuffer(raw[: len(raw) // (aw * ah) * aw * ah], np.uint8).reshape(-1, ah, aw)
    mv = (fr > 18) & mask(w, h, boxes)                       # a pixel that changed between two neighbouring frames
    col = mv.sum(axis=1).astype(np.float32)
    col = np.maximum(0, col - np.median(col, axis=1, keepdims=True))   # a camera pan changes every column
    rows = _rows(key)
    for k in range(n):
        part = col[k * FPS:(k + 1) * FPS]
        rows[s0 + k] = part.sum(axis=0) if len(part) else np.zeros(aw, np.float32)


def _energy(src, key, a, b, w, h, boxes):
    """(first second, array seconds x columns) of movement for the video range [a, b]."""
    s0, s1 = int(a), max(int(a) + 1, int(np.ceil(b)))
    rows = _rows(key)
    k = s0
    while k < s1:                             # measure only the seconds that are not known yet, in runs
        if k in rows:
            k += 1
            continue
        j = k
        while j < s1 and j not in rows:
            j += 1
        _measure(src, key, k, j - k, w, h, boxes)
        k = j
    aw, _ = _size(w, h)
    return s0, np.stack([rows.get(s, np.zeros(aw, np.float32)) for s in range(s0, s1)])


def path(E, cw, w, hint=None, hold=None):
    """Left edge of the window (video pixels) for every second: as much movement inside as possible, few moves.
    cw = window width in video pixels, w = video width. hint = where it would like to begin (a weak wish); hold =
    where the window IS when this piece begins (the same scene goes on)."""
    n, aw = E.shape
    scale = w / 1920.0
    half = max(1, round(cw / w * aw / 2))
    cs = np.arange(half, aw - half + 1)
    if len(cs) == 0:
        return [0.0] * n
    cum = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(E, axis=1)], axis=1)
    inside = cum[:, cs + half] - cum[:, cs - half]
    gain = inside / np.maximum(E.sum(axis=1, keepdims=True), 40)        # a calm second pulls only weakly
    move = np.abs(cs[:, None] - cs[None, :]).astype(np.float32) / aw
    best = gain[0].copy()
    start = hold if hold is not None else hint
    if start is not None:
        best -= STAY * np.abs(cs - (start + cw / 2) / w * aw) / aw
    back = np.zeros((n, len(cs)), np.int32)
    for t in range(1, n):
        cand = best[None, :] - STAY * move
        back[t] = cand.argmax(axis=1)
        best = cand.max(axis=1) + gain[t]
    idx = [int(best.argmax())]
    for t in range(n - 1, 0, -1):
        idx.append(int(back[t][idx[-1]]))
    gx = cs[np.array(idx[::-1])] / aw * w - cw / 2
    gx = np.clip(gx, 0, w - cw)
    if n >= 3:                                # soften: it starts to move a little before the action has moved
        pad = np.concatenate([gx[:1], gx, gx[-1:]])
        gx = (pad[:-2] + pad[1:-1] + pad[2:]) / 3
    out, cur, moving = [], float(gx[0] if hold is None else hold), False
    for k, g in enumerate(gx):
        # small differences do not START a move; a move that started goes on until the action stops moving
        if (k or hold is not None) and (moving or abs(g - cur) >= DEADBAND * scale):
            cur += float(np.clip(g - cur, -MAX_SPEED * scale, MAX_SPEED * scale))
            moving = abs(g - cur) > 2 * scale or (k + 1 < n and abs(float(gx[k + 1]) - float(g)) > 8 * scale)
        out.append(cur)
    return out


def track(src, segs, cw, w, h, boxes=None, start=None):
    """Where the window is while the Short plays -> [(time in the Short, left edge in video pixels), ...].
    segs = the Short's pieces [[a, b, speed], ...]. Between two points the window moves evenly; two points at the
    same time are a jump (a cut to another part of the video)."""
    key = _key(src)
    pts, t0, prev_end, last = [], 0.0, None, None
    for a, b, sp in segs:
        joined = prev_end is not None and abs(a - prev_end) < 0.5       # the same scene goes on
        s0, E = _energy(src, key, a, b, w, h, boxes)
        p = path(E, cw, w, hint=start if prev_end is None else None, hold=last if joined else None)
        if not joined and pts:
            pts.append((t0, p[0]))                                     # a cut: the window jumps with the picture
        elif not pts:
            pts.append((0.0, p[0]))
        for k, g in enumerate(p):
            t_src = min(max(s0 + k + 0.5, a), b)
            pts.append((t0 + (t_src - a) / sp, g))
        t0 += (b - a) / sp
        pts.append((t0, p[-1]))
        prev_end, last = b, p[-1]
    _save(key)
    return pts


def expr(pts, w=None, cw=None):
    """The window's left edge as an ffmpeg expression of t (for crop=x=...). A still window -> a plain number."""
    lo, hi = 0, (int(w - cw) // 2 * 2) if w and cw else None

    def even(v):
        v = int(round(v / 2) * 2)
        return max(lo, min(hi, v)) if hi is not None else max(lo, v)
    x0 = even(pts[0][1])
    terms, cur = [], x0
    for (ta, _), (tb, xb) in zip(pts, pts[1:]):
        d = even(xb) - cur
        if abs(d) < 4:
            continue
        terms.append(f"{d:+d}*clip((t-{ta:.3f})/{max(0.001, tb - ta):.3f}\\,0\\,1)")
        cur += d
    return str(x0) if not terms else f"{x0}" + "".join(terms)


def crop_x(src, segs, cw, w, h, boxes=None, start=None):
    """What shorts.py puts into its crop filter. Any trouble -> the still window at `start` (default: the middle)."""
    still = int(round((w - cw) / 2 if start is None else start))
    try:
        return expr(track(src, segs, cw, w, h, boxes, start=still), w, cw)
    except Exception as ex:
        log.warning("follow camera: %s - the picture stays in the middle", ex)
        return str(max(0, still // 2 * 2))

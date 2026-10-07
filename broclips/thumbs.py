"""Thumbnails (SPEC §8): a generic port of LoL Clips' thumbnail maker (thumbs.py in the author's private app).

ONE renderer makes every thumbnail - the 3 automatic ones at the end of "Make" (pipeline step 6 calls auto()) AND every
picture of the 🎨 Thumbnail studio (web/studio.html, API in thumbs_api.py). A thumbnail is a DESIGN (a dict) saved as
thumbs/thumb_N.json next to thumbs/thumb_N.jpg in the video's folder, so the studio shows exactly the file the user
uploads, and every thumbnail can be opened and changed again.

Layers, back to front: the video's picture (Settings' "always blur" boxes are blurred FIRST, so no zoom can show
them, and the zoom steers around them) -> light rays -> subject(s) (a person or thing cut out of its background by
BiRefNet, white outline + glow; the sides the picture cut off fade out) -> circle / arrow -> big words (thick outline,
colour gradient) -> 3D emoji (Microsoft Fluent Emoji, MIT). Every layer is cached, so moving something in the studio
only stacks the layers again.

Arabic is shaped by Pillow's raqm layout (HarfBuzz is inside Pillow); on Windows raqm also needs fribidi-0.dll
(<data>/assets/bin, from assets.py), which must be on the DLL path BEFORE Pillow's font code loads - so this module
sets that up first. Any other module that draws text with Pillow must import this module first.
Coordinates in a design are fractions of the 1280x720 picture; sizes are fractions of its height."""
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import assets, config, jobs, media, projects
from .util import finish, load_json, log, part_path, programs_running, run, run_bytes, save_json

_DLL_DIRS = []                                  # keeps os.add_dll_directory handles alive


def _dll_setup():
    """fribidi-0.dll for Arabic (Windows): Pillow's raqm looks for it once, when its font module first loads."""
    if not config.IS_WINDOWS:
        return
    try:
        b = assets.bin_dir()
    except OSError:
        return
    if not (b / assets.FRIBIDI_DLL).is_file():
        return
    if str(b) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
    try:
        _DLL_DIRS.append(os.add_dll_directory(str(b)))
    except (OSError, AttributeError):
        pass
    if "PIL._imagingft" in sys.modules:
        log.warning("thumbnails: Pillow's font code was loaded before the Arabic letter fix - Arabic words may look "
                    "broken until BroClips is started again")


_dll_setup()
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, features  # noqa: E402  (after the DLL path)

W, H = 1280, 720                                # YouTube's recommended thumbnail size
TA = W / H
RAQM = bool(features.check("raqm"))             # False = Arabic letters come out unjoined and backwards
ACTION_MIN = 1500   # below this action "strength" a picture is calm: no circle / arrow, less zoom (a WEAK filter:
#                     LoL measured fights 1603-2979, a shopping moment 1454-2313 when the second moved by 0.02 s)
CUT_FADE = 0.16                                 # the sides where the picture cut the subject off fade over 16 %
CUT_MAX = 1600                                  # longest side of a stored cut-out (drawn at most ~1580 px tall)
NO_MODEL = ("Cutting out needs the cut-out AI, which is not installed yet. Download it in ⚙ Settings → “Fonts, "
            "emoji and the cut-out model” (or choose a model file you already have there).")
MAKING = "This video is being made right now — try again when it is done."

FONTS = {   # key: (file in <data>/assets/fonts, name shown, weight of a variable font, size factor, script)
    # Size factors make every font look as big at the same size. Arabic: from the ink height of "لازقة" at 120 px
    # (LoL Clips 2026-10-06, re-measured 2026-10-07: Lalezar 110, Alexandria 143, Cairo 125, Changa 125, Marhey 147,
    # Jomhuria 86). Latin: cap height of "HELLO" at 120 px (Montserrat ExtraBold 86, Anton 105, Bebas Neue 86).
    # The Latin fonts have no Arabic letters (they draw empty boxes), so Arabic words always get an Arabic font.
    "montserrat": ("Montserrat-ExtraBold.ttf", "Montserrat", None, 1.0, "latin"),
    "anton": ("Anton-Regular.ttf", "Anton", None, 0.82, "latin"),
    "bebas": ("BebasNeue-Regular.ttf", "Bebas Neue", None, 1.0, "latin"),
    "lalezar": ("Lalezar-Regular.ttf", "Lalezar", None, 1.0, "arabic"),
    "alexandria": ("Alexandria[wght].ttf", "Alexandria", 900, 0.78, "arabic"),
    "cairo": ("Cairo[slnt,wght].ttf", "Cairo", 1000, 0.88, "arabic"),
    "changa": ("Changa[wght].ttf", "Changa", 800, 0.88, "arabic"),
    "marhey": ("Marhey[wght].ttf", "Marhey", 700, 0.76, "arabic"),
    "jomhuria": ("Jomhuria-Regular.ttf", "Jomhuria", None, 1.28, "arabic"),
}
FONT_HELP = {"montserrat": "wide and very bold", "anton": "tall and strong", "bebas": "tall, capital letters only",
             "lalezar": "classic bold", "alexandria": "modern heavy", "cairo": "clean black", "changa": "condensed bold",
             "marhey": "playful", "jomhuria": "tall"}
LATIN_PICKS = ["montserrat"] * 3 + ["anton"] * 2 + ["bebas"] * 2           # 🎲 random mix weights
ARABIC_PICKS = ["lalezar"] * 3 + ["alexandria"] * 2 + ["cairo", "changa"] * 2 + ["marhey", "jomhuria"]
_SYSTEM_BOLD = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
STYLES = {  # text fill (top -> bottom), outline colour, outline width (x font size), 2nd outline, box behind
    "yellow": {"name": "Yellow", "fill": ("#fff95b", "#ffb000"), "line": "#000000", "lw": 0.10},
    "white": {"name": "White", "fill": ("#ffffff", "#dedede"), "line": "#000000", "lw": 0.10},
    "fire": {"name": "Fire", "fill": ("#ffef5a", "#ff3c00"), "line": "#1c0000", "lw": 0.09, "out": ("#ffffff", 0.035)},
    "ice": {"name": "Ice", "fill": ("#ffffff", "#5ad8ff"), "line": "#00142a", "lw": 0.10},
    "green": {"name": "Green", "fill": ("#d8ff3c", "#17c417"), "line": "#001a00", "lw": 0.10},
    "pink": {"name": "Pink", "fill": ("#ffd1f0", "#ff3fb4"), "line": "#24001a", "lw": 0.10},
    "redbox": {"name": "Red box", "fill": ("#ffffff", "#ffffff"), "line": "#000000", "lw": 0.035, "box": "#e10600"},
    "yellowbox": {"name": "Yellow box", "fill": ("#111111", "#111111"), "line": None, "lw": 0.0, "box": "#ffd400"},
}
BOX_STYLES = {k for k, v in STYLES.items() if v.get("box")}
OUTLINED = [k for k in STYLES if k not in BOX_STYLES]
GLOWS = ["#38c8ff", "#ffcc00", "#ff3355", "#b45cff", "#3dff8a", "#ffffff"]
EMOJI = {   # emoji -> Fluent Emoji 3D file (<data>/assets/emoji/<file>.png); the variation selector FE0F is ignored
    "😂": "face_with_tears_of_joy", "🤣": "rolling_on_the_floor_laughing", "💀": "skull", "😱": "face_screaming_in_fear",
    "😡": "pouting_face", "😠": "pouting_face", "🤯": "exploding_head", "😭": "loudly_crying_face",
    "😈": "smiling_face_with_horns", "🤡": "clown_face", "👀": "eyes", "🔥": "fire", "💯": "hundred_points",
    "👑": "crown", "😤": "face_with_steam_from_nose", "🥶": "cold_face", "🫠": "melting_face", "😳": "flushed_face",
    "🤬": "face_with_symbols_on_mouth", "😎": "smiling_face_with_sunglasses", "🙄": "face_with_rolling_eyes",
    "🤔": "thinking_face", "⚠": "warning", "❌": "cross_mark", "✅": "check_mark_button", "💥": "collision",
    "⚡": "high_voltage", "🏆": "trophy", "🗿": "moai", "😏": "smirking_face", "😬": "grimacing_face",
    "🫡": "saluting_face", "🥹": "face_holding_back_tears", "🥺": "pleading_face", "🤪": "zany_face",
    "🤭": "face_with_hand_over_mouth", "🤢": "nauseated_face", "👻": "ghost", "💣": "bomb", "⚔": "crossed_swords",
    "🎯": "bullseye", "✨": "sparkles", "❓": "red_question_mark", "❗": "red_exclamation_mark", "💰": "money_bag",
    "🐌": "snail", "🐐": "goat", "🚀": "rocket",
}
# The emoji the studio offers, in this order (faces first: without a face on screen they are the "reaction")
EMOJI_ORDER = "😂 🤣 💀 😱 😭 🤯 😡 🤬 😤 😳 🥶 🫠 🥹 🥺 🤪 🤭 😏 🙄 🤔 😬 😎 😈 🤡 🤢 🫡 👻 🗿 👀 🔥 💥 ⚡ 💯 👑 🏆 " \
              "⚔ 🎯 💣 🚀 ✨ ❓ ❗ ⚠ ❌ ✅ 💰 🐐 🐌".split()
FACES = set("😂 🤣 💀 😱 😭 🤯 😡 🤬 😤 😳 🥶 🫠 🥹 🥺 🤪 🤭 😏 🙄 🤔 😬 😎 😈 🤡 🤢 🫡 👻 🗿".split())
PRESETS = [("subject", "🧍 Subject"), ("zoom", "🎯 Zoom"), ("reaction", "😂 Reaction"), ("label", "📰 Label"),
           ("headline", "📢 Headline"), ("split", "⚔️ VS")]
PRESET_KEYS = [k for k, _ in PRESETS]
STYLE_HELP = {"subject": "the subject (a person or thing cut out of its background) big on one side + the words",
              "zoom": "the picture zoomed in on the action, a circle on it and an arrow, the words across",
              "reaction": "a giant emoji face + the picture + the words",
              "label": "the words in a red or yellow box + an arrow to the action",
              "headline": "huge words over a darkened, blurred picture (good for tutorials and lessons)",
              "split": "two subjects facing each other with a big 'VS' in the middle"}
AIM = {"subject": (0.30, 0.45), "zoom": (0.5, 0.42), "reaction": (0.34, 0.42), "label": (0.62, 0.55),
       "headline": (0.5, 0.5), "split": (0.5, 0.5)}                  # where each style puts the action
ZOOM = {"subject": 1.5, "zoom": 1.8, "reaction": 1.65, "label": 1.6, "headline": 1.0, "split": 1.3}
SID = re.compile(r"(?:fr|up)-[a-z0-9-]{4,90}")                       # a subject (cut-out) id: frame or upload
LANG_NAMES = {"ar": "Arabic", "en": "English", "fr": "French", "es": "Spanish", "de": "German", "tr": "Turkish",
              "pt": "Portuguese", "it": "Italian", "ru": "Russian", "hi": "Hindi", "ur": "Urdu", "id": "Indonesian",
              "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "fa": "Persian", "nl": "Dutch", "pl": "Polish"}

_LOCK = threading.RLock()                       # one drawing at a time (the server answers in threads)
_CUT_LOCK = threading.Lock()                    # one cut-out at a time (the model needs ~3-4 GB of memory)
_TILE_SEM = threading.BoundedSemaphore(2)       # at most 2 small-picture extractions at once (the studio's tiles)
_LAYERS = {}                                    # small cache of finished layers (key -> image)
_FONTS = {}
_PROBE = {}                                     # (src, size, mtime) -> (width, height, duration)
_LINES = {}                                     # video key -> [(start, end, text)]
_CUT_ERR = {}                                   # subject id -> why its cut-out failed (the studio shows it)
_AI_LAST = {}                                   # (project, video) -> {"job", "made", "msg"} of the last ✨ AI job


class ThumbError(Exception):
    """A plain-English problem the studio shows as is."""


# ---------------------------------------------------------------- small helpers
def _rgb(c):
    c = str(c or "#ffffff").lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", c):
        c = "ffffff"
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _hex(v):
    return v if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v) else None


def _solid(size, colour, alpha):
    """A layer of one colour whose see-through-ness is the mask `alpha` (L image)."""
    im = Image.new("RGBA", size, _rgb(colour) + (0,))
    im.putalpha(alpha)
    return im


def _scale_alpha(a, f):
    return a.point(lambda v: int(v * f))


def _grow(alpha, px):
    """The shape made about `px` pixels fatter, with a soft edge (outline and glow masks)."""
    if px <= 0:
        return alpha
    r = max(1.0, px / 1.6)
    return alpha.filter(ImageFilter.GaussianBlur(r)).point(lambda v: 255 if v >= 14 else v * 18)


def _put(base, layer, cx, cy):
    """Stack `layer` on `base` with its CENTRE at (cx, cy) pixels; what falls outside is cut off -> its box."""
    lw, lh = layer.size
    x0, y0 = int(round(cx - lw / 2)), int(round(cy - lh / 2))
    sx, sy = max(0, -x0), max(0, -y0)
    dx, dy = max(0, x0), max(0, y0)
    w, h = min(lw - sx, base.width - dx), min(lh - sy, base.height - dy)
    if w > 0 and h > 0:
        base.alpha_composite(layer, (dx, dy), (sx, sy, sx + w, sy + h))
    return [x0, y0, x0 + lw, y0 + lh]


def _cached(key, make):
    v = _LAYERS.get(key)
    if v is None:
        v = make()
        _LAYERS[key] = v
        while len(_LAYERS) > 24:                         # a picture layer is ~3.7 MB: keep the app small
            _LAYERS.pop(next(iter(_LAYERS)))
    return v


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else min(hi, max(lo, v))


def _dict(v):
    return v if isinstance(v, dict) else {}


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def is_arabic(text):
    return bool(_ARABIC.search(str(text or "")))


def default_font(text):
    """The font for a text when none was chosen: Lalezar for Arabic letters, else Montserrat ExtraBold."""
    return "lalezar" if is_arabic(text) else "montserrat"


def split_emoji(text):
    """'Never do this! ❌' -> ('Never do this!', ['❌']). The text fonts have no emoji, so they become 3D stickers."""
    text = re.sub(r"\{[^}]*\}", "", str(text or ""))   # transcript markers like {LOUD}
    keep, emo = [], []
    for ch in text:
        cp = ord(ch)
        if cp in (0xFE0F, 0x200D, 0x20E3) or 0x1F3FB <= cp <= 0x1F3FF:
            continue
        if cp >= 0x1F000 or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF or 0x2300 <= cp <= 0x23FF:
            emo.append(ch)
            continue
        keep.append(ch)
    s = re.sub(r"[ \t]+", " ", "".join(keep))
    s = "\n".join(ln.strip(" |-–—") for ln in s.split("\n")).strip()
    return s, emo


def emoji_file(ch):
    stem = EMOJI.get(str(ch or "").replace("\ufe0f", ""))
    f = assets.emoji_dir() / f"{stem}.png" if stem else None
    return f if f and f.exists() else None


def jpeg(img, quality=90):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=quality, subsampling=0 if quality >= 90 else 2, optimize=quality >= 90)
    return b.getvalue()


# ---------------------------------------------------------------- one video
class Ctx:
    """What the renderer needs about one video: its file (read only), size, length, project, blur boxes, folder."""

    def __init__(self, pid, vid, d, src, name, dur, w, h, boxes, project):
        self.pid, self.vid, self.dir, self.src, self.name = pid, vid, d, src, name
        self.dur, self.w, self.h, self.boxes, self.project = dur, w, h, boxes, project
        st = src.stat()
        sig = json.dumps([str(src), st.st_size, int(st.st_mtime), boxes])
        self.key = f"{vid}-{hashlib.sha1(sig.encode('utf-8')).hexdigest()[:8]}"   # frames change with these
        self.fa = (w / h) if w and h else TA                                       # the frame's shape
        self.video_type = str(project.get("video_type") or "other")
        self.avoid = [(x / 1920, y / 1080, (x + bw) / 1920, (y + bh) / 1080) for x, y, bw, bh in boxes]

    def cache(self):
        d = config.cache_dir() / "thumbs" / self.key
        d.mkdir(parents=True, exist_ok=True)
        return d


def context(pid, vid):
    """The Ctx of a project's video, or ThumbError with a plain reason."""
    if not (projects.valid_id(pid) and projects.valid_id(vid)):
        raise ThumbError("Unknown video.")
    d = projects.video_dir(pid, vid)
    v = load_json(d / "video.json")
    if not isinstance(v, dict):
        raise ThumbError("This video is not in the project (any more).")
    src = Path(str(v.get("src") or ""))
    if not v.get("src") or not src.is_file():
        raise ThumbError(f"The video file is not there any more: {v.get('src') or '?'}")
    st = src.stat()
    k = (str(src), st.st_size, int(st.st_mtime))
    if k not in _PROBE:
        pr = _dict(v.get("probe"))
        w, h, dur = pr.get("width"), pr.get("height"), v.get("dur") or pr.get("dur")
        if not (w and h and dur):
            info = media.probe(src)
            w, h, dur = info.get("width"), info.get("height"), info.get("dur")
        if not (w and h):
            raise ThumbError("This file has no picture, so it cannot make a thumbnail.")
        _PROBE[k] = (int(w), int(h), float(dur or 0))
    w, h, dur = _PROBE[k]
    boxes = []
    for b in config.settings().get("blur_boxes") or []:
        try:
            x, y, bw, bh = (float(n) for n in list(b)[:4])
            if bw > 0 and bh > 0:
                boxes.append([x, y, bw, bh])
        except (TypeError, ValueError):
            continue
    return Ctx(pid, vid, d, src, str(v.get("name") or src.stem), max(0.5, dur), w, h, boxes,
               projects.get(pid) or {})


# ---------------------------------------------------------------- the video's picture
def frame(ctx, t, small=False):
    """The video's picture at second t as a JPG with the always-blur boxes blurred -> Path (cached per video).
    small=True: a 320-wide tile for the studio's picture strip. The source file is only read."""
    t = round(min(max(float(t), 0.0), max(0.0, ctx.dur - 0.1)), 1)
    f = ctx.cache() / f"{'s' if small else 'f'}{int(round(t * 10)):06d}.jpg"
    if f.exists() and f.stat().st_size > 300:
        return f
    vf = [media.blur_filter(ctx.boxes, ctx.w, ctx.h)] if ctx.boxes else []
    if small:
        vf.append("scale=320:-2")
    elif max(ctx.w, ctx.h) > 1920:                      # 4K: 1920 is plenty for a 1280x720 thumbnail
        vf.append("scale=1920:-2" if ctx.w >= ctx.h else "scale=-2:1920")
    vf = [x for x in vf if x]
    tmp = part_path(f)
    for tt in (t, max(0.0, t - 0.5), max(0.0, t - 2.0)):    # the very end of a file may have no picture
        tmp.unlink(missing_ok=True)
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{tt:.2f}", "-i", str(ctx.src), "-frames:v", "1"]
        cmd += (["-vf", ",".join(vf)] if vf else []) + ["-q:v", "4" if small else "2", str(tmp)]
        try:
            run(cmd, timeout=180)
        except FileNotFoundError:
            raise ThumbError(media.FFMPEG_MISSING) from None
        except RuntimeError as ex:
            log.warning("thumbnail: no picture at %.1f s of %s: %s", tt, ctx.src.name, str(ex)[-300:])
            continue
        if tmp.exists() and tmp.stat().st_size > 300:
            finish(tmp, f)
            return f
    raise ThumbError(f"Could not read the picture at {t:.1f} s of the video.")


def _tile(ctx, t):
    with _TILE_SEM:
        return frame(ctx, t, small=True)


def _flat(ctx, t):
    """True for a picture with almost nothing in it (black, white, a fade): never a thumbnail."""
    try:
        a = np.asarray(Image.open(frame(ctx, t, small=True)).convert("L"), np.float32)
    except (ThumbError, OSError):
        return True
    return float(a.std()) < 10


CW, CH = 384, 216                                       # action analysis size (as LoL's action camera)


def action_at(ctx, t):
    """Where the action is in the picture at second t -> (x, y, strength), x/y as fractions of the frame. No AI:
    the action is where the picture changes most in the 0.8 s around t (a camera move changes everything the same
    way and is subtracted). strength < ACTION_MIN = a calm picture."""
    t = round(float(t), 1)                              # the same key always measures the same 0.8 s
    f = ctx.cache() / "action.json"
    known = load_json(f, {}) or {}
    k = f"{t:.1f}"
    if len(known.get(k) or []) == 3:
        return tuple(known[k])
    try:
        raw = run_bytes(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t - 0.4):.2f}", "-t", "0.8", "-i", str(ctx.src),
                         "-an", "-vf", f"scale={CW}:{CH},format=gray,tblend=all_mode=difference,fps=10",
                         "-f", "rawvideo", "pipe:1"], timeout=180)
    except (OSError, ValueError) as ex:
        log.info("thumbnail: action at %.1f s not measured (%s)", t, ex)
        raw = b""
    n = len(raw) // (CW * CH)
    pt = (0.5, 0.45, 0)
    if n:
        fr = np.frombuffer(raw[: n * CW * CH], np.uint8).reshape(n, CH, CW)
        mv = (fr > 18).sum(axis=0).astype(np.float32)
        for x0, y0, x1, y1 in ctx.avoid:                # a blurred box is no action
            mv[int(y0 * CH):int(math.ceil(y1 * CH)), int(x0 * CW):int(math.ceil(x1 * CW))] = 0
        mv = np.maximum(0, mv - np.median(mv))
        if mv.sum() > 150:
            k_ = 30                                     # the busiest square of ~1/7 of the picture's height
            c = np.pad(mv, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            box = c[k_:, k_:] - c[:-k_, k_:] - c[k_:, :-k_] + c[:-k_, :-k_]
            y, x = np.unravel_index(int(box.argmax()), box.shape)
            pt = (float((x + k_ / 2) / CW), float((y + k_ / 2) / CH), int(box.max()))
    known[k] = [round(pt[0], 3), round(pt[1], 3), pt[2]]
    try:
        save_json(known, f)
    except OSError:
        pass
    return pt


# ---------------------------------------------------------------- subjects (cut-outs)
def cut_dir():
    d = config.cache_dir() / "cutouts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cut_file(sid):
    if not SID.fullmatch(str(sid or "")):
        raise ThumbError("Unknown picture.")
    return cut_dir() / f"{sid}.png"


def _upload_src(sid):
    return cut_dir() / "src" / f"{sid}.png"


def cut_meta(sid):
    try:
        return load_json(cut_file(sid).with_suffix(".json"), {}) or {}
    except ThumbError:
        return {}


def ready(sid):
    try:
        return bool(sid) and cut_file(sid).exists()
    except ThumbError:
        return False


def frame_sid(ctx, t):
    """The subject id of a cut-out from the video's picture at second t (rounded to 0.1 s like the frame)."""
    return f"fr-{ctx.key}-{int(round(round(float(t), 1) * 10))}"


def cutout_model():
    """The BiRefNet model file (Settings' path, else the downloaded one) or None."""
    try:
        return assets.cutout_model_path()
    except Exception:
        return None


def _birefnet(im):
    """The cut-out mask of an RGB picture (uint8, its size): BiRefNet (MIT) on the CPU via onnxruntime, rembg's
    recipe (1024x1024, ImageNet mean/std, sigmoid, min-max). ~30-60 s and ~3-4 GB of memory with the full model."""
    model = cutout_model()
    if model is None:
        raise ThumbError(NO_MODEL)
    try:
        import onnxruntime as ort
    except ImportError:
        raise ThumbError("Cutting out needs the 'onnxruntime' package. Run install.bat again.") from None
    so = ort.SessionOptions()
    busy = programs_running(config.settings().get("pause_while_running") or [])
    so.intra_op_num_threads = 4 if busy else max(2, (os.cpu_count() or 4) - 2)   # fewer cores while a game runs
    t0 = time.time()
    try:
        sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    except Exception as ex:
        raise ThumbError(f"The cut-out model could not be opened ({str(ex)[:160]}). Download it again in "
                         "⚙ Settings.") from None
    try:
        x = np.asarray(im.resize((1024, 1024), Image.LANCZOS), np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        y = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    finally:
        del sess
    y = 1 / (1 + np.exp(-y))
    y = (y - y.min()) / max(1e-6, float(y.max() - y.min()))
    mask = Image.fromarray((y * 255).astype(np.uint8)).resize(im.size, Image.LANCZOS)
    a = np.asarray(mask, np.float32)
    a[a < 20] = 0
    a[a > 235] = 255
    log.info("thumbnail: cut-out mask in %.0f s (%s)", time.time() - t0, Path(model).name)
    return a.astype(np.uint8)


def cut_sides(a):
    """The sides where the picture itself cut the subject off (legs at the bottom, a shoulder at a side): drawing
    fades them out instead of showing a straight cut edge."""
    edges = {"bottom": a[-1], "top": a[0], "left": a[:, 0], "right": a[:, -1]}
    return [k for k, e in edges.items() if (e > 128).mean() >= 0.12]


def _save_cut(sid, rgb, a, how):
    """Keep a cut-out: RGBA cropped to the subject + its sides and size (.json). -> Path"""
    sides = cut_sides(a)
    cover = float((a > 128).mean())
    mask = Image.fromarray(a)
    bb = mask.point(lambda v: 255 if v > 60 else 0).getbbox()
    if not bb or cover < 0.003:
        raise ThumbError("Nothing to cut out was found in this picture. Try another picture.")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(mask)
    rgba = rgba.crop(bb)
    if max(rgba.size) > CUT_MAX:
        k = CUT_MAX / max(rgba.size)
        rgba = rgba.resize((max(1, int(rgba.width * k)), max(1, int(rgba.height * k))), Image.LANCZOS)
    out = cut_file(sid)
    tmp = out.with_name(out.stem + ".part.png")
    rgba.save(tmp)
    save_json({"sides": sides, "size": list(rgba.size), "cover": round(cover, 4), "how": how,
               "when": time.strftime("%Y-%m-%d %H:%M")}, out.with_suffix(".json"))
    finish(tmp, out)
    return out


def cut_from_frame(ctx, t):
    """The person / thing in the video's picture at second t, cut out (BiRefNet). -> (subject id, cover 0..1)"""
    sid = frame_sid(ctx, t)
    if ready(sid):
        return sid, float(cut_meta(sid).get("cover") or 0.2)
    with _CUT_LOCK:
        if not ready(sid):
            im = Image.open(frame(ctx, t)).convert("RGB")
            _save_cut(sid, im, _birefnet(im), "frame")
    return sid, float(cut_meta(sid).get("cover") or 0.2)


def _has_transparency(im):
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        a = np.asarray(im.convert("RGBA").getchannel("A"))
        return bool((a < 250).mean() >= 0.01)
    return False


def add_upload(data, name=""):
    """An image the user uploaded (a photo, a logo). A PNG with transparency is used as is (ready now); any other
    picture is kept and has to be cut out (a `cutout` job). -> (subject id, ready)"""
    if not data or len(data) < 50:
        raise ThumbError("That file is empty.")
    sid = "up-" + hashlib.sha1(data).hexdigest()[:16]
    if ready(sid):
        return sid, True
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        raise ThumbError("That file is not a picture BroClips can read (use PNG, JPG or WebP).") from None
    if im.width < 16 or im.height < 16:
        raise ThumbError("That picture is too small.")
    im.thumbnail((2048, 2048), Image.LANCZOS)
    if _has_transparency(im):
        rgba = im.convert("RGBA")
        a = np.asarray(rgba.getchannel("A")).copy()
        _save_cut(sid, rgba.convert("RGB"), a, "upload")
        log.info("thumbnail: uploaded picture %s (%s) used as is (it has transparency)", sid, name[:60])
        return sid, True
    src = _upload_src(sid)
    src.parent.mkdir(parents=True, exist_ok=True)
    tmp = src.with_name(src.stem + ".part.png")
    im.convert("RGB").save(tmp)
    finish(tmp, src)
    return sid, False


def cut_from_upload(sid):
    if ready(sid):
        return sid
    src = _upload_src(sid)
    if not src.is_file():
        raise ThumbError("That uploaded picture is gone. Upload it again.")
    with _CUT_LOCK:
        if not ready(sid):
            im = Image.open(src).convert("RGB")
            _save_cut(sid, im, _birefnet(im), "upload")
    return sid


def _line_jobs():
    """The running job and the waiting ones (copies, with their args)."""
    with jobs.LINE.lock:
        cur = [dict(jobs.LINE.current, running=True)] if jobs.LINE.current else []
        return cur + [dict(j) for j in jobs.LINE.queue], jobs.LINE.waiting_for


def cut_state(sid):
    """'ready' | 'working' | 'queued' | 'waiting' (for a program to close) | 'no-model' | 'missing' | 'error'
    -> (state, extra) where extra = the error text, or the program waited for, or the job id."""
    if not SID.fullmatch(str(sid or "")):
        return "error", "Unknown picture."
    if ready(sid):
        return "ready", ""
    line, waiting_for = _line_jobs()
    for k, j in enumerate(line):
        if j.get("kind") == "cutout" and _dict(j.get("args")).get("id") == sid:
            if j.get("running"):
                return "working", j["id"]
            return ("waiting", waiting_for) if waiting_for and not j.get("force") else ("queued", j["id"])
    if sid in _CUT_ERR:
        return "error", _CUT_ERR[sid]
    if cutout_model() is None:
        return "no-model", NO_MODEL
    return "missing", ""


def _cut_job(job, progress, cancelled):
    """The `cutout` job: args {"id": subject id, "t": seconds (a picture of the video)}."""
    a = _dict(job.get("args"))
    sid = str(a.get("id") or "")
    if not SID.fullmatch(sid):
        raise jobs.JobError("Unknown picture.")
    if ready(sid):
        return "✅ Already cut out"
    progress(1, "✂️ Cutting out the subject", "about 30-60 seconds, only the first time", 0.05)
    try:
        if sid.startswith("fr-"):
            ctx = context(job.get("project"), job.get("video"))
            got, cover = cut_from_frame(ctx, float(a.get("t") or 0))
            if got != sid:
                raise ThumbError("The video changed — cut it out again from the studio.")
        else:
            cut_from_upload(sid)
    except ThumbError as ex:
        _CUT_ERR[sid] = str(ex)
        raise jobs.JobError(str(ex)) from None
    except Exception as ex:
        _CUT_ERR[sid] = f"The cut-out failed ({type(ex).__name__}: {str(ex)[:160]})."
        raise
    _CUT_ERR.pop(sid, None)
    return "✅ Cut out — it is ready in the Thumbnail studio"


def start_cut(pid, vid, sid, t=None, title=""):
    """Put a cut-out into the work line (one heavy thing at a time). -> the job"""
    _CUT_ERR.pop(sid, None)
    args = {"id": sid}
    if t is not None:
        args["t"] = round(float(t), 1)
    return jobs.enqueue("cutout", project=pid, video=vid, args=args, title=title or "Cut-out")


def restart_cut(pid, vid, sid, title=""):
    """A remembered subject that is not cut out (the model was missing, it failed, the cache was emptied): start its
    cut-out again when that is possible. -> its state"""
    state, _ = cut_state(sid)
    if state not in ("missing", "error") or cutout_model() is None:
        return state
    if sid.startswith("up-"):
        if not _upload_src(sid).is_file():
            return state
        start_cut(pid, vid, sid, title=title)
    else:
        items = (load_json(_subjects_file(pid, vid), {}) or {}).get("items") or []
        t = next((i.get("t") for i in items if isinstance(i, dict) and i.get("id") == sid), None)
        try:
            if t is None or frame_sid(context(pid, vid), t) != sid:
                return state
        except ThumbError:
            return state
        start_cut(pid, vid, sid, t, title=title)
    return cut_state(sid)[0]


# remembered subjects of a video: <video>/thumb_subjects.json {"items": [{id, how, t?, name?}], "main", "second"}
def _subjects_file(pid, vid):
    return projects.video_dir(pid, vid) / "thumb_subjects.json"


def remember_subject(pid, vid, sid, how="", slot=None, **extra):
    if not SID.fullmatch(str(sid or "")):
        return
    f = _subjects_file(pid, vid)
    s = load_json(f, {}) or {}
    items = [i for i in (s.get("items") or []) if isinstance(i, dict) and i.get("id") != sid]
    old = next((i for i in (s.get("items") or []) if isinstance(i, dict) and i.get("id") == sid), {})
    items.insert(0, dict(old, id=sid, how=how or old.get("how") or "", **extra))
    s["items"] = items[:24]
    if slot == 1:
        s["main"] = sid
    elif slot == 2:
        s["second"] = sid
    save_json(s, f)


def subjects(pid, vid):
    """The subjects the studio offers: this video's, then pictures uploaded for other videos of the project."""
    s = load_json(_subjects_file(pid, vid), {}) or {}
    out, seen = [], set()
    for i in s.get("items") or []:
        if isinstance(i, dict) and SID.fullmatch(str(i.get("id") or "")) and i["id"] not in seen:
            seen.add(i["id"])
            out.append({"id": i["id"], "how": i.get("how", ""), "name": i.get("name", ""), "here": True})
    vids = projects.project_dir(pid) / "videos"
    for f in sorted(vids.glob("*/thumb_subjects.json")) if vids.is_dir() else []:
        if f.parent.name == vid:
            continue
        for i in (load_json(f, {}) or {}).get("items") or []:
            sid = str(_dict(i).get("id") or "")
            if sid.startswith("up-") and SID.fullmatch(sid) and sid not in seen and ready(sid):
                seen.add(sid)
                out.append({"id": sid, "how": "upload", "name": i.get("name", ""), "here": False})
    for o in out:
        o["state"] = cut_state(o["id"])[0]
    return out[:30], s.get("main"), s.get("second")


def main_subjects(pid, vid):
    """(subject 1, subject 2) ids remembered for this video whose cut-outs are ready (None when not)."""
    s = load_json(_subjects_file(pid, vid), {}) or {}
    a, b = s.get("main"), s.get("second")
    a = a if ready(a) else None
    b = b if ready(b) and b != a else None
    return a, b


# ---------------------------------------------------------------- the layers
def _font_path(key):
    f = assets.fonts_dir() / FONTS[key][0]
    return f if f.is_file() else None


def _font(key, px):
    """FONTS[key] at px pixels; a font that is not downloaded falls back to its script's default, then a system
    bold font, then Pillow's own."""
    if key not in FONTS:
        key = "montserrat"
    path, wght = _font_path(key), FONTS[key][2]
    if path is None:
        alt = "lalezar" if FONTS[key][4] == "arabic" else "montserrat"
        path, wght = _font_path(alt), FONTS[alt][2]
    if path is None:
        path, wght = next((Path(p) for p in _SYSTEM_BOLD if Path(p).is_file()), None), None
    k = (str(path), px, wght)
    f = _FONTS.get(k)
    if f is None:
        if path is None:
            f = ImageFont.load_default(px)
        else:
            f = ImageFont.truetype(str(path), px,
                                   layout_engine=ImageFont.Layout.RAQM if RAQM else ImageFont.Layout.BASIC)
            if wght:
                try:
                    axes = f.get_variation_axes()
                    f.set_variation_by_axes([wght if a.get("name") in (b"Weight", "Weight") else a.get("default", 0)
                                             for a in axes])
                except Exception as ex:
                    log.info("thumbnail: font weight not set (%s)", ex)
        _FONTS[k] = f
        while len(_FONTS) > 40:
            _FONTS.pop(next(iter(_FONTS)))
    return f


def _line_masks(line, font, size, lw, ow):
    """Masks of one line of text: its letters, letters + outline, letters + both outlines (cut to the ink)."""
    pad = lw + ow + size
    w, h = int(font.getlength(line)) + 2 * pad, int(size * 2.4) + 2 * (lw + ow)
    c = (w / 2, h / 2)

    def one(sw):
        m = Image.new("L", (w, h))
        ImageDraw.Draw(m).text(c, line, font=font, fill=255, anchor="mm", stroke_width=sw, stroke_fill=255)
        return m
    fill = one(0)
    stroke = one(lw) if lw else fill
    outer = one(lw + ow) if ow else None
    bb = (outer or stroke).getbbox() or (0, 0, w, h)
    return [m.crop(bb) if m is not None else None for m in (fill, stroke, outer)]


def _size_of(t):
    return max(30, int(t["size"] * FONTS.get(t["font"], FONTS["montserrat"])[3]))


def _wrap(t):
    """The lines of a text: the user's own line breaks; else a long phrase that would have to shrink a lot on one
    line is split into two lines of about equal width (bigger letters read better on a phone)."""
    lines = [ln.strip() for ln in t["s"].splitlines() if ln.strip()][:3]
    if len(lines) != 1 or len(lines[0].split()) < 2:
        return lines
    st = STYLES[t["style"]]
    size = _size_of(t)
    f = _font(t["font"], size)
    edge = size * 2 * (st["lw"] + (st.get("out") or (0, 0))[1])
    if f.getlength(lines[0]) + edge <= t["w"] * W * 1.12:
        return lines
    words = lines[0].split()
    k = min(range(1, len(words)),
            key=lambda i: max(f.getlength(" ".join(words[:i])), f.getlength(" ".join(words[i:]))))
    return [" ".join(words[:k]), " ".join(words[k:])]


def _text_layer(t):
    """t = {"s", "font", "size", "style", "rot", "w"} -> the text as a picture: gradient letters, thick outline,
    soft shadow (or a coloured box), tilted by rot degrees."""
    st = STYLES[t["style"]]
    lines = _wrap(t)
    if not lines:
        return None
    size = _size_of(t)
    maxw = t["w"] * W
    for _ in range(14):                                 # shrink until the longest line fits its width
        f = _font(t["font"], size)
        widest = max(f.getlength(ln) for ln in lines) + size * 2 * (st["lw"] + (st.get("out") or (0, 0))[1])
        if widest <= maxw or size <= 40:
            break
        size = max(40, int(size * max(0.6, min(0.96, maxw / widest))))
    lw = int(round(size * st["lw"]))
    ow = int(round(size * st["out"][1])) if st.get("out") else 0
    rows = [_line_masks(ln, f, size, lw, ow) for ln in lines]
    gap = int(size * 0.04)
    bw = max(r[0].width for r in rows)
    bh = sum(r[0].height for r in rows) + gap * (len(rows) - 1)
    pad = int(size * (0.34 if st.get("box") else 0.16))
    cw, ch = bw + 2 * pad, bh + 2 * pad
    F, S, O = (Image.new("L", (cw, ch)) for _ in range(3))
    ramp = np.zeros(ch, np.float32)                     # the fill colour runs top -> bottom inside EVERY line
    yy = pad
    for fill, stroke, outer in rows:                    # (the three masks of a line have the same size)
        x0 = pad + (bw - fill.width) // 2
        if outer is not None:
            O.paste(outer, (x0, yy))
        S.paste(stroke, (x0, yy))
        F.paste(fill, (x0, yy))
        fb = fill.getbbox() or (0, 0, 1, fill.height)
        ramp[yy + fb[1]:yy + fb[3]] = np.linspace(0, 1, fb[3] - fb[1], dtype=np.float32)
        ramp[yy + fb[3]:] = 1
        yy += fill.height + gap
    out = Image.new("RGBA", (cw, ch))
    edge = O if ow else S
    if st.get("box"):
        bb = S.getbbox() or (0, 0, cw, ch)
        px, py = int(size * 0.24), int(size * 0.10)
        box = Image.new("L", (cw, ch))
        ImageDraw.Draw(box).rounded_rectangle((bb[0] - px, bb[1] - py, bb[2] + px, bb[3] + py),
                                              radius=int(size * 0.14), fill=255)
        sh = box.filter(ImageFilter.GaussianBlur(size * 0.06))
        out.alpha_composite(_solid((cw, ch), "#000000", _scale_alpha(sh, 0.6)), (int(size * 0.03), int(size * 0.06)))
        out.alpha_composite(_solid((cw, ch), st["box"], box))
    else:
        sh = edge.filter(ImageFilter.GaussianBlur(size * 0.05))
        out.alpha_composite(_solid((cw, ch), "#000000", _scale_alpha(sh, 0.75)), (int(size * 0.03), int(size * 0.06)))
    if ow:
        out.alpha_composite(_solid((cw, ch), st["out"][0], O))
    if lw and st.get("line"):
        out.alpha_composite(_solid((cw, ch), st["line"], S))
    c1, c2 = np.array(_rgb(st["fill"][0]), np.float32), np.array(_rgb(st["fill"][1]), np.float32)
    k = ramp[:, None, None]
    grad = Image.fromarray(np.repeat((c1 * (1 - k) + c2 * k).astype(np.uint8), cw, axis=1)).convert("RGBA")
    grad.putalpha(F)
    out.alpha_composite(grad)
    if t.get("rot"):
        out = out.rotate(t["rot"], resample=Image.BICUBIC, expand=True)
    return out


def fade_mask(th, tw, sides):
    """1.0 inside, falling to 0 over CUT_FADE of the size at the given sides (where the picture cut it off)."""
    fade = np.ones((th, tw), np.float32)
    for side in sides or []:
        n = max(2, int((th if side in ("top", "bottom") else tw) * CUT_FADE))
        ramp = np.linspace(0, 1, n, dtype=np.float32) ** 1.4
        if side == "bottom":
            fade[-n:] *= ramp[::-1, None]
        elif side == "top":
            fade[:n] *= ramp[:, None]
        elif side == "left":
            fade[:, :n] *= ramp[None, :]
        elif side == "right":
            fade[:, -n:] *= ramp[None, ::-1]
    return fade


def _subject_layer(c, path=None, sides=None):
    """c = {"id", "h", "flip", "line", "glow"} -> the cut-out subject with a white outline and a coloured glow
    behind it (None if it has not been cut out yet). The sides the picture cut it off fade out, and so do its
    outline and glow (LoL: a white line along a cut cape looked like a grey box)."""
    f = Path(path) if path else cut_file(c["id"])
    if not f.exists():
        return None
    im = Image.open(f).convert("RGBA")
    th = max(40, int(c["h"] * H))
    tw = max(16, int(im.width * th / im.height))
    im = im.resize((tw, th), Image.LANCZOS)
    fade = fade_mask(th, tw, cut_meta(c["id"]).get("sides", []) if sides is None else sides)
    alpha = Image.fromarray((np.asarray(im.getchannel("A"), np.float32) * fade).astype(np.uint8))
    rgb = im.convert("RGB").filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))
    rgb.putalpha(alpha)
    im = rgb
    if c.get("flip"):
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        fade = fade[:, ::-1]
    lw, gl = max(4, int(th * 0.009)), int(th * 0.045)
    pad = gl * 2 + lw
    size = (tw + 2 * pad, th + 2 * pad)
    a = Image.new("L", size)
    a.paste(im.getchannel("A"), (pad, pad))
    edge_fade = np.pad(fade, pad, mode="edge")             # outside the picture: the same as at its border

    def faded(m):
        return Image.fromarray((np.asarray(m, np.float32) * edge_fade).astype(np.uint8))
    out = Image.new("RGBA", size)
    if c.get("glow"):
        g = _grow(a, gl).filter(ImageFilter.GaussianBlur(gl * 0.8))
        out.alpha_composite(_solid(size, c["glow"], _scale_alpha(faded(g), 0.9)))
    if c.get("line"):
        out.alpha_composite(_solid(size, c["line"], faded(_grow(a, lw))))
    out.alpha_composite(im, (pad, pad))
    return out


def _emoji_layer(s):
    f = emoji_file(s.get("e"))
    if not f:
        return None
    px = max(24, int(s["s"] * H))
    em = Image.open(f).convert("RGBA").resize((px, px), Image.LANCZOS)
    rgb = em.convert("RGB").filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    rgb.putalpha(em.getchannel("A"))
    pad = int(px * 0.14)
    size = (px + 2 * pad, px + 2 * pad)
    a = Image.new("L", size)
    a.paste(em.getchannel("A"), (pad, pad))
    out = Image.new("RGBA", size)
    sh = _grow(a, px * 0.03).filter(ImageFilter.GaussianBlur(px * 0.035))
    out.alpha_composite(_solid(size, "#000000", _scale_alpha(sh, 0.55)), (int(px * 0.025), int(px * 0.045)))
    out.alpha_composite(_solid(size, "#ffffff", _grow(a, max(3, px * 0.035))))
    out.alpha_composite(rgb, (pad, pad))
    return out.rotate(s.get("rot", 0), resample=Image.BICUBIC, expand=True) if s.get("rot") else out


def _arrow_layer(s):
    """A fat arrow, `s` x canvas height long, pointing at angle rot (0 = right, counter-clockwise), white edge."""
    k = 2                                               # drawn twice as big, then shrunk: smooth edges
    L = max(40, int(s["s"] * H)) * k
    pad = int(L * 0.22)
    size = (L + 2 * pad, L + 2 * pad)
    cx, cy = size[0] / 2, size[1] / 2
    t, hw, hx = 0.12 * L, 0.31 * L, 0.10 * L
    pts = [(-0.5 * L, -t), (hx, -t), (hx, -hw), (0.5 * L, 0), (hx, hw), (hx, t), (-0.5 * L, t)]
    pts = [(cx + x, cy + y) for x, y in pts]
    shape = Image.new("L", size)
    ImageDraw.Draw(shape).polygon(pts, fill=255)
    edge = _grow(shape, 0.05 * L)
    out = Image.new("RGBA", size)
    sh = edge.filter(ImageFilter.GaussianBlur(0.04 * L))
    out.alpha_composite(_solid(size, "#000000", _scale_alpha(sh, 0.6)), (int(0.03 * L), int(0.05 * L)))
    out.alpha_composite(_solid(size, "#ffffff", edge))
    out.alpha_composite(_solid(size, s.get("color") or "#ff1e1e", shape))
    out = out.resize((size[0] // k, size[1] // k), Image.LANCZOS)
    return out.rotate(s.get("rot", 0), resample=Image.BICUBIC, expand=True)


def _circle_layer(s):
    """A thick ring (radius s x canvas height) with a dark edge and a soft glow, around the action."""
    k = 2
    r = max(20, int(s["s"] * H)) * k
    th = max(10, int(r * 0.12))
    pad = int(r * 0.35)
    size = (2 * (r + pad), 2 * (r + pad))
    c = size[0] / 2
    ring = Image.new("L", size)
    ImageDraw.Draw(ring).ellipse((c - r, c - r, c + r, c + r), outline=255, width=th)
    edge = _grow(ring, th * 0.35)
    out = Image.new("RGBA", size)
    col = s.get("color") or "#ffe600"
    out.alpha_composite(_solid(size, col, _scale_alpha(edge.filter(ImageFilter.GaussianBlur(th * 1.2)), 0.7)))
    out.alpha_composite(_solid(size, "#000000", _scale_alpha(edge, 0.85)))
    out.alpha_composite(_solid(size, col, ring))
    return out.resize((size[0] // k, size[1] // k), Image.LANCZOS)


def _rays_layer(r):
    """Soft light rays from behind the subject (a classic thumbnail trick), drawn small and blown up."""
    w, h = W // 2, H // 2
    cx, cy = r["x"] * w, r["y"] * h
    m = Image.new("L", (w, h))
    d = ImageDraw.Draw(m)
    n = 18
    for i in range(n):
        a0 = (i + 0.5 * (i % 2)) * 2 * math.pi / n + r.get("spin", 0.3)
        a1 = a0 + math.pi / n * 0.55
        d.polygon([(cx, cy), (cx + 2000 * math.cos(a0), cy + 2000 * math.sin(a0)),
                   (cx + 2000 * math.cos(a1), cy + 2000 * math.sin(a1))], fill=120 if i % 2 else 70)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fall = np.clip(1.15 - np.hypot(xx - cx, yy - cy) / (0.75 * w), 0, 1)
    a = (np.asarray(m.filter(ImageFilter.GaussianBlur(2)), np.float32) * fall).astype(np.uint8)
    glow = np.clip(1 - np.hypot(xx - cx, yy - cy) / (0.32 * w), 0, 1) ** 1.5 * 170
    a = np.maximum(a, glow.astype(np.uint8))
    lay = _solid((w, h), r.get("color") or "#ffcc00", Image.fromarray(a))
    return lay.resize((W, H), Image.BILINEAR)


_VIG = {}


def _vignette():
    if "v" not in _VIG:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / math.sqrt(2)
        _VIG["v"] = 1 - 0.42 * np.clip((rr - 0.42) / 0.58, 0, 1) ** 1.5
        _VIG["x"], _VIG["y"] = xx / W, yy / H
    return _VIG


def _window(bg, fa=TA):
    """The part of the video picture that is shown, in fractions of the frame: (x0, y0, width, height). At zoom 1
    it is the biggest 16:9 part of the frame (a tall phone video shows a band of it)."""
    z = bg["z"]
    ww0, wh0 = (TA / fa, 1.0) if fa >= TA else (1.0, fa / TA)
    ww, wh = ww0 / z, wh0 / z
    x0 = min(max(bg["cx"] - ww / 2, 0.0), 1 - ww)
    y0 = min(max(bg["cy"] - wh / 2, 0.0), 1 - wh)
    return x0, y0, ww, wh


def _bg_layer(bg, ctx):
    im = Image.open(frame(ctx, bg["t"])).convert("RGB")
    x0, y0, ww, wh = _window(bg, ctx.fa)
    sw, sh = im.size
    im = im.resize((W, H), Image.LANCZOS, box=(x0 * sw, y0 * sh, (x0 + ww) * sw, (y0 + wh) * sh))
    p = bg["punch"]
    if p:
        im = ImageEnhance.Color(im).enhance(1 + 0.22 * p)
        im = ImageEnhance.Contrast(im).enhance(1 + 0.07 * p)
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=40 + 30 * p, threshold=2))
    if bg["blur"]:
        im = im.filter(ImageFilter.GaussianBlur(bg["blur"]))
    v = _vignette()
    m = v["v"] * (1 - bg["dim"])
    side = {"left": 1 - v["x"], "right": v["x"], "top": 1 - v["y"], "bottom": v["y"]}.get(bg["shade"])
    if side is not None:
        m = m * (1 - 0.62 * np.clip((side - 0.38) / 0.62, 0, 1) ** 1.2)
    a = np.asarray(im, np.float32) * m[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")


# ---------------------------------------------------------------- designs
def _text_spec(t, size=150, style="yellow"):
    t = _dict(t)
    s, _ = split_emoji(str(t.get("s") or "")[:80])
    font = t.get("font") if t.get("font") in FONTS else default_font(s)
    if FONTS[font][4] == "latin" and is_arabic(s):      # Latin fonts have no Arabic letters
        font = default_font(s)
    return {"s": s, "font": font, "size": int(_clamp(t.get("size"), 30, 320, size)),
            "style": t.get("style") if t.get("style") in STYLES else style, "rot": _clamp(t.get("rot"), -30, 30, 0),
            "w": _clamp(t.get("w"), 0.15, 1.0, 0.8), "x": _clamp(t.get("x"), -0.2, 1.2, 0.5),
            "y": _clamp(t.get("y"), -0.2, 1.2, 0.8)}


def _subject_spec(c):
    c = _dict(c)
    if not SID.fullmatch(str(c.get("id") or "")):
        return None
    return {"id": str(c["id"]), "x": _clamp(c.get("x"), -0.3, 1.3, 0.75), "y": _clamp(c.get("y"), -0.3, 1.5, 0.55),
            "h": _clamp(c.get("h"), 0.1, 2.2, 0.9), "flip": bool(c.get("flip")),
            "line": _hex(c["line"]) if "line" in c else "#ffffff",           # None = no outline (the user's choice)
            "glow": _hex(c.get("glow")), "on": c.get("on", True) is not False}


def normal(d):
    """A clean, safe copy of a design from the studio (unknown things dropped, numbers kept in range)."""
    d = _dict(d)
    b = _dict(d.get("bg"))
    out = {"v": 1, "preset": d.get("preset") if d.get("preset") in PRESET_KEYS else "zoom",
           "bg": {"t": round(_clamp(b.get("t"), 0, 1e6, 0.0), 2), "z": _clamp(b.get("z"), 1.0, 4.0, 1.6),
                  "cx": _clamp(b.get("cx"), 0, 1, 0.5), "cy": _clamp(b.get("cy"), 0, 1, 0.45),
                  "punch": int(_clamp(b.get("punch"), 0, 2, 1)), "blur": _clamp(b.get("blur"), 0, 20, 0),
                  "dim": _clamp(b.get("dim"), 0, 0.8, 0.0),
                  "shade": b.get("shade") if b.get("shade") in ("left", "right", "top", "bottom") else None},
           "text": _text_spec(d.get("text")),
           "text2": _text_spec(d.get("text2"), 190, "fire") if _dict(d.get("text2")).get("s") else None,
           "subject": _subject_spec(d.get("subject")), "subject2": _subject_spec(d.get("subject2")),
           "rays": None, "stickers": []}
    r = d.get("rays")
    if isinstance(r, dict):
        out["rays"] = {"x": _clamp(r.get("x"), -0.2, 1.2, 0.75), "y": _clamp(r.get("y"), -0.2, 1.2, 0.45),
                       "color": _hex(r.get("color")) or "#ffcc00", "spin": _clamp(r.get("spin"), 0, 7, 0.3)}
    stickers = d.get("stickers") if isinstance(d.get("stickers"), list) else []
    for s in stickers[:12]:
        s = _dict(s)
        k = s.get("k")
        if k not in ("emoji", "arrow", "circle"):
            continue
        if k == "emoji" and not emoji_file(s.get("e")):
            continue
        out["stickers"].append({"k": k, "e": str(s.get("e")).replace("\ufe0f", "") if k == "emoji" else None,
                                "x": _clamp(s.get("x"), -0.2, 1.2, 0.5), "y": _clamp(s.get("y"), -0.2, 1.2, 0.5),
                                "s": _clamp(s.get("s"), 0.04, 1.2, 0.25), "rot": _clamp(s.get("rot"), -360, 360, 0),
                                "color": _hex(s.get("color"))})
    return out


def draw(design, ctx):
    """design -> (RGB picture 1280x720, {"text": box, "subject": box, "s0": box, ...}). The boxes (pixels) let the
    studio find what the mouse is on."""
    d = normal(design)
    with _LOCK:
        bgk = ("bg", ctx.key, json.dumps(d["bg"], sort_keys=True))
        img = _cached(bgk, lambda: _bg_layer(d["bg"], ctx)).copy()
        boxes = {}
        if d["rays"]:
            img.alpha_composite(_cached(("rays", json.dumps(d["rays"], sort_keys=True)),
                                        lambda: _rays_layer(d["rays"])))
        for k in ("subject2", "subject"):
            c = d[k]
            if c and c["on"] and ready(c["id"]):
                lay = _cached(("subj", c["id"], round(c["h"], 3), c["flip"], c["line"], c["glow"]),
                              lambda c=c: _subject_layer(c))
                if lay is not None:
                    boxes[k] = _put(img, lay, c["x"] * W, c["y"] * H)
        for order in (("circle", "arrow"), ("text",), ("emoji",)):
            if order == ("text",):
                for k in ("text2", "text"):
                    t = d[k]
                    if t and t["s"]:
                        key = (k, json.dumps({a: t[a] for a in ("s", "font", "size", "style", "rot", "w")},
                                             sort_keys=True, ensure_ascii=False), RAQM)
                        lay = _cached(key, lambda t=t: _text_layer(t))
                        if lay is not None:
                            boxes[k] = _put(img, lay, t["x"] * W, t["y"] * H)
                continue
            for i, s in enumerate(d["stickers"]):
                if s["k"] not in order:
                    continue
                make = {"emoji": _emoji_layer, "arrow": _arrow_layer, "circle": _circle_layer}[s["k"]]
                key = ("st", s["k"], s["e"], round(s["s"], 3), round(s["rot"], 1), s["color"])
                lay = _cached(key, lambda s=s, make=make: make(s))
                if lay is not None:
                    boxes[f"s{i}"] = _put(img, lay, s["x"] * W, s["y"] * H)
        return img.convert("RGB"), boxes


# ---------------------------------------------------------------- styles (where everything goes)
def _aim(bg, ax, ay, px, py, fa=TA, avoid=()):
    """Move the zoomed window so the action (ax, ay of the frame) lands near (px, py) of the thumbnail - but steer
    around the avoid boxes (Settings' blur boxes) as long as the action stays well inside the picture."""
    _, _, ww, wh = _window(dict(bg, cx=0.5, cy=0.5), fa)
    hw, hh = ww / 2, wh / 2
    wx, wy = ax + (0.5 - px) * ww, ay + (0.5 - py) * wh
    best, best_cost = None, 1e9
    for dy in np.linspace(-0.35, 0.35, 15):
        for dx in np.linspace(-0.35, 0.35, 15):
            cx = min(max(wx + dx, hw), 1 - hw)
            cy = min(max(wy + dy, hh), 1 - hh)
            x0, y0 = cx - hw, cy - hh
            u, v = (ax - x0) / ww, (ay - y0) / wh
            if not (0.12 <= u <= 0.88 and 0.12 <= v <= 0.88):
                continue
            cover = sum(max(0.0, min(cx + hw, zx1) - max(x0, zx0)) * max(0.0, min(cy + hh, zy1) - max(y0, zy0))
                        for zx0, zy0, zx1, zy1 in avoid) / (ww * wh)
            cost = cover * 40 + math.hypot(cx - wx, cy - wy)
            if cost < best_cost:
                best, best_cost = (cx, cy), cost
    if best is None:                                    # the action is at the very edge: just keep it in view
        best = (min(max(wx, hw), 1 - hw), min(max(wy, hh), 1 - hh))
    bg["cx"], bg["cy"] = round(float(best[0]), 4), round(float(best[1]), 4)


def _where(bg, ax, ay, fa=TA):
    """Where the action (ax, ay of the frame) is in the thumbnail, in fractions."""
    x0, y0, ww, wh = _window(bg, fa)
    return (ax - x0) / ww, (ay - y0) / wh


def _arrow_to(tx, ty, fx, fy, length=0.26):
    """An arrow whose head ends just before (tx, ty), coming from the direction of (fx, fy)."""
    ang = math.atan2(-(ty - fy) * H, (tx - fx) * W)
    gap = 0.20                                          # leave room for the circle around the target
    lx, ly = math.cos(ang) * length * H / W, -math.sin(ang) * length
    gx, gy = math.cos(ang) * gap * H / W, -math.sin(ang) * gap
    cx, cy = tx - gx - lx / 2, ty - gy - ly / 2
    return {"k": "arrow", "x": round(cx, 4), "y": round(cy, 4), "s": length, "rot": round(math.degrees(ang), 1),
            "color": "#ff1e1e"}


def _rect_dist(x, y, r):
    """Pixels from point (x, y) to the box r = (x0, y0, x1, y1); 0 inside."""
    return math.hypot(max(r[0] - x, 0, x - r[2]), max(r[1] - y, 0, y - r[3]))


def _text_box(t):
    """Roughly where the words will be (pixels) - before drawing them, for keeping other things off them."""
    lines = len(_wrap(t)) or 1
    hw, hh = t["w"] * W / 2, min(0.42, 0.12 * lines * t["size"] / 150) * H
    return (t["x"] * W - hw, t["y"] * H - hh, t["x"] * W + hw, t["y"] * H + hh)


def _fit_text(t):
    """Keep the words inside the picture (a long text becomes two lines and is taller than it was planned)."""
    bx = _text_box(t)
    hh = (bx[3] - bx[1]) / 2 / H
    t["y"] = min(max(t["y"], hh + 0.02), 1 - hh - 0.02)


def _spot(cands, avoid, r_px):
    """The candidate (x, y) whose circle of radius r_px stays farthest from every box in `avoid`."""
    def room(p):
        return min((_rect_dist(p[0] * W, p[1] * H, a) - r_px for a in avoid), default=1e9)
    return max(cands, key=room)


def _free_arrow(u, v, avoid, length=0.26):
    """An arrow pointing at (u, v), coming from the direction with the most free room (arrows from above win
    a tie: they read as "look down here")."""
    best, best_room = None, -1e9
    for deg in range(0, 360, 20):
        r = math.radians(deg)
        a = _arrow_to(u, v, u - math.cos(r) * 0.3, v + math.sin(r) * 0.3, length)
        x, y = a["x"] * W, a["y"] * H
        half = length * H / 2
        if min(x - half, W - x - half, y - half * 0.6, H - y - half * 0.6) < 0:
            continue
        room = min((_rect_dist(x, y, b) - half for b in avoid), default=400) + (25 if 200 <= deg <= 340 else 0)
        if room > best_room:
            best, best_room = a, room
    return best if best is not None and best_room > -0.25 * length * H else None


def place_subject(c, cx, region_w=0.46, region_h=0.9):
    """Size and place subject spec `c` (in place) around x=cx: one the picture cut off at the bottom (a person in a
    video) stands on the bottom edge, big; a whole thing (a logo, a photo without background) fits the region."""
    meta = cut_meta(c["id"])
    iw, ih = meta.get("size") or (1, 1)
    asp = max(0.05, float(iw) / max(1.0, float(ih)))
    fit_h = region_w * W / (asp * H)                    # the height at which it is region_w wide
    if "bottom" in (meta.get("sides") or []):
        h = max(0.35, min(1.12, fit_h))
        y = 1.04 - h / 2
    else:
        h = max(0.25, min(region_h, fit_h))
        y = 0.52
    c.update(x=round(cx, 4), y=round(y, 4), h=round(h, 3), on=True)
    return c


def layout(design, preset, act, fa=TA, avoid=()):
    """Put everything of `design` where style `preset` wants it; the picture, the words, the subjects and the emoji
    stay. act = (x, y[, strength]) of the action in the picture (action_at); fa = the frame's shape; avoid = boxes
    (fractions of the frame) the zoom steers around. The words, the emoji and the arrow go where they cover neither
    the action nor each other; a circle / arrow only when there IS action. Returns a new design."""
    d = normal(design)
    ax, ay = act[0], act[1]
    strong = len(act) < 3 or act[2] >= ACTION_MIN
    emo = next((s["e"] for s in d["stickers"] if s["k"] == "emoji"), None)
    has1 = bool(d["subject"]) and ready(d["subject"]["id"])
    has2 = bool(d["subject2"]) and ready(d["subject2"]["id"])
    preset = preset if preset in PRESET_KEYS else "zoom"
    if preset == "split" and not (has1 and has2):
        preset = "subject"
    if preset == "subject" and not has1:
        preset = "reaction"
    d["preset"] = preset
    bg, tx = d["bg"], d["text"]
    if tx["style"] in BOX_STYLES and preset != "label":
        tx["style"] = "yellow"
    d["text2"], d["rays"], st = None, None, []
    bg.update(blur=0, dim=0.0, punch=1, shade=None)
    for k in ("subject", "subject2"):
        if d[k]:
            d[k]["on"] = preset == "split" or (preset == "subject" and k == "subject")
    z = ZOOM[preset]
    if not strong and preset in ("zoom", "reaction", "label", "subject"):
        z = 1 + (z - 1) * 0.45                          # a calm picture: show more of it
    bg["z"] = round(z, 3)
    _aim(bg, ax, ay, *AIM[preset], fa=fa, avoid=avoid)
    u, v = _where(bg, ax, ay, fa)
    low = v > 0.56                                      # the action is low -> the words go to the top

    def ring(r):
        return (u * W - r * H, v * H - r * H, u * W + r * H, v * H + r * H)
    if preset == "subject":
        place_subject(d["subject"], 0.75)
        d["subject"]["glow"] = d["subject"].get("glow") or GLOWS[0]
        d["rays"] = {"x": 0.75, "y": 0.42, "color": d["subject"]["glow"], "spin": 0.3}
        bg["shade"] = "left"
        tx.update(x=0.31, y=0.24 if low else 0.75, w=0.6, rot=3)
        _fit_text(tx)
        if emo:
            avoid_ = [_text_box(tx), ring(0.12), (0.52 * W, 0, W, H)]
            x, y = _spot([(0.1, 0.18), (0.1, 0.82), (0.45, 0.16), (0.45, 0.84)], avoid_, 0.14 * H)
            st.append({"k": "emoji", "e": emo, "x": x, "y": y, "s": 0.28, "rot": 12})
    elif preset == "zoom":
        bg.update(punch=2, shade="top" if low else "bottom")
        tx.update(x=0.5, y=0.16 if low else 0.84, w=0.92, rot=0)
        _fit_text(tx)
        if strong:
            st.append({"k": "circle", "x": u, "y": v, "s": 0.17, "color": "#ffe600"})
        avoid_ = [_text_box(tx), ring(0.2)]
        if emo:
            x, y = _spot([(0.12, 0.24), (0.88, 0.24), (0.12, 0.76), (0.88, 0.76)], avoid_, 0.15 * H)
            st.append({"k": "emoji", "e": emo, "x": x, "y": y, "s": 0.26, "rot": 10 if x < 0.5 else -10})
            avoid_.append((x * W - 0.14 * H, y * H - 0.14 * H, x * W + 0.14 * H, y * H + 0.14 * H))
        a = _free_arrow(u, v, avoid_) if strong else None
        if a:
            st.append(a)
    elif preset == "reaction":
        bg["shade"] = "top" if low else "bottom"
        side = 0.77 if u < 0.55 else 0.23               # the big face goes opposite the action
        # the words stay on the other half: the face (0.62 x the height) covers x 0.60-0.94 (or 0.06-0.40)
        tx.update(x=0.31 if side > 0.5 else 0.69, y=0.16 if low else 0.84, w=0.56, rot=2)
        _fit_text(tx)
        # no circle here: the big face is what the eye should land on; and it is always a face (a giant ❌ is no
        # reaction - LoL's first result)
        st.append({"k": "emoji", "e": emo if emo in FACES else "😂", "x": side, "y": 0.6 if low else 0.42, "s": 0.62,
                   "rot": -8 if side > 0.5 else 8})
    elif preset == "label":
        if tx["style"] not in BOX_STYLES:
            tx["style"] = "redbox"
        tx.update(x=0.31 if u > 0.5 else 0.69, y=0.8 if v < 0.45 else 0.2, w=0.56, rot=3)
        _fit_text(tx)
        if strong:
            st.append({"k": "circle", "x": u, "y": v, "s": 0.14, "color": "#ff1e1e"})
        avoid_ = [_text_box(tx), ring(0.16)]
        if emo:
            x, y = _spot([(tx["x"] + 0.31 * (1 if tx["x"] < 0.5 else -1), tx["y"]), (0.1, 0.5), (0.9, 0.5)],
                         avoid_, 0.11 * H)
            st.append({"k": "emoji", "e": emo, "x": x, "y": y, "s": 0.2, "rot": -12})
            avoid_.append((x * W - 0.1 * H, y * H - 0.1 * H, x * W + 0.1 * H, y * H + 0.1 * H))
        a = _free_arrow(u, v, avoid_, length=0.24) if strong else None
        if a:
            st.append(a)
    elif preset == "headline":
        bg.update(blur=8, dim=0.45)
        tx.update(x=0.5, y=0.5, w=0.92, rot=0, size=max(tx["size"], 230))
        if emo:                                         # the words make room for the emoji at the side
            tx.update(x=0.44, w=0.8)
        _fit_text(tx)
        if emo:
            x, y = _spot([(0.92, 0.22), (0.92, 0.78), (0.92, 0.5)], [_text_box(tx)], 0.1 * H)
            st.append({"k": "emoji", "e": emo, "x": x, "y": y, "s": 0.2, "rot": -10})
    elif preset == "split":
        bg.update(blur=7, dim=0.35)
        a_, b_ = d["subject"], d["subject2"]
        place_subject(a_, 0.25, region_w=0.44)
        place_subject(b_, 0.75, region_w=0.44)
        a_["glow"], b_["glow"] = "#38c8ff", "#ff3355"
        a_["flip"] = False
        b_["flip"] = b_["id"].startswith("fr-")         # people face each other; an uploaded logo is never mirrored
        d["rays"] = {"x": 0.5, "y": 0.45, "color": "#ffcc00", "spin": 0.0}
        d["text2"] = _text_spec({"s": "VS", "font": "anton", "size": 210, "style": "fire", "rot": 6, "w": 0.4,
                                 "x": 0.5, "y": 0.42})
        tx.update(x=0.5, y=0.86, w=0.9, rot=0)
        _fit_text(tx)
        if emo in FACES:
            st.append({"k": "emoji", "e": emo, "x": 0.5, "y": 0.14, "s": 0.2, "rot": 0})
    d["stickers"] = st
    return normal(d)


def follow(design, act, fa=TA, avoid=()):
    """Another picture was picked: zoom onto ITS action the way the style does, circles go round the action and
    arrows point at it; the words, the subjects and the emoji stay where the user put them."""
    d = normal(design)
    ax, ay = act[0], act[1]
    _aim(d["bg"], ax, ay, *AIM.get(d["preset"], (0.5, 0.45)), fa=fa, avoid=avoid)
    u, v = _where(d["bg"], ax, ay, fa)
    for i, s in enumerate(d["stickers"]):
        if s["k"] == "circle":
            s["x"], s["y"] = u, v
        elif s["k"] == "arrow":                         # keep the direction it comes from
            r = math.radians(s["rot"])
            d["stickers"][i] = dict(_arrow_to(u, v, u - math.cos(r) * 0.3, v + math.sin(r) * 0.3, s["s"]),
                                    color=s["color"])
    return normal(d)


def new_design(t, text="", subject=None, subject2=None, emoji=None, font=None, style="yellow"):
    s, emo = split_emoji(text)
    emoji = emoji or (emo[0] if emo and emoji_file(emo[0]) else None)
    d = {"bg": {"t": t, "z": 1.6, "cx": 0.5, "cy": 0.45},
         "text": {"s": s, "font": font or default_font(s), "style": style, "size": 150},
         "subject": {"id": subject, "line": "#ffffff", "glow": GLOWS[0]} if subject else None,
         "subject2": {"id": subject2, "line": "#ffffff", "glow": "#ff3355", "on": False} if subject2 else None,
         "stickers": [{"k": "emoji", "e": emoji, "s": 0.28}] if emoji else []}
    return normal(d)


def arrange(ctx, design, preset):
    """layout() for this video's picture (action measured at the design's second)."""
    d = normal(design)
    return layout(d, preset, action_at(ctx, d["bg"]["t"]), ctx.fa, ctx.avoid)


# ---------------------------------------------------------------- what the video gives us
def _moments(ctx):
    """moments.json (SPEC §6: [{s, e, title, hook, why, type, score, segments}]) without deleted / skipped ones."""
    m = load_json(ctx.dir / "moments.json", [])
    if isinstance(m, dict):
        m = m.get("moments") or []
    out = []
    for x in m if isinstance(m, list) else []:
        if not isinstance(x, dict) or x.get("deleted") or x.get("skip"):
            continue
        try:
            s, e = float(x.get("s")), float(x.get("e"))
        except (TypeError, ValueError):
            continue
        if e > s:
            out.append(x)
    return out


def _segments(m):
    segs = []
    for seg in m.get("segments") or []:
        try:
            a, b = float(seg[0]), float(seg[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b > a:
            segs.append((a, b))
    return segs or [(float(m["s"]), float(m["e"]))]


def _main_span(m):
    return max(_segments(m), key=lambda ab: ab[1] - ab[0])


def _walk_texts(obj, words, out, key=""):
    """Every short string under a key that contains one of `words` (any depth of texts.json)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_texts(v, words, out, str(k).lower())
    elif isinstance(obj, list):
        for v in obj:
            _walk_texts(v, words, out, key)
    elif isinstance(obj, str) and any(w in key for w in words) and \
            not any(b in key for b in ("description", "hashtag", "caption", "tags", "chapter")):
        out.append(obj)
    return out


def _texts_json(ctx):
    return load_json(ctx.dir / "texts.json", {}) or {}


def suggestions(ctx, moments=None):
    """Words to click in the studio: thumbnail texts from texts.json first, then the moments' hooks and titles,
    then pieces of the video titles. Short ones only (2-6 words read best)."""
    out, seen = [], set()

    def add(t):
        s, emo = split_emoji(t)
        s = s.strip(" .،!?؟:") if len(s) > 30 else s
        if s and len(s.split()) <= 6 and s.lower() not in seen:
            seen.add(s.lower())
            out.append({"s": s, "e": next((e for e in emo if emoji_file(e)), "")})
    tx = _texts_json(ctx)
    for t in _walk_texts(tx, ("thumb",), []):
        add(t)
    for m in moments if moments is not None else _moments(ctx):
        add(str(m.get("hook") or ""))
        add(str(m.get("title") or ""))
    for t in _walk_texts(tx, ("title",), []):
        for part in re.split(r"\s[|｜–—-]\s|[|｜]|:\s", t):
            add(part)
    return out[:16]


def video_title(ctx):
    """The title for the YouTube previews: the first title BroClips wrote, else the video's name."""
    titles = _walk_texts(_texts_json(ctx), ("title",), [])
    return split_emoji(titles[0])[0] if titles else ctx.name


def _spread(ctx, n):
    """n seconds spread over the video (not the first / last 3 s: fades, intros)."""
    a, b = (3.0, ctx.dur - 3.0) if ctx.dur > 12 else (0.0, max(0.1, ctx.dur - 0.2))
    return [round(a + (b - a) * (k + 0.5) / n, 1) for k in range(n)]


def candidates(ctx, moments=None, used=()):
    """Pictures the studio offers: the ones other thumbnails use, three from every moment, then spread over the
    whole video."""
    moments = _moments(ctx) if moments is None else moments
    out = []

    def add(t, i, title):
        try:
            t = round(float(t), 1)
        except (TypeError, ValueError):
            return
        if 0 <= t < ctx.dur and all(abs(t - c["t"]) > 2.5 for c in out):
            out.append({"t": t, "i": i, "title": title})
    for t in used:
        add(t, 0, "")
    for i, m in enumerate(moments, 1):
        a, b = _main_span(m)
        for f in (0.3, 0.55, 0.8):
            add(a + (b - a) * f, i, split_emoji(str(m.get("title") or m.get("hook") or ""))[0])
    for t in _spread(ctx, 14 if moments else 24):
        add(t, 0, "")
    return out[:40]


def strip(ctx, m, n=12):
    """n moments spread over all the parts of one moment (to pick the exact picture)."""
    segs = _segments(m)
    total = sum(b - a for a, b in segs) or 1
    out = []
    for k in range(n):
        x = total * (k + 0.5) / n
        for a, b in segs:
            if x <= b - a:
                out.append(round(a + x, 1))
                break
            x -= b - a
    return out


def _lines(ctx):
    """[(start, end, text)] of what is said (lines.json, else transcript.json words in ~6 s pieces)."""
    if ctx.key in _LINES:
        return _LINES[ctx.key]
    out = []
    ls = load_json(ctx.dir / "lines.json")
    if isinstance(ls, dict):
        ls = ls.get("lines")
    for x in ls if isinstance(ls, list) else []:
        try:
            out.append((float(x["s"]), float(x["e"]), str(x.get("text") or "")))
        except (TypeError, ValueError, KeyError):
            continue
    if not out:
        words = (_dict(load_json(ctx.dir / "transcript.json", {}))).get("words") or []
        cur, start = [], None
        for w in words:
            try:
                s, e, txt = float(w["s"]), float(w["e"]), str(w["w"])
            except (TypeError, ValueError, KeyError):
                continue
            if start is None:
                start = s
            cur.append(txt)
            if e - start >= 6:
                out.append((start, e, " ".join(cur)))
                cur, start = [], None
        if cur and start is not None:
            out.append((start, start + 6, " ".join(cur)))
    _LINES[ctx.key] = out
    if len(_LINES) > 20:
        _LINES.pop(next(iter(_LINES)))
    return out


def said_at(ctx, t, span=4.0, limit=110):
    """What is said around second t (for the AI: what is this picture about)."""
    parts = [x for s, e, x in _lines(ctx) if e >= t - span and s <= t + span]
    s = " ".join(parts).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def content_language(ctx):
    """The project's language code; "auto" = the transcript's language when there is one, else English."""
    lang = str(ctx.project.get("language") or "auto")
    if lang == "auto":
        lang = str((_dict(load_json(ctx.dir / "transcript.json", {}))).get("language") or "en")
    return lang.split("-")[0].lower() or "en"


def _face_for(text, video_type=""):
    """A reaction face that fits a moment (its type / why / hook)."""
    t = str(text or "").lower()
    for pat, e in ((r"funn|laugh|joke|comed|lol|هه|ضحك", "😂"), (r"fail|died|dead|death|mistake|wrong|oops", "💀"),
                   (r"shock|surpris|scar|insane|crazy|wow|unbeliev", "😱"), (r"angry|rage|mad\b|annoy", "😡"),
                   (r"sad|emotion|cry|tears", "😭"),
                   (r"tip|learn|lesson|teach|explain|trick|how to|insight|secret|fact", "🤯")):
        if re.search(pat, t):
            return e
    return "🤯" if video_type == "screen" else "😳"


# ---------------------------------------------------------------- files
def thumbs_dir(pid, vid):
    return projects.video_dir(pid, vid) / "thumbs"


def thumb_files(pid, vid):
    d = thumbs_dir(pid, vid)
    if not d.is_dir():
        return []
    return sorted((f for f in d.glob("thumb_*.jpg") if re.fullmatch(r"thumb_\d+\.jpg", f.name)),
                  key=lambda f: int(re.findall(r"\d+", f.name)[0]))


def thumb_list(pid, vid):
    """[{"n", "file" (relative to the video folder, for /files/<pid>/<vid>/...), "v" (changes when it changes)}]"""
    out = []
    for f in thumb_files(pid, vid):
        try:
            out.append({"n": int(re.findall(r"\d+", f.name)[0]), "file": f"thumbs/{f.name}",
                        "v": int(f.stat().st_mtime)})
        except OSError:
            continue
    return out


def next_n(pid, vid):
    return max((int(re.findall(r"\d+", f.name)[0]) for f in thumb_files(pid, vid)), default=0) + 1


def save(ctx, n, design):
    """Draw the design at full quality and keep it as thumbs/thumb_<n>.jpg (+ .json to open it again).
    n=None: the next free number. -> n"""
    d = normal(design)
    out_dir = thumbs_dir(ctx.pid, ctx.vid)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(n) if n else next_n(ctx.pid, ctx.vid)
    img, _ = draw(d, ctx)
    data = jpeg(img, 92)
    if len(data) > 1_900_000:                           # YouTube refuses thumbnails over 2 MB
        data = jpeg(img, 80)
    f = out_dir / f"thumb_{n}.jpg"
    save_json(d, f.with_suffix(".json"))
    tmp = part_path(f)
    tmp.write_bytes(data)
    finish(tmp, f)
    for slot, k in ((1, "subject"), (2, "subject2")):  # the subjects it shows are remembered for this video
        if d[k] and d[k]["on"] and ready(d[k]["id"]):
            remember_subject(ctx.pid, ctx.vid, d[k]["id"], slot=slot)
    return n


def delete(pid, vid, n):
    """Thumbnail n (and its design) to the Recycle Bin."""
    f = thumbs_dir(pid, vid) / f"thumb_{int(n)}.jpg"
    return projects.to_recycle_bin([p for p in (f, f.with_suffix(".json")) if p.exists()])


def making(pid, vid):
    """True while "Make" works on this video or waits in line for it (it writes the thumbnails too)."""
    line, _ = _line_jobs()
    return any(j.get("kind") == "make" and j.get("project") == pid and j.get("video") == vid for j in line)


def brief(pid, vid):
    """What the video is about, in the user's words (the studio's "About this video" box) - context for the AI."""
    return str((load_json(projects.video_dir(pid, vid) / "thumb_brief.json", {}) or {}).get("text") or "")


def set_brief(pid, vid, text):
    save_json({"text": str(text or "").strip()[:1500], "when": time.strftime("%Y-%m-%d %H:%M")},
              projects.video_dir(pid, vid) / "thumb_brief.json")


def _seen(pid, vid):
    """(picture second, style, words) of every thumbnail the video already has - so new ones are really new."""
    keys = set()
    for f in thumb_files(pid, vid):
        d = load_json(f.with_suffix(".json"))
        if isinstance(d, dict):
            try:
                keys.add((round(float(d["bg"]["t"]), 1), d.get("preset"), (d.get("text") or {}).get("s", "")))
            except (KeyError, TypeError, ValueError):
                continue
    return keys


def _finish_look(d, color=None, font=None):
    """A colour / font after layout(): box colours only for the Label style, outlined colours for the others; a Latin
    font never for Arabic words."""
    t = d["text"]
    if font in FONTS and not (FONTS[font][4] == "latin" and is_arabic(t["s"])):
        t["font"] = font
    if color in STYLES and (color in BOX_STYLES) == (d["preset"] == "label"):
        t["style"] = color
    return d


# ---------------------------------------------------------------- automatic designs
def picks(ctx, moments=None, n=3, quick=False):
    """Up to n pictures (seconds) from n different top moments - in each, of 4 tries the one with the most action
    (LoL: the most colourful one alone was once a calm shopping moment); black / empty pictures are skipped. Without
    moments: spread over the video. quick=True: the middle of each moment, no measuring (for the studio)."""
    moments = _moments(ctx) if moments is None else moments
    gap = min(20.0, ctx.dur / 6)
    out = []

    def far(t):
        return all(abs(t - u) >= gap for u in out)
    for m in sorted(moments, key=lambda x: -_num(x.get("score"))):
        if len(out) == n:
            break
        a, b = _main_span(m)
        tries = [a + (b - a) * 0.5] if quick else [a + (b - a) * f for f in (0.3, 0.45, 0.6, 0.75)]
        tries = [t for t in tries if far(t) and (quick or not _flat(ctx, t))]
        if tries:
            out.append(round(max(tries, key=lambda t: action_at(ctx, t)[2]) if len(tries) > 1 else tries[0], 1))
    if len(out) < n:                # few or no moments: pictures spread over the video, the most action first
        spare = [t for t in _spread(ctx, 9) if quick or not _flat(ctx, t)]
        if not quick:
            spare.sort(key=lambda t: -action_at(ctx, t)[2])
        for t in spare:
            if len(out) < n and far(t):
                out.append(t)
    return out[:n]


def plan(ctx, has_subject):
    """The 3 styles of the automatic thumbnails (3 different looks for YouTube's "Test & compare")."""
    third = "headline" if ctx.video_type in ("screen", "podcast") else "label"
    if has_subject:
        return ["subject", "zoom", "reaction" if ctx.video_type == "gameplay" else third]
    return ["zoom", "reaction", third]


def _moment_at(moments, t):
    return next((m for m in moments if float(m["s"]) - 1 <= t <= float(m["e"]) + 1), {})


def _words_for(moments, t, texts, k, used=()):
    """The words for a picture: the hook of the moment it comes from (the words then fit the picture), else the k-th
    suggestion not used yet."""
    hook, emo = split_emoji(str(_moment_at(moments, t).get("hook") or ""))
    if hook and len(hook.split()) <= 6 and hook.lower() not in used:
        return {"s": hook, "e": next((e for e in emo if emoji_file(e)), "")}
    free = [x for x in texts if x["s"].lower() not in used] or texts
    return free[k % len(free)] if free else {"s": "", "e": ""}


def fresh(ctx, k, t=None, subject=None, subject2=None, moments=None, texts=None, used=()):
    """The k-th automatic design (k = 0, 1, 2, ...) - or one for picture t - laid out and ready to draw.
    used = words other thumbnails already have (lower case)."""
    moments = _moments(ctx) if moments is None else moments
    texts = suggestions(ctx, moments) if texts is None else texts
    styles = plan(ctx, bool(subject))
    if t is None:
        ts = picks(ctx, moments, quick=True) or _spread(ctx, 3)
        t = ts[k % len(ts)]
    x = _words_for(moments, t, texts, k, used)
    preset = styles[k % 3]
    if preset == "headline" and not x["s"]:
        preset = "label"
    mom = _moment_at(moments, t)
    about = " ".join(str(mom.get(f) or "") for f in ("type", "why", "hook", "title")).strip() or x["s"]
    face = x.get("e") or _face_for(about, ctx.video_type)
    emoji = face if (preset == "reaction" or x.get("e")) and emoji_file(face) else None
    d = new_design(t, x["s"], subject=subject, subject2=subject2, emoji=emoji,
                   style=("yellow", "white", "fire")[k % 3])
    return arrange(ctx, d, preset)


def _person_wanted(ctx):
    return ctx.video_type in ("camera", "podcast")


def auto(pid, vid, first_n=1, wait_cut=True):
    """The 3 automatic thumbnails of a video, each from a different top moment and in a different style + text colour
    (YouTube's "Test & compare" takes 3), saved as thumb_<first_n> .. thumb_<first_n + 2> (those files are
    replaced; pass first_n=next_n(pid, vid) to keep what is there). A subject is used when one was cut out for this
    video before; with wait_cut=True a talking-to-the-camera / podcast video also gets its person cut out of a good
    picture first (~30-60 s on the CPU, needs the cut-out model). Returns the JPG paths; one bad picture never stops
    the others."""
    ctx = context(pid, vid)
    moments = _moments(ctx)
    texts = suggestions(ctx, moments)
    if any(is_arabic(x["s"]) for x in texts) and not RAQM:
        log.warning("thumbnail: Arabic words but no Arabic letter fix (fribidi) - Settings → Download")
    ts = picks(ctx, moments)
    if not ts:
        ts = _spread(ctx, 3)
    s1, s2 = main_subjects(pid, vid)
    if not s1 and wait_cut and _person_wanted(ctx) and cutout_model():
        try:
            sid, cover = cut_from_frame(ctx, ts[0])
            if 0.03 <= cover <= 0.85:
                s1 = sid
                remember_subject(pid, vid, sid, how="frame", slot=1, t=ts[0])
            else:
                log.info("thumbnail: the cut-out at %.1f s covers %.0f %% of the picture - not used", ts[0], 100 * cover)
        except Exception as ex:
            log.warning("thumbnail: no person cut out (%s)", ex)
    done, used = [], set()
    for k in range(3):
        t = ts[k % len(ts)]
        try:
            d = fresh(ctx, k, t=t, subject=s1, subject2=s2, moments=moments, texts=texts, used=used)
            used.add(d["text"]["s"].lower())
            n = save(ctx, first_n + k, d)
            done.append(thumbs_dir(pid, vid) / f"thumb_{n}.jpg")
            log.info("thumbnail %d: %s style, picture at %.1f s", n, d["preset"], t)
        except Exception as ex:
            log.warning("thumbnail %d failed: %s", first_n + k, ex)
    return done


def open_design(ctx, n=None):
    """The design of thumbnail n to edit; a picture without a design file or a new one gets a fresh design.
    -> (design, note)"""
    if n:
        d = load_json(thumbs_dir(ctx.pid, ctx.vid) / f"thumb_{int(n)}.json")
        if isinstance(d, dict):
            return normal(d), ""
        note = "This thumbnail has no design file — here is a new design for it. Press 💾 Save to replace it."
        k, t = int(n) - 1, None
    else:                                       # a new one: a picture and words no other thumbnail of this video uses
        files = thumb_files(ctx.pid, ctx.vid)
        k, note = len(files), ""
        designs = [_dict(load_json(f.with_suffix(".json"))) for f in files]
        used = [_num(_dict(d.get("bg")).get("t"), None) for d in designs]
        cands = candidates(ctx)
        t = next((c["t"] for c in cands if all(u is None or abs(c["t"] - u) > 20 for u in used)),
                 cands[0]["t"] if cands else None)
    s1, s2 = main_subjects(ctx.pid, ctx.vid)
    words = {str(_dict(d.get("text")).get("s") or "").lower() for d in designs} if not n else set()
    return fresh(ctx, k, t=t, subject=s1, subject2=s2, used=words), note


# ---------------------------------------------------------------- more thumbnails: 🎲 random mix
def more(ctx, count=6, seed=None):
    """`count` new RANDOM thumbnails, saved after the ones there are: a random picture, style, words, font, colour
    and emoji, each combination new for this video. Instant (no AI) - it can be pressed as often as wanted."""
    rng = random.Random(seed)
    count = max(1, min(int(count or 6), 20))
    moments = _moments(ctx)
    s1, s2 = main_subjects(ctx.pid, ctx.vid)
    cands = candidates(ctx, moments)
    texts = suggestions(ctx, moments) or [{"s": "", "e": ""}]
    styles = (["subject"] * 3 if s1 else []) + ["zoom"] * 2 + ["reaction"] * 2 + ["label"] + \
        ["headline"] * (2 if ctx.video_type in ("screen", "podcast") else 1) + (["split"] if s1 and s2 else [])
    seen, first, made = _seen(ctx.pid, ctx.vid), next_n(ctx.pid, ctx.vid), []
    for _ in range(count * 6):
        if len(made) >= count:
            break
        c, x, preset = rng.choice(cands), rng.choice(texts), rng.choice(styles)
        if preset == "headline" and not x["s"]:
            continue
        key = (round(float(c["t"]), 1), preset, x["s"])
        if key in seen:
            continue
        seen.add(key)
        emoji = x["e"] or (rng.choice(sorted(FACES)) if preset == "reaction" or rng.random() < 0.6 else None)
        try:
            d = new_design(c["t"], x["s"], subject=s1, subject2=s2 if preset == "split" else None,
                           emoji=emoji if emoji_file(emoji) else None)
            if d["subject"]:
                d["subject"]["glow"] = rng.choice(GLOWS[:5])
            d = arrange(ctx, d, preset)
            d = _finish_look(d, rng.choice(sorted(BOX_STYLES)) if d["preset"] == "label" else
                             rng.choice(OUTLINED), rng.choice(ARABIC_PICKS if is_arabic(x["s"]) else LATIN_PICKS))
            made.append(save(ctx, first + len(made), d))
        except ThumbError as ex:
            log.info("thumbnail random mix: picture at %.1f s skipped (%s)", c["t"], ex)
            continue
    log.info("thumbnail studio: %s/%s - %d random thumbnails from #%d", ctx.pid, ctx.vid, len(made), first)
    return made


# ---------------------------------------------------------------- ✨ AI ideas from a prompt
def count_in(text, default=6):
    """'make 4 thumbnails about...' / 'اعمل ٥ صور' -> the number asked for (1-12), else `default`. The number must
    stand next to a word like thumbnails / pictures / صور ("1v5" or "top 10 tips" is no count)."""
    t = str(text or "").translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
    m = re.search(r"(\d{1,2})\s*(?:new\s+|more\s+|different\s+|other\s+)?(?:thumbnails?|thumbs?|pictures?|"
                  r"images?|designs?|ideas?|versions?|options?|ثامبنيل\w*|صور\w*|تصميم\w*|تصاميم|مصغر\w*|افكار|أفكار|"
                  r"فكر\w*)", t, re.I)
    return max(1, min(12, int(m.group(1)))) if m else default


_WANT = [("split", r"\bvs\b|\bversus\b|ضد|في مواجهة"), ("zoom", r"\bzoom|زووم|زوم"),
         ("reaction", r"\breaction|رياكشن|ريأكشن|\bemoji|ايموجي|إيموجي"),
         ("label", r"\blabel|\bbox\b|بوكس|صندوق|ليبل"),
         ("headline", r"\bheadline|\bbig (?:text|words|title)|\bjust (?:the )?(?:text|words)|مانشيت|كلام كبير|خط كبير"),
         ("subject", r"\bsubject\b|\bmy (?:face|photo|picture|logo)\b|\bcut[ -]?out\b|صورتي|وشي|لوجو")]


def _wants(prompt, has_subject=False, two=False):
    """The styles the user asked for in plain words, in the order asked, as fixed RULES for the first ideas (LoL: a
    local 12B ignores a requested style - asked for "VS" it made a label and a reaction). "vs" / "ضد" = the split
    style, only when two subjects exist; subject words only when there is a subject."""
    p = str(prompt or "").lower()
    found = []
    for style, pat in _WANT:
        m = re.search(pat, p)
        if not m or (style == "split" and not two) or (style == "subject" and not has_subject):
            continue
        found.append((m.start(), style))
    return [s for _, s in sorted(found)]


AI_SYSTEM = """You design YouTube thumbnails for a video creator. Thumbnail words: 2-4 words, very bold, curiosity-
driven, in {lang} unless the creator's request asks for another language{dialect}. Use only what is really in the
video (the creator's description, the moments and what is said) - never invent events, numbers or promises. Follow the
creator's request exactly. Every idea must be clearly different from the others: other words, another picture,
another style."""


def _lang_name(code):
    return LANG_NAMES.get(code, code)


def ai_ideas(ctx, prompt="", count=6, progress=None, llm=None):
    """`count` new thumbnails designed by the Thinking AI from the user's request + what the video is about (one
    question). Pictures are sent only to a local AI, or with Settings' "allow pictures" switch. Clear requests in
    the prompt are enforced in code (_wants). -> (numbers made, a message for the user)"""
    from . import providers
    say = progress or (lambda *a, **k: None)
    count = max(1, min(int(count or 6), 12))
    llm = llm or providers.get_llm()
    moments = _moments(ctx)
    s1, s2 = main_subjects(ctx.pid, ctx.vid)
    styles = ["zoom", "reaction", "label", "headline"] + (["subject"] if s1 else []) + (["split"] if s1 and s2 else [])
    allow = bool(getattr(llm, "local", False) or config.settings().get("allow_images_to_cloud"))
    images = []
    if allow:
        try:
            see = bool(llm.vision)
        except Exception:
            see = False
        if see:
            cands = candidates(ctx, moments)[:8]
            for c in cands:
                try:
                    im = Image.open(frame(ctx, c["t"])).convert("RGB")
                    im.thumbnail((448, 448))
                    images.append(jpeg(im, 80))
                except (ThumbError, OSError):
                    images.append(None)
            keep = [i for i, b in enumerate(images) if b]
            cands, images = [cands[i] for i in keep], [images[i] for i in keep]
    if not images:
        cands = candidates(ctx, moments)[:24]
    if not cands:
        raise ThumbError("No picture found in this video.")
    lang = content_language(ctx)
    dialect = (" - in the same Arabic the creator speaks in the video (for example Egyptian colloquial when they speak "
               "Egyptian), not formal Arabic") if lang == "ar" else ""
    fonts = list(FONTS)
    props = {"text": {"type": "string"}, "style": {"type": "string", "enum": styles},
             "picture": {"type": "integer"}, "color": {"type": "string", "enum": list(STYLES)},
             "font": {"type": "string", "enum": fonts}, "emoji": {"type": "string", "enum": EMOJI_ORDER + [""]},
             "why": {"type": "string"}}
    schema = {"type": "object", "properties": {"ideas": {"type": "array", "minItems": 1, "maxItems": count, "items": {
        "type": "object", "properties": props, "required": list(props)}}}, "required": ["ideas"]}
    moments_txt = "\n".join(f"- {split_emoji(str(m.get('title') or ''))[0]} | {split_emoji(str(m.get('hook') or ''))[0]}"
                            f" | {str(m.get('why') or '')[:160]}" for m in moments[:20]) or "- (none yet)"
    pics = "\n".join(f"{k}: at {int(c['t'] // 60)}:{int(c['t'] % 60):02d}"
                     + (f" from \"{c['title']}\"" if c.get("title") else "")
                     + (f" - said there: \"{said_at(ctx, c['t'])}\"" if said_at(ctx, c["t"]) else "")
                     for k, c in enumerate(cands))
    words = " / ".join(x["s"] for x in suggestions(ctx, moments)[:10]) or "(none)"
    b = brief(ctx.pid, ctx.vid)
    user = (f"THE CHANNEL / PROJECT (the creator's words): {str(ctx.project.get('prompt') or '(not given)')[:800]}\n"
            f"VIDEO TYPE: {ctx.video_type}\n"
            f"WHAT THIS VIDEO IS ABOUT (the creator's words): {b or '(not given)'}\n"
            f"THE CREATOR'S REQUEST FOR THESE THUMBNAILS: {str(prompt or '').strip() or '(none - surprise them, all different)'}\n"
            f"MAKE EXACTLY {count} DIFFERENT THUMBNAILS. Words in {_lang_name(lang)}.\n\n"
            f"MOMENTS IN THIS VIDEO (title | hook | why it is good):\n{moments_txt}\n\n"
            f"EARLIER TEXT IDEAS (reuse or write better ones): {words}\n\n"
            + (f"PICTURES YOU CAN USE ({len(cands)} pictures are attached, in this order - picture number: when):\n"
               if images else "PICTURES YOU CAN USE (picture number: when, from which moment, what is said):\n")
            + pics + "\n\nSTYLES:\n" + "\n".join(f"- {s}: {STYLE_HELP[s]}" for s in styles)
            + "\nCOLOURS: yellow, white, fire, ice, green, pink (outlined words); redbox, yellowbox (words in a box - "
              "only with the label style)\nFONTS: " + ", ".join(
                f"{k} ({FONT_HELP[k]}{', Latin letters only' if FONTS[k][4] == 'latin' else ', Arabic and Latin'})"
                for k in fonts)
            + "\nFor every idea give: text, style, picture number, colour, font, one emoji (or \"\"), and why it is "
              "clickable.")
    say(1, f"✨ Designing {count} thumbnails", "asking the AI…", 0.1)
    j = llm.chat_json(AI_SYSTEM.format(lang=_lang_name(lang), dialect=dialect), user, schema,
                      images=images or None, temperature=0.8, max_tokens=240 * count + 300)
    ideas = [i for i in (_dict(j).get("ideas") or []) if isinstance(i, dict)]
    for k, style in enumerate(_wants(prompt, bool(s1), bool(s1 and s2))):   # the user's clear words win
        if k < len(ideas) and style in styles:
            ideas[k]["style"] = style
    seen, first, made = _seen(ctx.pid, ctx.vid), next_n(ctx.pid, ctx.vid), []
    for k, idea in enumerate(ideas[:count]):
        say(1, f"✨ Designing {count} thumbnails", f"drawing {k + 1} of {min(count, len(ideas))}",
            0.5 + 0.5 * k / max(1, min(count, len(ideas))))
        try:
            pi = idea.get("picture")
            c = cands[pi] if isinstance(pi, int) and 0 <= pi < len(cands) else cands[len(made) % len(cands)]
            preset = idea.get("style") if idea.get("style") in styles else "zoom"
            text, emo = split_emoji(str(idea.get("text") or ""))
            if preset == "headline" and not text:
                preset = "zoom"
            emoji = idea.get("emoji") or (emo[0] if emo else None)
            d = new_design(c["t"], text, subject=s1, subject2=s2 if preset == "split" else None,
                           emoji=emoji if emoji_file(emoji) else None)
            d = arrange(ctx, d, preset)
            d = _finish_look(d, idea.get("color"), idea.get("font"))
            key = (round(float(c["t"]), 1), d["preset"], d["text"]["s"])
            if key in seen:
                continue
            seen.add(key)
            made.append(save(ctx, first + len(made), d))
        except ThumbError as ex:
            log.info("thumbnail AI: idea %d skipped (%s)", k + 1, ex)
            continue
    log.info("thumbnail AI: %s/%s - %d of %d asked (%d ideas came back)", ctx.pid, ctx.vid, len(made), count,
             len(ideas))
    if not ideas:
        return made, "The AI did not answer with ideas — try again (or use 🎲 Random mix)."
    return made, f"Made {len(made)} new thumbnail{'s' if len(made) != 1 else ''} from your idea" + \
        (f" ({count - len(made)} were the same as ones you have)." if len(made) < count else ".")


def _ai_job(job, progress, cancelled):
    """The `thumbai` job: args {"prompt", "count"}."""
    a = _dict(job.get("args"))
    pid, vid = job.get("project"), job.get("video")
    try:
        ctx = context(pid, vid)
        made, msg = ai_ideas(ctx, str(a.get("prompt") or ""), int(a.get("count") or 6), progress)
    except ThumbError as ex:
        _AI_LAST[(pid, vid)] = {"job": job["id"], "made": [], "msg": "⚠️ " + str(ex)}
        raise jobs.JobError(str(ex)) from None
    except Exception as ex:
        _AI_LAST[(pid, vid)] = {"job": job["id"], "made": [], "msg": f"⚠️ The AI thumbnails failed ({ex})"[:300]}
        raise
    text = ("✅ " if made else "⚠️ ") + msg
    _AI_LAST[(pid, vid)] = {"job": job["id"], "made": made, "msg": text}
    return text


def start_ai(pid, vid, prompt="", count=None, title=""):
    """Put "✨ Make with AI" into the work line. -> (job, count)"""
    prompt = str(prompt or "")[:1500]
    n = count_in(prompt) if count in (None, "") else max(1, min(12, int(count)))
    job = jobs.enqueue("thumbai", project=pid, video=vid, args={"prompt": prompt, "count": n},
                       title=title or "AI thumbnails")
    return job, n


def ai_state(pid, vid, job_id=None):
    """What the studio shows about ✨ AI thumbnails of this video: working / in line / waiting / done + message."""
    line, waiting_for = _line_jobs()
    for k, j in enumerate(line):
        if j.get("kind") == "thumbai" and j.get("project") == pid and j.get("video") == vid and \
                (not job_id or j.get("id") == job_id):
            if j.get("running"):
                return {"state": "working", "job": j["id"]}
            ahead = k
            if waiting_for and not j.get("force"):
                return {"state": "waiting", "job": j["id"], "waiting_for": waiting_for, "ahead": ahead}
            return {"state": "queued", "job": j["id"], "ahead": ahead}
    last = _AI_LAST.get((pid, vid))
    if last and (not job_id or last.get("job") == job_id):
        return {"state": "done", "job": last.get("job"), "msg": last.get("msg", ""), "made": last.get("made", [])}
    r = jobs.result_for(pid, vid, kind="thumbai")
    if r and (not job_id or r.get("id") == job_id):
        return {"state": "done", "job": r.get("id"), "msg": r.get("msg", ""), "made": []}
    return {"state": "none"}


jobs.register("cutout", _cut_job, steps=1, title="Cut-out")
jobs.register("thumbai", _ai_job, steps=1, title="AI thumbnails")

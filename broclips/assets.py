"""Downloads the free files BroClips uses (port of LoL's tools/get_thumb_assets.py): rate-limited so a game stays
smooth, resumable (.part files), hash-checked where a hash is known, files already there are skipped.

  fonts      Google Fonts (OFL, + their OFL.txt): Arabic Lalezar, Alexandria, Cairo, Changa, Marhey, Jomhuria;
             Latin Montserrat, Anton, Bebas Neue; + the static Montserrat ExtraBold from the Montserrat project
             (Google Fonts only has the variable file, which libass cannot use as ExtraBold for captions)
  emoji      Microsoft Fluent Emoji 3D pictures (MIT, + LICENSE)
  fribidi    fribidi-0.dll from conda-forge (LGPL-2.1, sha256 checked; Windows only) - Pillow needs it for Arabic
  cut-out    BiRefNet general (928 MB, best) or general-lite (214 MB) ONNX from rembg's releases (MIT, md5 from
             rembg's source) -> config.models_dir()

  python -m broclips.assets --essential            fonts + emoji + fribidi
  python -m broclips.assets --cutout general|lite  the cut-out model for thumbnails
  python -m broclips.assets --from DIR ...         copy matching files found in DIR first (DIR alone = essential)
      --rate BYTES   speed limit in bytes per second (default 1000000, about 1 MB/s)
      --no-download  only copy from DIR, never download

Settings' "Download" buttons run the same code as an `assets` job in the work line."""
import argparse
import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

from . import config, jobs
from .util import NO_WINDOW, log

RATE = 1_000_000                    # bytes per second (~8 Mbit/s): leaves room for a game
GF = "https://raw.githubusercontent.com/google/fonts/main/ofl"
FE = "https://raw.githubusercontent.com/microsoft/fluentui-emoji/main"
# (Google Fonts folder, file, family name used for OFL-<family>.txt)
FONTS = [("lalezar", "Lalezar-Regular.ttf", "Lalezar"), ("alexandria", "Alexandria[wght].ttf", "Alexandria"),
         ("cairo", "Cairo[slnt,wght].ttf", "Cairo"), ("changa", "Changa[wght].ttf", "Changa"),
         ("marhey", "Marhey[wght].ttf", "Marhey"), ("jomhuria", "Jomhuria-Regular.ttf", "Jomhuria"),
         ("montserrat", "Montserrat[wght].ttf", "Montserrat"), ("anton", "Anton-Regular.ttf", "Anton"),
         ("bebasneue", "BebasNeue-Regular.ttf", "BebasNeue")]
EXTRA_FONTS = [("https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-ExtraBold.ttf",
                "Montserrat-ExtraBold.ttf")]
EMOJI = ["Face with tears of joy", "Rolling on the floor laughing", "Skull", "Face screaming in fear", "Pouting face",
         "Exploding head", "Loudly crying face", "Smiling face with horns", "Clown face", "Eyes", "Fire",
         "Hundred points", "Crown", "Face with steam from nose", "Cold face", "Melting face", "Flushed face",
         "Face with symbols on mouth", "Smiling face with sunglasses", "Face with rolling eyes", "Thinking face",
         "Warning", "Cross mark", "Check mark button", "Collision", "High voltage", "Trophy", "Moai", "Smirking face",
         "Grimacing face", "Saluting face", "Face holding back tears", "Pleading face", "Zany face",
         "Face with hand over mouth", "Nauseated face", "Ghost", "Bomb", "Crossed swords", "Bullseye", "Sparkles",
         "Red question mark", "Red exclamation mark", "Money bag", "Snail", "Goat", "Rocket"]
EMOJI_LICENSE = "LICENSE-fluentui-emoji.txt"
FRIBIDI = ("https://conda.anaconda.org/conda-forge/win-64/fribidi-1.0.17-hfd05255_0.conda",
           "30a93a5132923a5679e0729a02bf5b36052c84cc437decc14a35caee5e276f46")
FRIBIDI_DLL = "fribidi-0.dll"
REMBG = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
CUTOUT = {      # sizes from the release server (HEAD, 2026-10-07); md5 from rembg's own source (sessions/*.py)
    "general": {"file": "BiRefNet-general-epoch_244.onnx", "md5": "7a35a0141cbbc80de11d9c9a28f52697",
                "size": 972_666_916, "label": "full cut-out model (928 MB, best)"},
    "lite": {"file": "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", "md5": "4fab47adc4ff364be1713e97b7e66334",
             "size": 224_005_088, "label": "light cut-out model (214 MB)"},
}


def fonts_dir():
    return config.assets_dir() / "fonts"


def emoji_dir():
    return config.assets_dir() / "emoji"


def bin_dir():
    return config.assets_dir() / "bin"


def emoji_stem(name):
    """'Face with tears of joy' -> 'face_with_tears_of_joy' (the file is <stem>.png)."""
    return name.lower().replace(" ", "_").replace("-", "_")


def _wanted_essential():
    """[(local path, url or None, min_bytes)] of every essential file."""
    out = []
    for folder, file, fam in FONTS:
        out.append((fonts_dir() / file, f"{GF}/{folder}/{quote(file, safe='')}", 20_000))
        out.append((fonts_dir() / f"OFL-{fam}.txt", f"{GF}/{folder}/OFL.txt", 1000))
    for url, file in EXTRA_FONTS:
        out.append((fonts_dir() / file, url, 20_000))
    out.append((emoji_dir() / EMOJI_LICENSE, f"{FE}/LICENSE", 200))
    for name in EMOJI:
        stem = emoji_stem(name)
        out.append((emoji_dir() / f"{stem}.png", f"{FE}/assets/{quote(name)}/3D/{stem}_3d.png", 2000))
    return out


def cutout_path(variant):
    return config.models_dir() / CUTOUT[variant]["file"]


def cutout_model_path():
    """The cut-out model to use: Settings' path if it exists, else the full model, else the light one, else None."""
    p = str(config.settings().get("cutout_model") or "").strip()
    if p and Path(p).is_file():
        return Path(p)
    for variant in ("general", "lite"):
        if cutout_path(variant).is_file():
            return cutout_path(variant)
    return None


def status():
    """What is there (for Settings)."""
    fonts = [fonts_dir() / f for _, f, _ in FONTS] + [fonts_dir() / f for _, f in EXTRA_FONTS]
    emo = [emoji_dir() / f"{emoji_stem(n)}.png" for n in EMOJI]
    have_fonts, have_emoji = sum(f.is_file() for f in fonts), sum(f.is_file() for f in emo)
    fribidi = (bin_dir() / FRIBIDI_DLL).is_file() if config.IS_WINDOWS else None
    model = cutout_model_path()
    return {"fonts": {"have": have_fonts, "of": len(fonts)}, "emoji": {"have": have_emoji, "of": len(emo)},
            "fribidi": fribidi,
            "essential_ok": have_fonts == len(fonts) and have_emoji == len(emo) and fribidi is not False,
            "cutout": {v: cutout_path(v).is_file() for v in CUTOUT} | {"path": str(model) if model else ""},
            "cutout_sizes": {v: c["label"] for v, c in CUTOUT.items()}}


# ---------- downloading ----------
def _sleep(sec, cancelled):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        if cancelled and cancelled():
            raise jobs.Cancelled()
        time.sleep(min(0.5, max(0.0, end - time.monotonic())))


def _hash(path, md5=None, sha256=None):
    h = hashlib.md5() if md5 else hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest() == (md5 or sha256).lower()


def _finish(tmp, dst):
    """Rename tmp -> dst, retrying: Windows Defender briefly locks a fresh file (a big one for a while)."""
    for i in range(60):
        try:
            Path(tmp).replace(dst)
            return dst
        except PermissionError:
            if i == 59:
                raise
            time.sleep(1)


def get(url, dst, min_bytes=200, md5=None, sha256=None, rate=None, on_bytes=None, cancelled=None):
    """Download url -> dst at most `rate` bytes/s, resuming dst.part; hash-checked when a hash is given.
    on_bytes(done, total) reports progress; cancelled() stops it (the .part stays for next time)."""
    rate = rate or RATE
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size >= min_bytes:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(6):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": f"BroClips/{config.VERSION}"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                resumed = bool(have) and r.status == 206
                total = int(r.headers.get("Content-Length") or 0) + (have if resumed else 0)
                done = have if resumed else 0
                t0, sent = time.time(), 0
                with open(tmp, "ab" if resumed else "wb") as f:
                    while True:
                        if cancelled and cancelled():
                            raise jobs.Cancelled()
                        chunk = r.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        sent += len(chunk)
                        done += len(chunk)
                        if on_bytes:
                            on_bytes(done, total)
                        ahead = sent / rate - (time.time() - t0)
                        if ahead > 0:
                            _sleep(ahead, cancelled)
            break
        except urllib.error.HTTPError as ex:
            if ex.code == 416 and have:             # the .part is already complete (only its rename failed)
                break
            if ex.code in (403, 404, 410):
                raise RuntimeError(f"{dst.name}: the download address does not work any more ({ex.code}).")
            log.info("   retry %d for %s: %s", attempt + 1, dst.name, ex)
            _sleep(10 * (attempt + 1), cancelled)
        except (OSError, ValueError) as ex:       # network trouble (URLError, timeouts, resets)
            log.info("   retry %d for %s: %s", attempt + 1, dst.name, ex)
            _sleep(10 * (attempt + 1), cancelled)
    else:
        raise RuntimeError(f"Could not download {dst.name}. Check the internet connection and try again.")
    if (md5 or sha256) and not _hash(tmp, md5, sha256):
        tmp.unlink()
        raise RuntimeError(f"{dst.name}: the download was damaged (wrong checksum), so it was thrown away. "
                           "Try again.")
    if tmp.stat().st_size < min_bytes:
        tmp.unlink()
        raise RuntimeError(f"{dst.name}: the download is too small to be right. Try again.")
    return _finish(tmp, dst)


def _unzstd(data):
    """Unpack zstd data with whatever this PC has: the 'zstandard' module, Python 3.14's compression.zstd, or
    Node.js (one hidden call)."""
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    except ImportError:
        pass
    try:
        from compression import zstd                    # Python 3.14+
        return zstd.decompress(data)
    except ImportError:
        pass
    node = shutil.which("node")
    if node:
        p = subprocess.run([node, "-e", "process.stdout.write(require('zlib').zstdDecompressSync("
                                        "require('fs').readFileSync(0)))"],
                           input=data, capture_output=True, creationflags=NO_WINDOW, timeout=300)
        if p.returncode == 0 and p.stdout:
            return p.stdout
        log.info("node could not unpack zstd (Node 22.15 or newer is needed): %s", p.stderr[-300:])
    raise RuntimeError("To unpack the Arabic letter fix (fribidi), BroClips needs the Python module 'zstandard' "
                       "or Node.js 22.15+. Install one (for example: python -m pip install zstandard) and run this "
                       "again. Until then, Arabic words in thumbnails may look broken.")


def fribidi(rate=None, on_bytes=None, cancelled=None):
    """fribidi-0.dll (Windows only): Pillow's raqm needs it to join and order Arabic letters."""
    out = bin_dir() / FRIBIDI_DLL
    if out.exists() or not config.IS_WINDOWS:
        return
    pkg = get(FRIBIDI[0], config.cache_dir() / "downloads" / FRIBIDI[0].rsplit("/", 1)[1], min_bytes=10_000,
              sha256=FRIBIDI[1], rate=rate, on_bytes=on_bytes, cancelled=cancelled)
    found = False
    with zipfile.ZipFile(pkg) as z:
        for name in z.namelist():
            if not name.endswith(".tar.zst"):
                continue
            with tarfile.open(fileobj=io.BytesIO(_unzstd(z.read(name)))) as t:
                for m in t.getmembers():
                    low = m.name.lower()
                    if low.endswith("bin/" + FRIBIDI_DLL) and m.isfile():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        tmp = out.with_name(out.name + ".part")
                        tmp.write_bytes(t.extractfile(m).read())
                        _finish(tmp, out)
                        found = True
                    elif "licenses/" in low and m.isfile():
                        bin_dir().mkdir(parents=True, exist_ok=True)
                        (bin_dir() / ("fribidi-" + Path(m.name).name + ".txt")).write_bytes(t.extractfile(m).read())
    if not found:
        raise RuntimeError("fribidi-0.dll was not found inside the downloaded package.")


def _free_space_ok(folder, need):
    try:
        return shutil.disk_usage(folder).free > need + 300_000_000
    except OSError:
        return True


def cutout(variant, rate=None, on_bytes=None, cancelled=None):
    c = CUTOUT[variant]
    dst = cutout_path(variant)
    if dst.exists():
        return dst
    have = dst.with_name(dst.name + ".part")
    need = c["size"] - (have.stat().st_size if have.exists() else 0)
    if not _free_space_ok(dst.parent, need):
        raise RuntimeError(f"Not enough free space in {dst.parent} for the {c['label']}.")
    return get(REMBG + c["file"], dst, min_bytes=c["size"] // 2, md5=c["md5"], rate=rate, on_bytes=on_bytes,
               cancelled=cancelled)


# ---------- copying from a folder you already have ----------
def _index(src, limit=50_000):
    """{file name (lower case): path} of every file under src (first one wins)."""
    out = {}
    for i, p in enumerate(Path(src).rglob("*")):
        if i >= limit:
            break
        try:
            if p.is_file():
                out.setdefault(p.name.lower(), p)
        except OSError:
            pass
    return out


def copy_from(src, essential=True, variants=(), cancelled=None):
    """Copy the files BroClips wants from a folder that already has them (e.g. LoL Clips' assets). Returns how many."""
    src = Path(src).expanduser()
    if not src.is_dir():
        raise RuntimeError(f"Folder not found: {src}")
    idx = _index(src)
    targets = []
    if essential:
        targets += [dst for dst, _, _ in _wanted_essential()]
        if config.IS_WINDOWS:
            targets += [bin_dir() / FRIBIDI_DLL, bin_dir() / "fribidi-COPYING.txt"]
    for v in variants:
        targets.append(cutout_path(v))
    n = 0
    for dst in targets:
        if cancelled and cancelled():
            raise jobs.Cancelled()
        hit = idx.get(dst.name.lower())
        if dst.exists() or hit is None or hit.resolve() == dst.resolve():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(hit, tmp)
        variant = next((v for v in variants if cutout_path(v) == dst), None)
        if variant and not _hash(tmp, md5=CUTOUT[variant]["md5"]):
            tmp.unlink()
            log.warning("%s in %s is damaged or a different version (checksum) - not used", dst.name, src)
            continue
        _finish(tmp, dst)
        n += 1
    log.info("copied %d files from %s", n, src)
    return n


# ---------- one call for the CLI and the work line ----------
def run(essential=False, variant=None, src=None, rate=RATE, download=True, progress=None, cancelled=None):
    """Copy (from src) and/or download. Returns a list of plain-English problems ([] = all fine)."""
    def say(label, detail="", frac=None):
        if progress:
            progress(1, label, detail, frac)
        else:
            log.info("%s %s", label, detail)

    variants = [variant] if variant else []
    problems = []
    if src:
        say("Copying files", str(src))
        try:
            copy_from(src, essential=essential, variants=variants, cancelled=cancelled)
        except (OSError, RuntimeError) as ex:
            problems.append(f"Copying from {src} failed: {ex}")
    if not download:
        return problems
    if essential:
        todo = [w for w in _wanted_essential() if not (w[0].exists() and w[0].stat().st_size >= w[2])]
        for k, (dst, url, min_bytes) in enumerate(todo):
            say("Downloading fonts and emoji", f"{k + 1} of {len(todo)}: {dst.name}", k / max(1, len(todo)))
            try:
                get(url, dst, min_bytes=min_bytes, rate=rate, cancelled=cancelled)
            except (OSError, RuntimeError) as ex:
                problems.append(str(ex))
        if config.IS_WINDOWS and not (bin_dir() / FRIBIDI_DLL).exists():
            say("Downloading the Arabic letter fix (fribidi)")
            try:
                fribidi(rate=rate, cancelled=cancelled)
            except (OSError, RuntimeError, tarfile.TarError, zipfile.BadZipFile, subprocess.SubprocessError) as ex:
                problems.append(str(ex))
    if variant:
        c = CUTOUT[variant]
        mb = c["size"] / 1e6
        last = {"t": 0.0}

        def on_bytes(done, total):
            if time.monotonic() - last["t"] >= 1.0:
                last["t"] = time.monotonic()
                say(f"Downloading the {c['label']}", f"{done / 1e6:.0f} of {(total or c['size']) / 1e6:.0f} MB",
                    done / (total or c["size"]))
        say(f"Downloading the {c['label']}", f"0 of {mb:.0f} MB", 0.0)
        try:
            cutout(variant, rate=rate, on_bytes=on_bytes, cancelled=cancelled)
        except (OSError, RuntimeError) as ex:
            problems.append(str(ex))
    return problems


def _job(job, progress, cancelled):
    """The `assets` job (Settings' Download buttons): args {"essential": bool, "cutout": "general"|"lite"}."""
    a = job.get("args") or {}
    variant = a.get("cutout") if a.get("cutout") in CUTOUT else None
    problems = run(essential=bool(a.get("essential")), variant=variant, rate=int(a.get("rate") or RATE),
                   progress=progress, cancelled=cancelled)
    if problems:
        raise jobs.JobError(" · ".join(problems)[:600])
    return "✅ Downloaded" + (f" the {CUTOUT[variant]['label']}" if variant else " the fonts and emoji")


jobs.register("assets", _job, steps=1, title="Downloads")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m broclips.assets",
                                 description="Download (or copy) the free fonts, emoji and models BroClips uses.")
    ap.add_argument("--essential", action="store_true", help="fonts + 3D emoji + the Arabic letter fix (Windows)")
    ap.add_argument("--cutout", choices=sorted(CUTOUT), help="the cut-out AI model for thumbnails: general (928 MB, "
                                                             "best) or lite (214 MB)")
    ap.add_argument("--from", dest="src", metavar="DIR", help="copy matching files from this folder first")
    ap.add_argument("--rate", type=int, default=RATE, help="speed limit in bytes per second (default 1000000)")
    ap.add_argument("--no-download", action="store_true", help="only copy from --from, never download")
    a = ap.parse_args(argv)
    if not (a.essential or a.cutout or a.src):
        ap.print_help()
        return 2
    essential = a.essential or (bool(a.src) and not a.cutout)
    print(f"BroClips downloads -> {config.assets_dir()}" + (f" and {config.models_dir()}" if a.cutout else ""))
    print(f"Speed limit: {a.rate / 1e6:.1f} MB/s. You can stop with Ctrl+C and run it again later (it resumes).")
    try:
        problems = run(essential=essential, variant=a.cutout, src=a.src, rate=max(10_000, a.rate),
                       download=not a.no_download)
    except (KeyboardInterrupt, jobs.Cancelled):
        print("Stopped. Run it again later to continue.")
        return 1
    st = status()
    print(f"Fonts {st['fonts']['have']}/{st['fonts']['of']} | emoji {st['emoji']['have']}/{st['emoji']['of']}"
          + ("" if st["fribidi"] is None else f" | Arabic letter fix {'OK' if st['fribidi'] else 'missing'}")
          + f" | cut-out model: {st['cutout']['path'] or 'not installed'}")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

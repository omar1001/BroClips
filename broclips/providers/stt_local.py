"""Local speech-to-text with faster-whisper (SPEC §4.2): free, nothing leaves the PC. Ported from LoL Clips
asr.py (transcribe, _looks_hallucinated) and util.cuda_dll_setup.

Model "auto" = large-v3 on an NVIDIA GPU (float16), else small on the CPU (int8: slower, works everywhere); if the
GPU fails while listening (e.g. missing CUDA libraries) an "auto" setup falls back to the CPU once. The model cache
is opts["models_dir"] (used like HF_HOME, i.e. <models_dir>/hub), else the HF_HOME environment variable, else the
Hugging Face default — the first use downloads the model there. Word times come from word_timestamps=True, silence
is skipped by the VAD filter, the user's prompt (names, words, style) goes in as initial_prompt, and segments that
look made up (Whisper invents text over silence or loops a phrase) are dropped. Every number is a plain float:
numpy floats break JSON."""
import gc
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

from .. import config
from ..util import log
from .base import STT, ProviderError

NO_FASTER_WHISPER = ("Local listening needs faster-whisper. Run install.bat again and answer Y to 'Local "
                     "speech-to-text', or pick a cloud listening AI in Settings.")
_LOADED = {"key": None, "model": None}      # the one Whisper model in memory, shared by all LocalSTT instances
_LOCK = threading.Lock()
# model name -> Hugging Face repo, where it is not "Systran/faster-whisper-<name>" (faster_whisper.utils._MODELS)
_REPOS = {"large": "Systran/faster-whisper-large-v3", "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
          "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
          "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
          "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
          "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}


def _installed(module):
    """Is a package installed? (found without importing it)"""
    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _looks_hallucinated(seg_text, avg_logprob, no_speech_prob):
    words = (seg_text or "").split()
    if not words:
        return True
    if no_speech_prob > 0.6 and avg_logprob < -0.8:
        return True
    if len(words) >= 8 and len(set(words)) <= len(words) / 4:     # "خد سر وخد سر وخد سر ..." loops
        return True
    return False


_DLLS = {"done": False}


def cuda_dll_setup():
    """CTranslate2 on Windows needs the pip-installed NVIDIA DLLs (cuBLAS, cuDNN, NVRTC) on the DLL search path.
    Looks in the `nvidia` package wherever pip put it (a venv or a normal Python). Runs once."""
    if _DLLS["done"] or not config.IS_WINDOWS:
        return
    _DLLS["done"] = True
    roots = [Path(sys.executable).parent.parent / "Lib" / "site-packages" / "nvidia"]      # LoL Clips' way (venvs)
    try:
        import nvidia                                      # namespace package of nvidia-cublas-cu12 & co.
        roots += [Path(p) for p in getattr(nvidia, "__path__", [])]
    except ImportError:
        pass
    seen = set()
    for root in roots:
        for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            d = root / sub / "bin"
            if d.is_dir() and str(d).lower() not in seen:
                seen.add(str(d).lower())
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def cuda_available():
    cuda_dll_setup()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _is_gpu_error(ex):
    text = str(ex).lower()
    return any(k in text for k in ("cuda", "cublas", "cudnn", "nvrtc", "gpu", ".dll"))


class LocalSTT(STT):
    name = "local"
    local = True

    def __init__(self, opts=None):
        self.opts = dict(opts or {})
        self.model_opt = str(self.opts.get("model") or "auto").strip()
        self.device_opt = str(self.opts.get("device") or "auto").strip().lower()
        self.models_dir = str(self.opts.get("models_dir") or "").strip()
        self._cpu_only = False                 # set after a GPU failure in "auto" mode
        self.device = self.model_name = None   # what this instance chose

    # ---------- choosing ----------
    def _pick_device(self):
        if self._cpu_only or self.device_opt == "cpu":
            return "cpu"
        if self.device_opt in ("cuda", "gpu"):
            if not cuda_available():
                raise ProviderError("No NVIDIA graphics card was found for listening. Set the listening device to "
                                    "Auto or CPU in Settings.")
            return "cuda"
        return "cuda" if cuda_available() else "cpu"

    def _pick_model(self, device):
        if self.model_opt.lower() not in ("", "auto"):
            return self.model_opt
        return "large-v3" if device == "cuda" else "small"

    def _download_root(self):
        return str(Path(self.models_dir) / "hub") if self.models_dir else None

    def _hub_dir(self):
        if self.models_dir:
            return Path(self.models_dir) / "hub"
        if os.environ.get("HF_HUB_CACHE"):
            return Path(os.environ["HF_HUB_CACHE"])
        return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"

    def _is_downloaded(self, name):
        if Path(name).is_dir():
            return True
        utils = sys.modules.get("faster_whisper.utils")          # only if already imported (importing is slow)
        repo = getattr(utils, "_MODELS", {}).get(name) or _REPOS.get(name) or (
            name if "/" in name else f"Systran/faster-whisper-{name}")
        snap = self._hub_dir() / ("models--" + repo.replace("/", "--")) / "snapshots"
        return snap.is_dir() and any(snap.iterdir())

    def _load(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ProviderError(NO_FASTER_WHISPER) from None
        device = self._pick_device()
        name = self._pick_model(device)
        compute = "float16" if device == "cuda" else "int8"
        self.device, self.model_name = device, name           # set first: a GPU failure while loading falls back too
        key = (name, device, compute, self._download_root())
        with _LOCK:
            if _LOADED["model"] is not None and _LOADED["key"] == key:
                return _LOADED["model"]
            _LOADED["model"] = _LOADED["key"] = None           # a different model: free the old one first
            gc.collect()
            kw = {"download_root": self._download_root()} if self.models_dir else {}
            t0 = time.monotonic()
            log.info("listening: loading Whisper %s on %s (%s)%s", name, device, compute,
                     "" if self._is_downloaded(name) else " — first use, downloading the model")
            _LOADED["model"] = WhisperModel(name, device=device, compute_type=compute, **kw)
            _LOADED["key"] = key
            log.info("listening: model ready in %.1f s", time.monotonic() - t0)
            return _LOADED["model"]

    # ---------- the job ----------
    def transcribe(self, wav_path, language=None, prompt="", on_progress=None):
        try:
            return self._transcribe(wav_path, language, prompt, on_progress)
        except ProviderError:
            raise
        except Exception as ex:
            if self.device == "cuda" and self.device_opt == "auto" and _is_gpu_error(ex):
                log.warning("listening on the GPU failed (%s) — using the CPU instead", str(ex)[:200])
                self.unload()
                self._cpu_only = True
                return self._transcribe(wav_path, language, prompt, on_progress)
            raise

    def _transcribe(self, wav_path, language, prompt, on_progress):
        lang = (language or "").split("-")[0].strip().lower()
        lang = None if lang in ("", "auto") else lang
        model = self._load()
        t0 = time.monotonic()
        segments, info = model.transcribe(
            str(wav_path), language=lang, task="transcribe", beam_size=5, word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False, initial_prompt=(prompt or "").strip() or None)
        dur = float(getattr(info, "duration", 0) or 0)
        words, dropped = [], 0
        for s in segments:                                         # a generator: the work happens here
            if on_progress and dur > 0:
                on_progress(min(1.0, float(s.end) / dur))
            if _looks_hallucinated(s.text, float(s.avg_logprob), float(s.no_speech_prob)):
                dropped += 1
                continue
            for w in s.words or []:
                t = w.word.strip()
                if t:
                    words.append({"w": t, "s": round(float(w.start), 2), "e": round(float(w.end), 2),
                                  "p": round(float(w.probability), 3)})
        words.sort(key=lambda w: w["s"])
        if on_progress:
            on_progress(1.0)
        log.info("listening: %d words in %.1f s (%s, %s), %d suspicious segments dropped", len(words),
                 time.monotonic() - t0, self.model_name, self.device, dropped)
        return {"language": str(getattr(info, "language", None) or lang or ""), "words": words}

    def unload(self):
        """Free the model (graphics card / RAM) before the next heavy step. The loaded model is shared by every
        LocalSTT (jobs.py unloads through a fresh instance), so any instance can free it."""
        with _LOCK:
            _LOADED["model"] = _LOADED["key"] = None
        gc.collect()

    def test(self):
        """Light check, no model loaded: faster-whisper is installed + which device and model would be used. It does
        not import faster-whisper (PyAV, onnxruntime, …: measured 86 s from a cold USB disk) — only ctranslate2 for
        the GPU check."""
        if not _installed("faster_whisper"):
            raise ProviderError(NO_FASTER_WHISPER)
        device = self._pick_device()
        name = self._pick_model(device)
        where = "GPU" if device == "cuda" else "CPU (no NVIDIA graphics card found — slower)"
        note = "" if self._is_downloaded(name) else " — the model downloads once on first use"
        return f"Ready: {where}, {name}{note}"

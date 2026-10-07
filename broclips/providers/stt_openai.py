"""OpenAI, Groq or any OpenAI-compatible server as the Listening AI (SPEC §4.2).

POST {base_url}/audio/transcriptions (multipart) with response_format=verbose_json and timestamp_granularities[] =
word + segment: words come with times; segments carry Whisper's own confidence, so the same made-up-text filter as
local listening drops words inside suspicious segments. If a server sends segments without words, each segment's
time is shared out by character length. Services take at most ~25 MB per file, so audio over 20 MB is sent as mp3
pieces of at most 10 minutes (cut in a quiet moment) and the times are moved back by each piece's start.
Default model: whisper-1 (OpenAI) / whisper-large-v3 (Groq). The audio leaves the PC: the UI says so first."""
import os
import tempfile
from pathlib import Path

from .. import config, media
from ..util import log
from . import _http
from ._http import HTTPError, guess_language, is_local_url, spread_words
from .base import STT, ProviderError
from .openai_compat import PRESETS, auth_headers, service_label
from .stt_local import _looks_hallucinated

MAX_BYTES = 20 * 1024 * 1024
PIECE_S = 600.0
NAMES = {"english": "en", "arabic": "ar", "french": "fr", "spanish": "es", "german": "de", "italian": "it",
         "portuguese": "pt", "turkish": "tr", "russian": "ru", "hindi": "hi", "urdu": "ur", "indonesian": "id",
         "japanese": "ja", "korean": "ko", "chinese": "zh", "dutch": "nl", "polish": "pl", "persian": "fa"}


def _code(name):
    n = str(name or "").strip().lower()
    return NAMES.get(n, n if 1 < len(n) <= 3 else "")


class OpenAISTT(STT):
    name = "openai"

    def __init__(self, opts=None):
        self.opts = dict(opts or {})
        self.base = str(self.opts.get("base_url") or PRESETS["OpenAI"]).strip().rstrip("/")
        self.key_name = str(self.opts.get("key_name") or "openai")
        groq = "api.groq.com" in self.base
        model = str(self.opts.get("model") or "").strip()
        if not model or (groq and model == "whisper-1"):       # the OpenAI default is not a Groq model
            model = "whisper-large-v3" if groq else "whisper-1"
        self.model = model
        self.local = is_local_url(self.base)
        self.label = service_label(self.base)

    def _piece(self, path, offset, language, prompt):
        fields = {"model": self.model, "response_format": "verbose_json",
                  "timestamp_granularities[]": ["word", "segment"], "temperature": "0"}
        if language:
            fields["language"] = language
        if prompt:
            fields["prompt"] = prompt[-800:]
        ext = Path(path).suffix.lower()
        mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}.get(ext, "application/octet-stream")
        try:
            r = _http.post_multipart(f"{self.base}/audio/transcriptions", fields,
                                     {"file": ("audio" + ext, Path(path).read_bytes(), mime)},
                                     auth_headers(self.base, self.key_name), timeout=900, label=self.label)
        except HTTPError as ex:
            if ex.status == 400 and any(k in ex.detail.lower() for k in ("verbose_json", "timestamp",
                                                                         "response_format")):
                raise ProviderError(f"{self.label}: the model {self.model} gives no word times, so captions cannot "
                                    f"be timed. Use whisper-1 (OpenAI) or whisper-large-v3 (Groq).") from None
            raise
        segs = [sg for sg in (r.get("segments") or []) if isinstance(sg, dict)]
        bad = [(float(sg.get("start", 0)), float(sg.get("end", 0))) for sg in segs
               if _looks_hallucinated(str(sg.get("text") or ""), float(sg.get("avg_logprob", 0) or 0),
                                      float(sg.get("no_speech_prob", 0) or 0))]
        words = []
        for w in r.get("words") or []:
            t = str(w.get("word") or "").strip()
            if not t:
                continue
            s, e = float(w.get("start", 0)), float(w.get("end", 0))
            if any(a <= (s + e) / 2 <= b for a, b in bad):
                continue
            words.append({"w": t, "s": round(s + offset, 2), "e": round(max(e, s) + offset, 2), "p": 1.0})
        if not words and segs:
            ok = [(sg.get("text"), float(sg.get("start", 0)) + offset, float(sg.get("end", 0)) + offset)
                  for sg in segs if (float(sg.get("start", 0)), float(sg.get("end", 0))) not in bad]
            words = spread_words(ok)
        if bad:
            log.info("%s listening: %d suspicious segment(s) dropped", self.label, len(bad))
        return words, _code(r.get("language"))

    def transcribe(self, wav_path, language=None, prompt="", on_progress=None):
        auth_headers(self.base, self.key_name)                 # no key: say so before cutting any audio
        lang = (language or "").split("-")[0].strip().lower()
        lang = "" if lang == "auto" else lang
        words, found = [], ""
        if os.path.getsize(wav_path) <= MAX_BYTES:
            words, found = self._piece(wav_path, 0.0, lang, prompt)
            if on_progress:
                on_progress(1.0)
        else:
            with tempfile.TemporaryDirectory(prefix="stt_", dir=config.cache_dir()) as tmp:
                pieces = media.split_audio(wav_path, tmp, PIECE_S, fmt="mp3")
                for i, (path, offset, _dur) in enumerate(pieces):
                    got, code = self._piece(path, offset, lang, prompt)
                    words += got
                    found = found or code
                    if on_progress:
                        on_progress((i + 1) / len(pieces))
        words.sort(key=lambda w: w["s"])
        lang = lang or found or guess_language(" ".join(w["w"] for w in words[:400]))
        return {"language": lang, "words": words}

    def test(self):
        """Free check: the key works and the service offers the model (when it lists its models)."""
        try:
            r = _http.get_json(f"{self.base}/models", auth_headers(self.base, self.key_name), label=self.label)
        except HTTPError as ex:
            if ex.status in (404, 405):                        # a server without a model list
                return f"Ready: {self.label} (could not list its models), model {self.model}"
            raise
        ids = {str(m.get("id")) for m in (r.get("data") or []) if isinstance(m, dict)}
        if ids and self.model not in ids:
            raise ProviderError(f"{self.label} has no model named {self.model} — check the model name in Settings.")
        where = "stays on your own network" if self.local else f"is sent to {self.label}"
        return f"Ready: {self.label} listening, model {self.model} (the audio {where})"

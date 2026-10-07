"""Gemini as the Listening AI (SPEC §4.2) — as in LoL Clips cloud.transcribe (facts in gemini.py's docstring).

Model gemini-3.5-transcribe (GROUND-TRUTH, Sept 2026): lists Arabic (Egypt) ar-EG and handles code-switching; word
timestamps work for up to 30 minutes per request, so the audio goes in pieces of at most 25 minutes (mp3, cut in a
quiet moment), each uploaded with the Files API, transcribed through the Interactions API and deleted right after.
Word times come from `word_info` annotations; if only segment times come back, each segment's time is shared out by
character length. Custom vocabulary cannot be combined with word timestamps (Google), so the prompt is not sent.
Sending audio to Google: the UI says so before the first use (SPEC §4.2)."""
import re
import tempfile

from .. import config, media
from ..util import log
from . import gemini
from ._http import guess_language, spread_words
from .base import STT, ProviderError

MODEL = "gemini-3.5-transcribe"
PIECE_S = 25 * 60
MIME = "audio/mp3"
# a few full codes (Google lists BCP-47 codes); anything else is passed as given
LANG_CODES = {"ar": "ar-EG", "en": "en-US", "fr": "fr-FR", "es": "es-ES", "de": "de-DE", "it": "it-IT",
              "pt": "pt-BR", "tr": "tr-TR", "ru": "ru-RU", "hi": "hi-IN", "ur": "ur-PK", "id": "id-ID",
              "ja": "ja-JP", "ko": "ko-KR", "nl": "nl-NL", "pl": "pl-PL", "fa": "fa-IR"}


def language_codes(language):
    """"ar" -> ["ar-EG", "en-US"] (people mix English words in); "en" -> ["en-US"]; None/"auto" -> [] (Gemini
    decides)."""
    lang = str(language or "").strip()
    if not lang or lang.lower() == "auto":
        return []
    code = lang if "-" in lang else LANG_CODES.get(lang.lower(), lang.lower())
    return [code] if code.lower().startswith("en") else [code, "en-US"]


def _secs(v):
    """"1.230s" / 1.23 / {"seconds": 1, "nanos": 230000000} -> 1.23 (None if unreadable)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        try:
            return float(v.get("seconds") or 0) + float(v.get("nanos") or 0) / 1e9
        except (TypeError, ValueError):
            return None
    m = re.match(r"\s*([\d.]+)\s*s?\s*$", str(v or ""))
    return float(m.group(1)) if m else None


def words_of(resp, offset=0.0):
    """Words with times from a Transcribe answer: `word_info` annotations, else segment-like annotations spread by
    character length. Times are moved by `offset` (where the piece starts in the whole audio)."""
    words, segs = [], []
    for step in resp.get("steps") or resp.get("outputs") or []:
        if not isinstance(step, dict):
            continue
        for c in step.get("content") or [step]:
            if not isinstance(c, dict):
                continue
            for a in c.get("annotations") or []:
                if not isinstance(a, dict):
                    continue
                text = str(a.get("text") or a.get("word") or "").strip()
                s = _secs(a.get("start_offset", a.get("start_time", a.get("start"))))
                e = _secs(a.get("end_offset", a.get("end_time", a.get("end"))))
                if not text or s is None or e is None:
                    continue
                if "word" in str(a.get("type") or "word"):
                    words.append({"w": text, "s": round(s + offset, 2), "e": round(max(e, s) + offset, 2), "p": 1.0})
                else:
                    segs.append((text, s + offset, e + offset))
    if words:
        return words
    return spread_words(segs)


class GeminiSTT(STT):
    name = "gemini"

    def __init__(self, opts=None):
        self.opts = dict(opts or {})
        self.model = str(self.opts.get("model") or MODEL).strip()

    def _piece(self, path, offset, codes):
        name, uri = gemini.upload(path, MIME)
        try:
            tc = {"mode": {"type": "verbatim", "timestamp_granularities": ["word"]}}
            if codes:
                tc["language_codes"] = codes
            resp = gemini.interact({"model": self.model, "input": [{"type": "audio", "uri": uri, "mime_type": MIME}],
                                    "generation_config": {"transcription_config": tc}}, timeout=900)
        finally:
            gemini.delete_file(name)
        words = words_of(resp, offset)
        if not words and gemini.output_text(resp).strip():
            raise ProviderError("Gemini wrote the text but gave no word times, so captions cannot be timed. Pick "
                                "another listening AI in Settings.")
        return words

    def transcribe(self, wav_path, language=None, prompt="", on_progress=None):
        gemini.headers()                                        # no key: say so before cutting any audio
        codes = language_codes(language)
        words = []
        with tempfile.TemporaryDirectory(prefix="gemini_", dir=config.cache_dir()) as tmp:
            pieces = media.split_audio(wav_path, tmp, PIECE_S, fmt="mp3")
            for i, (path, offset, _dur) in enumerate(pieces):
                words += self._piece(path, offset, codes)
                if on_progress:
                    on_progress((i + 1) / len(pieces))
        words.sort(key=lambda w: w["s"])
        lang = (language or "").split("-")[0].strip().lower()
        if lang in ("", "auto"):
            lang = guess_language(" ".join(w["w"] for w in words[:400]))
        log.info("Gemini listening: %d words in %d piece(s)", len(words), len(pieces))
        return {"language": lang, "words": words}

    def test(self):
        """Free check (no daily request used): the key works and the model exists."""
        gemini.model_info(self.model)
        return f"Ready: Gemini listening, model {self.model} (the audio is sent to Google)"

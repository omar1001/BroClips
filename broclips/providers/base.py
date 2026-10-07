"""The two AI roles (SPEC §4): Thinking (LLM: answers forced into a JSON schema) and Listening (speech-to-text with
word times). Every provider implements one of these classes; `providers.get_llm()` / `get_stt()` pick the one chosen
in Settings."""


class ProviderError(Exception):
    """A plain-English problem the UI can show as is (no key, server not running, quota, bad answer)."""


class LLM:
    name = "base"
    vision = False                  # can it look at pictures?
    local = False                   # runs on this PC (no data leaves it)

    def chat_json(self, system, user, schema, images=None, temperature=0.2, max_tokens=1500):
        """One question; the answer is a dict matching `schema` (objects, arrays with maxItems, enums, strings,
        integers, numbers, booleans), or None after retries. `images` = list of JPEG/PNG bytes (vision only)."""
        raise NotImplementedError

    def unload(self):
        """Free local memory after a job (no-op for cloud providers)."""

    def test(self):
        """A tiny real call. Returns a short success text; raises ProviderError with a plain-English reason."""
        raise NotImplementedError


class STT:
    name = "base"
    local = False

    def transcribe(self, wav_path, language=None, prompt="", on_progress=None):
        """-> {"language": "en", "words": [{"w": "hello", "s": 0.52, "e": 0.81, "p": 0.97}, ...]}
        Times in seconds (plain floats), sorted. on_progress(fraction 0..1) is optional."""
        raise NotImplementedError

    def unload(self):
        """Free the model (local only)."""

    def test(self):
        raise NotImplementedError

"""Fake AI for tests and demos (SPEC §4.1 "fake"): deterministic answers, no network, no GPU, nothing leaves the PC.

FakeLLM walks the JSON schema it is given and returns the smallest answer of that shape:
object -> its required properties (all properties when the schema lists none), array -> `minItems` items (or 1),
enum -> the first value, string -> "text", integer -> 1, number -> 1.0, boolean -> false (numbers are kept inside
`minimum`/`maximum` so the answer stays valid). Tests can queue exact answers with opts["answers"] and read
every question back from `.calls`.

FakeSTT returns the words from a sidecar file next to the audio, if there is one — `<wav>.fake.json`
(e.g. audio16k.wav.fake.json; audio16k.fake.json works too) holding {"language": .., "words": [..]} or just the
word list — else a few dummy words with times."""
import json
from pathlib import Path

from .base import LLM, STT


def sample(schema):
    """The smallest value that matches a (simple) JSON schema."""
    if not isinstance(schema, dict):
        return None
    if schema.get("enum"):
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    t = schema.get("type")
    if isinstance(t, list):                       # e.g. ["string", "null"]
        t = next((x for x in t if x != "null"), "null")
    if t is None:
        t = "object" if "properties" in schema else "array" if "items" in schema else "string"
    if t == "object":
        props = schema.get("properties") or {}
        keys = schema["required"] if isinstance(schema.get("required"), list) else list(props)
        return {k: sample(props.get(k) or {}) for k in keys}
    if t == "array":
        n = schema.get("minItems")
        n = 1 if n is None else int(n)
        if schema.get("maxItems") is not None:
            n = min(n, int(schema["maxItems"]))
        return [sample(schema.get("items") or {}) for _ in range(n)]
    if t in ("integer", "number"):
        v = 1
        if schema.get("minimum") is not None:
            v = max(v, schema["minimum"])
        if schema.get("maximum") is not None:
            v = min(v, schema["maximum"])
        return int(v) if t == "integer" else float(v)
    if t == "boolean":
        return False
    if t == "null":
        return None
    return "text"


class FakeLLM(LLM):
    name = "fake"
    local = True

    def __init__(self, opts=None):
        self.opts = dict(opts or {})
        self.vision = bool(self.opts.get("vision", False))
        self.answers = list(self.opts.get("answers") or [])     # canned answers, used first (tests)
        self.calls = []                                           # every question, for tests to inspect

    def chat_json(self, system, user, schema, images=None, temperature=0.2, max_tokens=1500):
        self.calls.append({"system": system, "user": user, "schema": schema, "images": len(images or [])})
        if self.answers:
            return self.answers.pop(0)
        return sample(schema)

    def unload(self):
        pass

    def test(self):
        return "Fake AI ready"


DUMMY_WORDS = "hello and welcome this is a short test of bro clips thank you for watching".split()


class FakeSTT(STT):
    name = "fake"
    local = True

    def __init__(self, opts=None):
        self.opts = dict(opts or {})

    @staticmethod
    def _sidecar(wav_path):
        p = Path(wav_path)
        for side in (Path(str(p) + ".fake.json"), p.with_suffix(".fake.json")):
            if side.is_file():
                return json.loads(side.read_text(encoding="utf-8"))
        return None

    def transcribe(self, wav_path, language=None, prompt="", on_progress=None):
        data = self._sidecar(wav_path)
        if isinstance(data, list):
            data = {"words": data}
        if data is not None:
            words = [{"w": str(w["w"]), "s": float(w["s"]), "e": float(w["e"]), "p": float(w.get("p", 1.0))}
                     for w in data.get("words") or []]
            lang = data.get("language") or language or "en"
        else:
            words = [{"w": w, "s": round(0.5 + i * 0.6, 2), "e": round(0.5 + i * 0.6 + 0.5, 2), "p": 0.99}
                     for i, w in enumerate(DUMMY_WORDS)]
            lang = language or "en"
        words.sort(key=lambda w: w["s"])
        if on_progress:
            on_progress(1.0)
        return {"language": lang, "words": words}

    def unload(self):
        pass

    def test(self):
        return "Fake listening ready"

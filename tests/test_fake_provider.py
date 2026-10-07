"""The fake AI (SPEC §4.1): schema-shaped minimal answers, canned answers for tests, sidecar words for the STT."""
import json

from broclips.providers import fake, get_llm, get_stt


def test_sample_walks_the_schema():
    item = {"type": "object", "required": ["start", "score", "type", "keep", "title", "note"], "properties": {
        "start": {"type": "integer", "minimum": 3}, "score": {"type": "number", "maximum": 0.5},
        "type": {"enum": ["funny", "fail"]}, "keep": {"type": "boolean"}, "title": {"type": "string"},
        "note": {"type": ["string", "null"]}, "unused": {"type": "string"}}}
    schema = {"type": "object", "required": ["moments"],
              "properties": {"moments": {"type": "array", "minItems": 2, "maxItems": 5, "items": item},
                             "extra": {"type": "string"}}}
    one = {"start": 3, "score": 0.5, "type": "funny", "keep": False, "title": "text", "note": "text"}
    assert fake.sample(schema) == {"moments": [one, one]}
    assert fake.sample({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "array"}}}) == \
        {"a": 1, "b": ["text"]}                                          # no "required": every property
    assert fake.sample({"type": "array", "items": {"type": "number"}}) == [1.0]


def test_fake_llm():
    llm = get_llm(provider="fake")
    assert isinstance(llm, fake.FakeLLM) and llm.local and llm.test() == "Fake AI ready"
    assert llm.chat_json("sys", "pick", {"type": "object", "required": ["ok"],
                                         "properties": {"ok": {"type": "boolean"}}}) == {"ok": False}
    assert llm.calls[0]["user"] == "pick"
    canned = fake.FakeLLM({"answers": [{"best": 2}], "vision": True})
    assert canned.vision is True
    assert canned.chat_json("s", "u", {"type": "object"}, images=[b"jpg"]) == {"best": 2}
    assert canned.calls[0]["images"] == 1
    llm.unload()


def test_fake_stt(tmp_path):
    stt = get_stt(provider="fake")
    assert stt.test() == "Fake listening ready"
    wav = tmp_path / "audio16k.wav"
    seen = []
    out = stt.transcribe(str(wav), language="ar", on_progress=seen.append)
    assert out["language"] == "ar" and len(out["words"]) > 5 and seen == [1.0]
    assert all(isinstance(w["s"], float) and w["e"] > w["s"] for w in out["words"])
    words = [{"w": "ثانية", "s": 2.0, "e": 2.4}, {"w": "أهلا", "s": 1.0, "e": 1.5, "p": 0.9}]
    (tmp_path / "audio16k.wav.fake.json").write_text(json.dumps({"language": "ar", "words": words}), encoding="utf-8")
    out = stt.transcribe(str(wav))
    assert [w["w"] for w in out["words"]] == ["أهلا", "ثانية"] and out["words"][1]["p"] == 1.0
    other = tmp_path / "b.wav"
    (tmp_path / "b.fake.json").write_text(json.dumps(words), encoding="utf-8")         # a plain list works too
    assert len(stt.transcribe(str(other))["words"]) == 2

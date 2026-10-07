"""AI providers with mocked HTTP (SPEC §11): request shape, answer parsing, fallbacks and plain-English errors.
No network, no GPU: urllib.request.urlopen is replaced by FakeHTTP, faster-whisper by a fake module."""
import email.message
import io
import json
import logging
import socket
import sys
import types
import urllib.error
import urllib.request
import wave

import numpy as np
import pytest

from broclips import config
from broclips.providers import _http, get_llm, get_stt
from broclips.providers import anthropic as anth
from broclips.providers import gemini, ollama, openai_compat, stt_gemini, stt_local, stt_openai
from broclips.providers.base import ProviderError

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "title": {"type": "string"},
    "score": {"type": "integer", "minimum": 1, "maximum": 10},
    "kind": {"type": "string", "enum": ["funny", "skill"]},
    "ids": {"type": "array", "maxItems": 2, "items": {"type": "string"}}},
    "required": ["title", "score"]}
GOOD = {"title": "Big win", "score": 8, "kind": "funny", "ids": ["L001"]}
JPEG = b"\xff\xd8\xff\xe0fakejpeg"


# ---------- a fake network ----------
def _msg(headers):
    m = email.message.Message()
    for k, v in (headers or {}).items():
        m[k] = v
    return m


class _Resp:
    def __init__(self, status, body, headers):
        self.status, self._body, self.headers = status, body, _msg(headers)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeHTTP:
    """Stands in for urllib.request.urlopen. `on(method, url_part, *answers)`: each request goes to the first route
    whose method matches and whose text is in the URL; its answers are used in order, the last one repeats.
    An answer is a dict/str body (200), a (status, body[, headers]) tuple, or an exception to raise."""

    def __init__(self, monkeypatch):
        self.routes, self.calls, self.sleeps = [], [], []
        monkeypatch.setattr(urllib.request, "urlopen", self)
        monkeypatch.setattr(_http, "_sleep", self.sleeps.append)

    def on(self, method, part, *answers):
        self.routes.append([method, part, list(answers)])
        return self

    def __call__(self, req, timeout=None):
        call = {"method": req.get_method(), "url": req.full_url, "timeout": timeout, "data": req.data,
                "headers": {k.lower(): v for k, v in req.header_items()}}
        try:
            call["json"] = json.loads(req.data)
        except (TypeError, ValueError):
            call["json"] = None
        self.calls.append(call)
        for method, part, answers in self.routes:
            if method == call["method"] and part in req.full_url:
                a = answers.pop(0) if len(answers) > 1 else answers[0]
                break
        else:
            raise AssertionError(f"unexpected request {call['method']} {req.full_url}")
        if isinstance(a, BaseException):
            raise a
        status, body, headers = (a + ({},))[:3] if isinstance(a, tuple) else (200, a, {})
        raw = body if isinstance(body, bytes) else (json.dumps(body) if not isinstance(body, str) else body).encode()
        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, "error", _msg(headers), io.BytesIO(raw))
        return _Resp(status, raw, headers)

    def find(self, part):
        return [c for c in self.calls if part in c["url"]]


@pytest.fixture(autouse=True)
def own_data_dir(tmp_path, monkeypatch):
    """These tests write API keys: never into the real data folder, even if conftest.py is missing."""
    monkeypatch.setenv("BROCLIPS_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(config, "POINTER", tmp_path / "no_data_location.txt")


@pytest.fixture
def net(monkeypatch):
    return FakeHTTP(monkeypatch)


def set_keys(**keys):
    for k, v in keys.items():
        config.set_secret(k, v)


# ---------- _http: errors, retries, keys ----------
def test_post_json_sends_utf8_json(net):
    net.on("POST", "/x", {"ok": 1})
    assert _http.post_json("https://api.example.com/x", {"t": "مرحبا"}, {"Authorization": "Bearer k"}) == {"ok": 1}
    c = net.calls[0]
    assert c["headers"]["content-type"] == "application/json"
    assert "مرحبا" in c["data"].decode("utf-8") and c["json"] == {"t": "مرحبا"}


@pytest.mark.parametrize("status,text", [(401, _http.KEY_REFUSED), (403, _http.KEY_REFUSED),
                                         (429, _http.RATE_LIMIT), (500, _http.SERVER_TROUBLE),
                                         (503, _http.SERVER_TROUBLE)])
def test_http_errors_are_plain_english(net, status, text):
    net.on("POST", "/x", (status, {"error": {"message": "nope"}}))
    with pytest.raises(_http.HTTPError) as ei:
        _http.post_json("https://api.example.com/x", {}, label="Gemini")
    assert text in str(ei.value) and str(ei.value).startswith("Gemini")
    assert ei.value.status == status
    retried = status == 429 or status >= 500
    assert len(net.calls) == (3 if retried else 1) and len(net.sleeps) == (2 if retried else 0)


def test_server_error_keeps_ollamas_reason(net):
    net.on("POST", "/api/chat", (500, {"error": "model requires more system memory (9.1 GiB) than is available"}))
    with pytest.raises(_http.HTTPError, match="try again \\(model requires more system memory"):
        _http.post_json("http://127.0.0.1:11434/api/chat", {}, retries=0, label="Ollama")


def test_400_keeps_the_services_words(net):
    net.on("POST", "/x", (400, {"error": {"message": "Unknown parameter 'temperature'", "param": "temperature"}}))
    with pytest.raises(_http.HTTPError) as ei:
        _http.post_json("https://api.example.com/x", {})
    assert ei.value.status == 400 and "temperature" in ei.value.detail and "api.example.com" in str(ei.value)


def test_connection_problems(net):
    net.on("GET", "refused", urllib.error.URLError(ConnectionRefusedError()))
    net.on("GET", "nodns", urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed")))
    with pytest.raises(_http.HTTPError, match=r"Cannot reach 127\.0\.0\.1:9"):
        _http.get_json("http://127.0.0.1:9/refused")
    with pytest.raises(_http.HTTPError, match=r"Cannot reach nodns\.example \(no internet\?\)"):
        _http.get_json("https://nodns.example/nodns")
    assert not net.sleeps                                   # nothing to wait for


def test_timeout_is_retried_then_succeeds(net):
    net.on("GET", "/slow", TimeoutError(), {"fine": True})
    assert _http.get_json("https://api.example.com/slow") == {"fine": True}
    assert net.sleeps == [3.0]


def test_retry_after_header_is_respected(net):
    net.on("GET", "/busy", (429, "slow down", {"Retry-After": "7"}), {"ok": True})
    assert _http.get_json("https://api.example.com/busy") == {"ok": True}
    assert net.sleeps == [7.0]


def test_keys_never_reach_messages_or_logs(net, caplog):
    key = "sk-SECRETSECRETSECRETSECRET1234"
    net.on("POST", "/x", (401, {"error": {"message": f"Incorrect API key provided: {key}"}}))
    caplog.set_level(logging.DEBUG, logger="broclips")
    with pytest.raises(_http.HTTPError) as ei:
        _http.post_json("https://api.example.com/x?key=" + key, {"prompt": "hi"}, {"Authorization": "Bearer " + key})
    assert key not in str(ei.value) and key not in ei.value.detail
    assert key not in caplog.text and "?key=" not in caplog.text


def test_not_json_answer(net):
    net.on("GET", "/html", "<html>proxy login</html>")
    with pytest.raises(_http.HTTPError, match="not JSON"):
        _http.get_json("https://api.example.com/html")


def test_multipart_body(net):
    net.on("POST", "/up", {"ok": True})
    _http.post_multipart("https://api.example.com/up", {"model": "m", "t[]": ["word", "segment"]},
                         {"file": ("a.wav", b"RIFFDATA", "audio/wav")})
    c = net.calls[0]
    assert c["headers"]["content-type"].startswith("multipart/form-data; boundary=")
    body = c["data"]
    assert body.count(b'name="t[]"') == 2 and b'filename="a.wav"' in body and b"RIFFDATA" in body


# ---------- _http: helpers ----------
@pytest.mark.parametrize("text,want", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 2}\n```', {"a": 2}),
    ('Sure! Here it is: {"a": {"b": "a } in a string"}} hope it helps {"x": 1}', {"a": {"b": "a } in a string"}}),
    ('no json here', None), ("[1, 2]", None), ("", None), ('{"broken": ', None)])
def test_extract_json(text, want):
    assert _http.extract_json(text) == want


def test_fit_schema():
    fit = _http.fit_schema
    assert fit(dict(GOOD, ids=["a", "b", "c"]), SCHEMA)["ids"] == ["a", "b"]          # maxItems
    assert fit({"title": "x", "score": "7"}, SCHEMA)["score"] == 7                      # text -> number
    assert fit({"title": "x", "score": 7.6}, SCHEMA)["score"] == 8
    assert fit({"title": "x", "score": 7, "kind": "Funny"}, SCHEMA)["kind"] == "funny"  # enum case
    assert "kind" not in fit({"title": "x", "score": 7, "kind": "sad"}, SCHEMA)        # optional, wrong: dropped
    assert fit({"title": "x"}, SCHEMA) is None                                          # required missing
    assert fit({"title": "x", "score": True}, SCHEMA) is None
    assert fit(None, SCHEMA) is None and fit([GOOD], SCHEMA) is None
    nested = {"type": "object", "required": ["m"], "properties": {"m": {"type": "array", "items": {
        "type": "object", "required": ["s"], "properties": {"s": {"type": "number"}}}}}}
    assert fit({"m": [{"s": 1}, {"x": 2}, {"s": "2.5"}]}, nested) == {"m": [{"s": 1}, {"s": 2.5}]}


def test_to_openapi_strips_what_gemini_refuses():
    s = {"$schema": "x", "type": "object", "additionalProperties": False, "required": ["a", "ghost"],
         "properties": {"a": {"type": ["string", "null"], "default": "", "pattern": "^x"},
                        "b": {"const": "fixed", "type": "string"},
                        "c": {"type": "integer", "enum": [1, 2]},
                        "d": {"type": "array", "maxItems": 3, "items": {"type": "object", "additionalProperties": False,
                                                                        "properties": {"e": {"type": "boolean"}}}}}}
    o = _http.to_openapi(s)
    assert "$schema" not in o and "additionalProperties" not in o and o["required"] == ["a"]
    assert o["properties"]["a"] == {"type": "string", "nullable": True}
    assert o["properties"]["b"]["enum"] == ["fixed"] and "enum" not in o["properties"]["c"]
    assert o["properties"]["d"]["maxItems"] == 3 and "additionalProperties" not in o["properties"]["d"]["items"]
    assert s["additionalProperties"] is False                                           # the input is untouched


def test_spread_words_by_character_length():
    w = _http.spread_words([("hi everyone", 10.0, 11.2), ("", 12, 13), ("x", 5, 5)])
    assert [x["w"] for x in w] == ["hi", "everyone"]
    assert w[0]["s"] == 10.0 and w[1]["e"] == 11.2 and w[0]["e"] == pytest.approx(10.3, abs=0.01)


def test_drop_rejected():
    body = {"generation_config": {"temperature": 0.2, "thinking_level": "low"}}
    err = _http.HTTPError("x", 400, "Invalid value for thinking_level")
    assert _http.drop_rejected(err, body, [("generation_config", "temperature"),
                                           ("generation_config", "thinking_level")]) == "thinking_level"
    assert body == {"generation_config": {"temperature": 0.2}}
    assert _http.drop_rejected(_http.HTTPError("x", 400, "bad schema"), body,
                               [("generation_config", "temperature")]) == ""


def test_small_helpers():
    assert _http.is_local_url("http://127.0.0.1:1234/v1") and _http.is_local_url("http://localhost:8080")
    assert _http.is_local_url("http://192.168.1.20:1234/v1")
    assert not _http.is_local_url("https://api.openai.com/v1")
    assert _http.image_mime(b"\x89PNG\r\n\x1a\nxx") == "image/png" and _http.image_mime(JPEG) == "image/jpeg"
    assert _http.guess_language("يعني احنا كنا بنلعب ال game") == "ar" and _http.guess_language("hello") == "en"


# ---------- Ollama ----------
def ollama_llm(**opts):
    return ollama.OllamaLLM({"url": "http://127.0.0.1:11434", "model": "gemma3:12b", "num_ctx": 16384, **opts})


def test_ollama_chat_request_shape(net):
    net.on("GET", "/api/version", {"version": "0.12.0"})
    net.on("POST", "/api/generate", {"done": True})
    net.on("POST", "/api/show", {"capabilities": ["completion", "vision"]})
    net.on("POST", "/api/chat", {"message": {"content": json.dumps(dict(GOOD, ids=["a", "b", "c"]))}})
    llm = ollama_llm()
    assert llm.chat_json("SYS", "USER", SCHEMA, images=[JPEG], temperature=0.3, max_tokens=900) == \
        dict(GOOD, ids=["a", "b"])
    warm = net.find("/api/generate")[0]["json"]
    assert warm == {"model": "gemma3:12b", "keep_alive": "10m", "options": {"num_ctx": 16384}}
    chat = net.find("/api/chat")[0]
    b = chat["json"]
    assert b["format"] == SCHEMA and b["think"] is False and b["stream"] is False and b["keep_alive"] == "10m"
    assert b["options"] == {"num_ctx": 16384, "temperature": 0.3, "num_predict": 900}
    assert b["messages"][0] == {"role": "system", "content": "SYS"}
    assert b["messages"][1]["images"] == [_http.b64(JPEG)]
    assert chat["timeout"] == 1200 and llm.vision is True


def test_ollama_retries_bad_json_with_a_new_seed(net):
    net.on("GET", "/api/version", {"version": "0.12.0"})
    net.on("POST", "/api/generate", {})
    net.on("POST", "/api/chat", {"message": {"content": "{not json"}}, {"message": {"content": json.dumps(GOOD)}})
    assert ollama_llm().chat_json("s", "u", SCHEMA) == GOOD
    chats = net.find("/api/chat")
    assert "seed" not in chats[0]["json"]["options"] and chats[1]["json"]["options"]["seed"] == 1000


def test_ollama_gives_up_with_none(net):
    net.on("GET", "/api/version", {"version": "0.12.0"})
    net.on("POST", "/api/generate", {})
    net.on("POST", "/api/chat", {"message": {"content": "{}"}})
    assert ollama_llm().chat_json("s", "u", SCHEMA) is None
    assert len(net.find("/api/chat")) == 3                  # 1 + 2 retries (SPEC §4.1)


def test_ollama_missing_model_and_not_running(net):
    net.on("GET", "/api/version", {"version": "0.12.0"})
    net.on("POST", "/api/generate", (404, {"error": "model 'gemma3:12b' not found"}))
    with pytest.raises(ProviderError, match=r"ollama pull gemma3:12b"):
        ollama_llm().chat_json("s", "u", SCHEMA)
    net.routes.clear()
    net.on("GET", "/api/version", urllib.error.URLError(ConnectionRefusedError()))
    with pytest.raises(ProviderError, match="Ollama is not running"):
        ollama_llm().chat_json("s", "u", SCHEMA)


def test_ollama_private_server_is_started_without_a_window(net, monkeypatch, tmp_path):
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    started = []
    monkeypatch.setattr(ollama.subprocess, "Popen", lambda cmd, **kw: started.append((cmd, kw)))
    net.on("GET", "/api/version", urllib.error.URLError(ConnectionRefusedError()),
           urllib.error.URLError(ConnectionRefusedError()), {"version": "0.12.0"})
    llm = ollama_llm(manage={"enabled": True, "exe": str(exe), "models_dir": "C:\\models\\ollama", "port": 11435})
    assert llm.url == "http://127.0.0.1:11435"
    llm.ensure_server()
    cmd, kw = started[0]
    assert cmd == [str(exe), "serve"] and kw["creationflags"] == ollama.NO_WINDOW
    env = kw["env"]
    assert env["OLLAMA_HOST"] == "127.0.0.1:11435" and env["OLLAMA_MODELS"] == "C:\\models\\ollama"
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "1" and env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_FLASH_ATTENTION"] == "1" and env["OLLAMA_NOPRUNE"] == "1"
    assert env["OLLAMA_LOAD_TIMEOUT"] == "20m"
    assert all(c["url"].startswith("http://127.0.0.1:11435") for c in net.calls)


def test_ollama_unload_frees_every_loaded_model(net):
    net.on("GET", "/api/ps", {"models": [{"name": "gemma3:12b"}, {"name": "qwen3:8b"}]})
    net.on("POST", "/api/generate", {})
    ollama_llm().unload()
    assert [c["json"] for c in net.find("/api/generate")] == [{"model": "gemma3:12b", "keep_alive": 0},
                                                             {"model": "qwen3:8b", "keep_alive": 0}]
    net.routes.clear()
    net.on("GET", "/api/ps", urllib.error.URLError(ConnectionRefusedError()))
    ollama_llm().unload()                                   # not running: nothing to free, no error


def test_ollama_test_and_models(net):
    net.on("GET", "/api/version", {"version": "0.12.0"})
    net.on("GET", "/api/tags", {"models": [{"name": "gemma3:12b"}, {"name": "llava:latest"}]})
    net.on("POST", "/api/show", {"capabilities": ["completion"]})
    assert ollama_llm().test() == "Ready: Ollama 0.12.0, model gemma3:12b"
    assert ollama_llm(model="llava").list_models() == ["gemma3:12b", "llava:latest"]
    assert "llava" in ollama_llm(model="llava").test()        # "llava" means "llava:latest"
    with pytest.raises(ProviderError, match="ollama pull qwen3:8b"):
        ollama_llm(model="qwen3:8b").test()


# ---------- Gemini (thinking) ----------
def gemini_answer(obj, status="completed"):
    return {"id": "int_1", "status": status, "steps": [
        {"type": "thought", "content": [{"type": "text", "text": "thinking about {it}"}]},
        {"type": "model_output", "content": [{"type": "text", "text": json.dumps(obj)}]}]}


def test_gemini_request_shape_and_answer(net):
    set_keys(gemini="AIzaFAKEKEY")
    net.on("POST", "/v1beta/interactions", gemini_answer(GOOD))
    llm = get_llm(provider="gemini")
    assert llm.model == gemini.MODEL == "gemini-3.8-flash"
    assert llm.chat_json("SYS", "USER", SCHEMA, images=[JPEG], max_tokens=1500) == GOOD
    c = net.calls[0]
    assert c["url"] == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert c["headers"]["x-goog-api-key"] == "AIzaFAKEKEY" and "key=" not in c["url"]
    b = c["json"]
    assert b["model"] == "gemini-3.8-flash" and b["system_instruction"] == "SYS"
    assert b["input"] == [{"type": "text", "text": "USER"}]          # pictures are off by default (SPEC §4.3)
    assert b["response_format"]["mime_type"] == "application/json"
    assert b["response_format"]["schema"] == _http.to_openapi(SCHEMA)
    assert "additionalProperties" not in b["response_format"]["schema"]
    assert b["generation_config"] == {"temperature": 0.2, "max_output_tokens": 6000, "thinking_level": "low"}


def test_gemini_sends_pictures_only_when_allowed(net):
    set_keys(gemini="k")
    net.on("POST", "/v1beta/interactions", gemini_answer(GOOD))
    gemini.GeminiLLM({"allow_images": True}).chat_json("s", "u", SCHEMA, images=[JPEG])
    assert net.calls[0]["json"]["input"][1] == {"type": "image", "data": _http.b64(JPEG), "mime_type": "image/jpeg"}


def test_gemini_drops_a_refused_field_then_steps_down(net):
    set_keys(gemini="k")
    net.on("POST", "/v1beta/interactions",
           (400, {"error": {"message": "temperature is not supported for this model"}}),
           (400, {"error": {"message": "Invalid JSON payload: unknown name 'schema'"}}),
           gemini_answer(GOOD))
    llm = gemini.GeminiLLM({})
    assert llm.chat_json("SYS", "u", SCHEMA) == GOOD
    second, third = net.calls[1]["json"], net.calls[2]["json"]
    assert "temperature" not in second["generation_config"]
    assert third["response_format"] == {"type": "text", "mime_type": "application/json"}   # no schema now
    assert "JSON schema" in third["system_instruction"] and '"title"' in third["system_instruction"]
    llm.chat_json("SYS", "u", SCHEMA)                                # remembered: no new 400s
    assert len(net.calls) == 4 and "temperature" not in net.calls[3]["json"]["generation_config"]


def test_gemini_waits_for_a_long_job_and_grows_the_budget(net):
    set_keys(gemini="k")
    net.on("POST", "/v1beta/interactions", {"id": "int_7", "status": "in_progress"}, gemini_answer(GOOD))
    net.on("GET", "/v1beta/interactions/int_7", {"id": "int_7", "status": "incomplete", "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": '{"title": "cut off'}]}]})
    assert gemini.GeminiLLM({}).chat_json("s", "u", SCHEMA, max_tokens=1000) == GOOD
    posts = [c for c in net.calls if c["method"] == "POST"]
    assert posts[1]["json"]["generation_config"]["max_output_tokens"] == 8192    # doubled after "incomplete"


def test_gemini_errors(net):
    with pytest.raises(ProviderError, match="No Gemini API key"):
        gemini.GeminiLLM({}).chat_json("s", "u", SCHEMA)
    assert not net.calls
    set_keys(gemini="k")
    net.on("POST", "/v1beta/interactions", (429, {"error": {"message": "Quota exceeded"}}))
    with pytest.raises(ProviderError, match="Rate limit or quota reached"):
        gemini.GeminiLLM({}).chat_json("s", "u", SCHEMA)


def test_gemini_output_text_shapes():
    assert gemini.output_text({"output_text": "x"}) == "x"
    assert gemini.output_text({"outputs": [{"type": "thought", "text": "hm"}, {"type": "text", "text": "y"}]}) == "y"
    assert gemini.output_text({"steps": [{"type": "user_input", "content": [{"type": "text", "text": "q"}]}]}) == ""


# ---------- Gemini (listening) ----------
def tiny_wav(path, seconds=1.0, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(int(seconds * sr), np.int16).tobytes())
    return path


def test_gemini_stt_upload_transcribe_delete(net, tmp_path, monkeypatch):
    set_keys(gemini="AIzaK")
    wav = tiny_wav(tmp_path / "a.wav")
    piece = tmp_path / "piece_000.mp3"
    piece.write_bytes(b"ID3fake")
    monkeypatch.setattr(stt_gemini.media, "split_audio",
                        lambda w, out, piece_s, fmt: [(str(piece), 0.0, 1500.0), (str(piece), 1500.0, 30.0)])
    up = "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=xyz"
    net.on("POST", "upload_id=xyz", {"file": {"name": "files/abc", "uri": "https://g/files/abc", "state": "ACTIVE"}})
    net.on("POST", "/upload/v1beta/files", (200, b"", {"x-goog-upload-url": up}))
    net.on("DELETE", "/v1beta/files/abc", {})
    words = [{"type": "word_info", "text": "يا", "start_offset": "0.5s", "end_offset": "0.9s"},
             {"type": "word_info", "text": "جماعة", "start_offset": {"seconds": 1, "nanos": 0}, "end_offset": 1.6}]
    net.on("POST", "/v1beta/interactions", {"steps": [{"type": "model_output", "content": [
        {"type": "text", "text": "يا جماعة", "annotations": words}]}]})
    progress = []
    out = stt_gemini.GeminiSTT({}).transcribe(str(wav), language="ar", on_progress=progress.append)
    assert [w["w"] for w in out["words"]] == ["يا", "جماعة", "يا", "جماعة"]
    assert out["words"][0] == {"w": "يا", "s": 0.5, "e": 0.9, "p": 1.0}
    assert out["words"][2]["s"] == 1500.5 and out["language"] == "ar" and progress == [0.5, 1.0]
    start = net.find("/upload/v1beta/files")[0]
    assert start["headers"]["x-goog-upload-protocol"] == "resumable"
    assert start["headers"]["x-goog-upload-header-content-type"] == "audio/mp3"
    final = net.find("upload_id=xyz")[0]
    assert "x-goog-api-key" not in final["headers"] and final["headers"]["x-goog-upload-command"] == \
        "upload, finalize"
    body = net.find("/v1beta/interactions")[0]["json"]
    assert body["model"] == "gemini-3.5-transcribe"
    assert body["input"] == [{"type": "audio", "uri": "https://g/files/abc", "mime_type": "audio/mp3"}]
    tc = body["generation_config"]["transcription_config"]
    assert tc["language_codes"] == ["ar-EG", "en-US"] and tc["mode"]["timestamp_granularities"] == ["word"]
    assert len([c for c in net.calls if c["method"] == "DELETE"]) == 2           # every upload deleted


def test_gemini_stt_segment_times_are_spread():
    resp = {"steps": [{"type": "model_output", "content": [{"text": "x", "annotations": [
        {"type": "segment", "text": "hello there", "start_offset": "1s", "end_offset": "2.2s"}]}]}]}
    w = stt_gemini.words_of(resp, offset=10)
    assert [x["w"] for x in w] == ["hello", "there"] and w[0]["s"] == 11.0 and w[-1]["e"] == 12.2


def test_gemini_language_codes():
    assert stt_gemini.language_codes("ar") == ["ar-EG", "en-US"]
    assert stt_gemini.language_codes("en") == ["en-US"]
    assert stt_gemini.language_codes("ar-SA") == ["ar-SA", "en-US"]
    assert stt_gemini.language_codes("auto") == [] and stt_gemini.language_codes(None) == []


# ---------- OpenAI-compatible (thinking) ----------
def chat_answer(text, finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def test_openai_json_schema_request(net):
    set_keys(openai="sk-test")
    net.on("POST", "/chat/completions", chat_answer(json.dumps(GOOD)))
    llm = get_llm(provider="openai")
    assert llm.chat_json("SYS", "USER", SCHEMA, max_tokens=700) == GOOD
    c = net.calls[0]
    assert c["url"] == "https://api.openai.com/v1/chat/completions"
    assert c["headers"]["authorization"] == "Bearer sk-test"
    b = c["json"]
    assert b["model"] == "gpt-4o-mini" and b["max_tokens"] == 700 and b["temperature"] == 0.2
    assert b["response_format"] == {"type": "json_schema",
                                    "json_schema": {"name": "answer", "schema": SCHEMA, "strict": False}}
    assert b["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]


def test_openai_falls_back_json_object_then_plain(net):
    set_keys(openai="sk")
    net.on("POST", "/chat/completions",
           (400, {"error": {"message": "response_format json_schema is not supported"}}),
           (400, {"error": {"message": "response_format json_object is not supported"}}),
           chat_answer('Here you go:\n```json\n' + json.dumps(GOOD) + '\n```'))
    llm = openai_compat.OpenAILLM({})
    assert llm.chat_json("SYS", "u", SCHEMA) == GOOD
    b1, b2 = net.calls[1]["json"], net.calls[2]["json"]
    assert b1["response_format"] == {"type": "json_object"} and "JSON" in b1["messages"][0]["content"]
    assert "response_format" not in b2 and '"title"' in b2["messages"][0]["content"]
    llm.chat_json("SYS", "u", SCHEMA)
    assert len(net.calls) == 4 and "response_format" not in net.calls[3]["json"]     # remembered


def test_openai_repairs_temperature_and_max_tokens(net):
    set_keys(openai="sk")
    net.on("POST", "/chat/completions",
           (400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. "
                                       "Use 'max_completion_tokens' instead.", "param": "max_tokens"}}),
           (400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.2 with this model.",
                            "param": "temperature"}}),
           chat_answer(json.dumps(GOOD)))
    assert openai_compat.OpenAILLM({"model": "o4-mini"}).chat_json("s", "u", SCHEMA, max_tokens=500) == GOOD
    b = net.calls[2]["json"]
    assert b["max_completion_tokens"] == 500 and "max_tokens" not in b and "temperature" not in b
    assert b["response_format"]["type"] == "json_schema"                       # no step down was needed


def test_openai_keys_local_servers_and_pictures(net):
    with pytest.raises(ProviderError, match="No API key for OpenAI"):
        openai_compat.OpenAILLM({}).chat_json("s", "u", SCHEMA)
    with pytest.raises(ProviderError, match="No API key for Groq"):
        openai_compat.OpenAILLM({"base_url": openai_compat.PRESETS["Groq"], "key_name": "groq"}).chat_json(
            "s", "u", SCHEMA)
    net.on("POST", "/chat/completions", chat_answer(json.dumps(GOOD)))
    lm = openai_compat.OpenAILLM({"base_url": "http://127.0.0.1:1234/v1", "model": "qwen", "key_name": "lmstudio"})
    assert lm.chat_json("s", "u", SCHEMA, images=[JPEG]) == GOOD                 # no key needed on this PC
    c = net.calls[0]
    assert "authorization" not in c["headers"]
    parts = c["json"]["messages"][1]["content"]
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _http.b64(JPEG)}}
    set_keys(openai="sk")
    openai_compat.OpenAILLM({}).chat_json("s", "u", SCHEMA, images=[JPEG])   # cloud + pictures off: text only
    assert net.calls[1]["json"]["messages"][1]["content"] == "u"


def test_openai_truncated_answer_gets_more_room(net):
    set_keys(openai="sk")
    net.on("POST", "/chat/completions", chat_answer('{"title": "x", "sco', "length"), chat_answer(json.dumps(GOOD)))
    assert openai_compat.OpenAILLM({}).chat_json("s", "u", SCHEMA, max_tokens=300) == GOOD
    assert net.calls[1]["json"]["max_tokens"] == 600


def test_openai_presets():
    assert openai_compat.PRESETS == {"OpenAI": "https://api.openai.com/v1",
                                     "OpenRouter": "https://openrouter.ai/api/v1",
                                     "Groq": "https://api.groq.com/openai/v1",
                                     "LM Studio": "http://127.0.0.1:1234/v1"}


# ---------- OpenAI-compatible (listening) ----------
def verbose_json(words, segments, language="arabic"):
    return {"language": language, "duration": 9.0, "text": "...", "words": words, "segments": segments}


def test_openai_stt_request_and_filter(net, tmp_path):
    set_keys(openai="sk")
    wav = tiny_wav(tmp_path / "a.wav")
    net.on("POST", "/audio/transcriptions", verbose_json(
        [{"word": "يا", "start": 0.5, "end": 0.8}, {"word": " جماعة", "start": 0.8, "end": 1.4},
         {"word": "Thanks", "start": 5.0, "end": 5.5}, {"word": "for", "start": 5.5, "end": 5.6}],
        [{"text": "يا جماعة", "start": 0.4, "end": 1.5, "avg_logprob": -0.2, "no_speech_prob": 0.01},
         {"text": "Thanks for watching", "start": 4.9, "end": 6.0, "avg_logprob": -1.2, "no_speech_prob": 0.9}]))
    out = get_stt(provider="openai").transcribe(str(wav), language="ar", prompt="names: Omar")
    assert out == {"language": "ar", "words": [{"w": "يا", "s": 0.5, "e": 0.8, "p": 1.0},
                                               {"w": "جماعة", "s": 0.8, "e": 1.4, "p": 1.0}]}
    body = net.calls[0]["data"]
    assert net.calls[0]["url"] == "https://api.openai.com/v1/audio/transcriptions"
    for part in (b'name="model"\r\n\r\nwhisper-1', b'name="response_format"\r\n\r\nverbose_json',
                 b'name="timestamp_granularities[]"\r\n\r\nword', b'name="language"\r\n\r\nar',
                 b'filename="audio.wav"'):
        assert part in body
    assert "names: Omar".encode() in body


def test_openai_stt_big_audio_goes_in_pieces(net, tmp_path, monkeypatch):
    set_keys(groq="gsk")
    wav = tiny_wav(tmp_path / "a.wav")
    p1, p2 = tmp_path / "p1.mp3", tmp_path / "p2.mp3"
    p1.write_bytes(b"ID3a")
    p2.write_bytes(b"ID3b")
    monkeypatch.setattr(stt_openai, "MAX_BYTES", 10)
    monkeypatch.setattr(stt_openai.media, "split_audio",
                        lambda w, out, piece_s, fmt: [(str(p1), 0.0, 600.0), (str(p2), 600.0, 50.0)])
    net.on("POST", "/audio/transcriptions",
           verbose_json([{"word": "one", "start": 1.0, "end": 1.5}], [], "english"),
           verbose_json([{"word": "two", "start": 2.0, "end": 2.5}], [], "english"))
    stt = stt_openai.OpenAISTT({"base_url": openai_compat.PRESETS["Groq"], "model": "whisper-1", "key_name": "groq"})
    assert stt.model == "whisper-large-v3"                   # the OpenAI default is not a Groq model
    out = stt.transcribe(str(wav))
    assert out["language"] == "en" and [(w["w"], w["s"]) for w in out["words"]] == [("one", 1.0), ("two", 602.0)]
    assert len(net.calls) == 2 and b'filename="audio.mp3"' in net.calls[1]["data"]
    assert net.calls[0]["headers"]["authorization"] == "Bearer gsk"


def test_openai_stt_model_without_word_times(net, tmp_path):
    set_keys(openai="sk")
    net.on("POST", "/audio/transcriptions",
           (400, {"error": {"message": "response_format 'verbose_json' is not compatible with gpt-4o-transcribe"}}))
    with pytest.raises(ProviderError, match="gives no word times"):
        stt_openai.OpenAISTT({"model": "gpt-4o-transcribe"}).transcribe(str(tiny_wav(tmp_path / "a.wav")))


# ---------- Anthropic ----------
def claude_answer(content, stop="end_turn"):
    return {"type": "message", "role": "assistant", "content": content, "stop_reason": stop}


def test_anthropic_structured_output_request(net):
    set_keys(anthropic="sk-ant-test")
    net.on("POST", "/v1/messages", claude_answer([{"type": "text", "text": json.dumps(dict(GOOD, ids=list("abc")))}]))
    llm = anth.AnthropicLLM({"allow_images": True})
    assert llm.model == "claude-haiku-4-5"
    assert llm.chat_json("SYS", "USER", SCHEMA, images=[JPEG]) == dict(GOOD, ids=["a", "b"])
    c = net.calls[0]
    assert c["url"] == "https://api.anthropic.com/v1/messages"
    assert c["headers"]["x-api-key"] == "sk-ant-test" and c["headers"]["anthropic-version"] == "2023-06-01"
    b = c["json"]
    assert b["model"] == "claude-haiku-4-5" and b["system"] == "SYS" and b["max_tokens"] >= 8192
    assert b["temperature"] == 0.2 and "tool_choice" not in b
    sent = b["output_config"]["format"]
    assert sent["type"] == "json_schema" and sent["schema"]["additionalProperties"] is False
    props = sent["schema"]["properties"]
    assert "maxItems" not in props["ids"] and "minimum" not in props["score"] and props["kind"]["enum"]
    content = b["messages"][0]["content"]
    assert content[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                      "data": _http.b64(JPEG)}}
    assert content[-1] == {"type": "text", "text": "USER"}


def test_anthropic_strict_schema_nested_objects():
    s = anth.strict_schema({"type": "object", "properties": {"m": {"type": "array", "minItems": 3, "items": {
        "type": "object", "properties": {"maximum": {"type": "string", "maxLength": 5}}}}}})
    item = s["properties"]["m"]["items"]
    assert item["additionalProperties"] is False and "minItems" not in s["properties"]["m"]
    assert item["properties"]["maximum"] == {"type": "string"}                # a property NAMED "maximum" stays


def test_anthropic_temperature_then_tool_fallback(net):
    set_keys(anthropic="k")
    net.on("POST", "/v1/messages",
           (400, {"type": "error", "error": {"type": "invalid_request_error",
                                             "message": "temperature: not supported for this model"}}),
           (400, {"type": "error", "error": {"type": "invalid_request_error",
                                             "message": "output_config: Extra inputs are not permitted"}}),
           claude_answer([{"type": "tool_use", "id": "t1", "name": "answer", "input": GOOD}], "tool_use"))
    llm = anth.AnthropicLLM({"model": "claude-3-5-haiku-latest"})
    assert llm.chat_json("s", "u", SCHEMA) == GOOD
    second, third = net.calls[1]["json"], net.calls[2]["json"]
    assert "temperature" not in second and "output_config" in second
    assert "output_config" not in third and third["tool_choice"] == {"type": "tool", "name": "answer"}
    assert third["tools"][0]["input_schema"] == SCHEMA


def test_anthropic_lowers_max_tokens_for_a_small_output_cap(net):
    set_keys(anthropic="k")
    net.on("POST", "/v1/messages",
           (400, {"type": "error", "error": {"type": "invalid_request_error", "message":
                  "max_tokens: 8192 > 4096, which is the maximum allowed number of output tokens for this model"}}),
           claude_answer([{"type": "text", "text": json.dumps(GOOD)}]))
    assert anth.AnthropicLLM({"model": "claude-3-haiku-20240307"}).chat_json("s", "u", SCHEMA) == GOOD
    assert net.calls[1]["json"]["max_tokens"] == 4096 and "output_config" in net.calls[1]["json"]


def test_anthropic_refusal_and_no_key(net):
    with pytest.raises(ProviderError, match="No Anthropic API key"):
        anth.AnthropicLLM({}).chat_json("s", "u", SCHEMA)
    set_keys(anthropic="k")
    net.on("POST", "/v1/messages", claude_answer([], "refusal"))
    assert anth.AnthropicLLM({}).chat_json("s", "u", SCHEMA) is None
    assert len(net.calls) == 1


# ---------- local listening (faster-whisper faked) ----------
class FakeWhisper:
    loads = []

    def __init__(self, name, device, compute_type, **kw):
        FakeWhisper.loads.append((name, device, compute_type, kw))
        self.name = name

    def transcribe(self, path, **kw):
        self.kw = kw
        f = np.float32
        w = lambda t, s, e, p: types.SimpleNamespace(word=t, start=f(s), end=f(e), probability=f(p))  # noqa: E731
        segs = [types.SimpleNamespace(text=" hello there", avg_logprob=f(-0.2), no_speech_prob=f(0.01), end=f(2.0),
                                      words=[w(" hello", 0.5, 0.9, 0.98), w(" there", 1.0, 1.4, 0.91)]),
                types.SimpleNamespace(text=" la la la la la la la la", avg_logprob=f(-0.3), no_speech_prob=f(0.1),
                                      end=f(6.0), words=[w(" la", 3.0, 3.2, 0.5)]),
                types.SimpleNamespace(text=" bye", avg_logprob=f(-0.1), no_speech_prob=f(0.0), end=f(8.0),
                                      words=[w(" bye", 7.0, 7.5, 0.99)])]
        if self.name == "boom":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return iter(segs), types.SimpleNamespace(language="en", duration=f(8.0))


@pytest.fixture
def fake_fw(monkeypatch):
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = FakeWhisper
    utils = types.ModuleType("faster_whisper.utils")
    utils._MODELS = {"large-v3": "Systran/faster-whisper-large-v3", "small": "Systran/faster-whisper-small"}
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    FakeWhisper.loads = []
    stt_local.LocalSTT().unload()
    yield FakeWhisper
    stt_local.LocalSTT().unload()


def test_local_stt_words_are_plain_floats(fake_fw, monkeypatch, tmp_path):
    monkeypatch.setattr(stt_local, "cuda_available", lambda: True)
    progress = []
    stt = get_stt(provider="local")
    out = stt.transcribe("x.wav", language="en-US", prompt="BroClips, Omar", on_progress=progress.append)
    assert [w["w"] for w in out["words"]] == ["hello", "there", "bye"]          # the loop segment is dropped
    assert all(type(w[k]) is float for w in out["words"] for k in ("s", "e", "p"))
    json.dumps(out)                                                             # numpy floats would fail here
    assert out["words"][0] == {"w": "hello", "s": 0.5, "e": 0.9, "p": 0.98} and out["language"] == "en"
    assert progress == sorted(progress) and progress[-1] == 1.0
    assert fake_fw.loads == [("large-v3", "cuda", "float16", {})]
    kw = stt_local._LOADED["model"].kw
    assert kw["language"] == "en" and kw["initial_prompt"] == "BroClips, Omar"
    assert kw["word_timestamps"] is True and kw["vad_filter"] is True


def test_local_stt_cpu_choice_cache_and_unload(fake_fw, monkeypatch, tmp_path):
    monkeypatch.setattr(stt_local, "cuda_available", lambda: False)
    a = stt_local.LocalSTT({"model": "auto", "models_dir": str(tmp_path / "hf")})
    a.transcribe("x.wav")
    stt_local.LocalSTT({"model": "auto", "models_dir": str(tmp_path / "hf")}).transcribe("y.wav")
    assert fake_fw.loads == [("small", "cpu", "int8", {"download_root": str(tmp_path / "hf" / "hub")})]
    stt_local.LocalSTT().unload()                          # jobs.py frees memory through a NEW instance
    assert stt_local._LOADED["model"] is None


def test_local_stt_falls_back_to_cpu_when_the_gpu_fails(fake_fw, monkeypatch):
    monkeypatch.setattr(stt_local, "cuda_available", lambda: True)
    monkeypatch.setattr(stt_local.LocalSTT, "_pick_model",
                        lambda self, device: "boom" if device == "cuda" else "small")
    out = stt_local.LocalSTT({}).transcribe("x.wav")
    assert [l[:2] for l in fake_fw.loads] == [("boom", "cuda"), ("small", "cpu")] and out["words"]


def test_local_stt_test_is_light(fake_fw, monkeypatch, tmp_path):
    monkeypatch.setattr(stt_local, "cuda_available", lambda: True)
    snap = tmp_path / "hf" / "hub" / "models--Systran--faster-whisper-large-v3" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert stt_local.LocalSTT({"models_dir": str(tmp_path / "hf")}).test() == "Ready: GPU, large-v3"
    monkeypatch.setattr(stt_local, "cuda_available", lambda: False)
    msg = stt_local.LocalSTT({"models_dir": str(tmp_path / "hf")}).test()
    assert msg.startswith("Ready: CPU") and "small" in msg and "downloads once" in msg
    assert not fake_fw.loads                                                    # nothing was loaded
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(ProviderError, match="needs faster-whisper"):
        stt_local.LocalSTT({}).test()


def test_looks_hallucinated():
    h = stt_local._looks_hallucinated
    assert h("", -0.1, 0.0) and h("thanks for watching", -1.0, 0.8)
    assert h("خد سر خد سر خد سر خد سر", -0.2, 0.1)                      # a loop: 2 words repeated
    assert not h("خد سر وخد سر وخد سر وخد سر وخد سر", -0.2, 0.1)        # 3 different words in 10: kept
    assert not h("يعني احنا كنا بنلعب", -0.3, 0.1)

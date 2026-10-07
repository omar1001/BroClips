"""Google Gemini as the Thinking AI (SPEC §4.1), plus the Gemini plumbing stt_gemini.py shares (key, Interactions
API, Files API upload).

Facts from LoL Clips (the author's private app, its cloud.py and ground-truth notes, Sept
2026) and re-checked against Google's docs on 2026-10-07: base https://generativelanguage.googleapis.com, the key in
the `x-goog-api-key` header (a free key: aistudio.google.com/apikey), the Interactions API `POST
/v1beta/interactions` with a top-level `system_instruction`, `input` = [{"type": "text"|"image"|"audio", ...}],
`generation_config` (temperature, max_output_tokens — it counts thinking + answer —, thinking_level) and
`response_format` = {"type": "text", "mime_type": "application/json", "schema": ...}; the answer is in `steps[]` of
type "model_output" → `content[]` → {"type": "text", "text"} (older answers used `outputs[]`). Default model
gemini-3.8-flash (free tier; thinking low/medium/high). Free-tier data may be read by Google's reviewers (the UI
says so). These request shapes were never run against the real API on this PC (no key here): a refused optional
field is dropped and asked again (drop_rejected), then the answer falls back to JSON-without-schema, then plain text."""
import json
from pathlib import Path

from .. import config
from ..util import log
from . import _http
from ._http import (SCHEMA_HINT, HTTPError, b64, drop_rejected, extract_json, fit_schema, image_mime,
                    images_allowed, to_openapi)
from .base import LLM, ProviderError

BASE = "https://generativelanguage.googleapis.com"
MODEL = "gemini-3.8-flash"
RETRIES = 2
NO_KEY = "No Gemini API key yet — paste one in Settings (a free key: aistudio.google.com/apikey)."
_WAITING = ("in_progress", "queued", "pending", "running", "processing")


def headers():
    k = config.secret("gemini")
    if not k:
        raise ProviderError(NO_KEY)
    return {"x-goog-api-key": k}


def interact(body, timeout=600):
    """POST /v1beta/interactions; waits while a long job says it is still in progress. -> the answer dict."""
    r = _http.post_json(f"{BASE}/v1beta/interactions", body, headers(), timeout=timeout, label="Gemini")
    for _ in range(300):
        if str(r.get("status") or "completed").lower() not in _WAITING or not r.get("id"):
            break
        _http._sleep(3)
        rid = str(r["id"])
        r = _http.get_json(f"{BASE}/v1beta/{rid if '/' in rid else 'interactions/' + rid}", headers(), timeout=60,
                           label="Gemini")
    if str(r.get("status") or "").lower() in ("failed", "cancelled"):
        why = r.get("error") or {}
        raise ProviderError(f"Gemini could not do it: {(why.get('message') if isinstance(why, dict) else why) or '?'}")
    return r


def output_text(resp):
    """The answer text: the last "model_output" step's text parts (thought steps skipped); older shapes too."""
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    found = []
    for step in resp.get("steps") or resp.get("outputs") or []:
        if not isinstance(step, dict) or step.get("type") not in (None, "model_output", "text"):
            continue
        texts = [step["text"]] if isinstance(step.get("text"), str) else []
        texts += [c["text"] for c in step.get("content") or []
                  if isinstance(c, dict) and c.get("type") in (None, "text") and isinstance(c.get("text"), str)]
        if "".join(texts).strip():
            found.append("".join(texts))
    return found[-1] if found else ""


def upload(path, mime):
    """Files API resumable upload -> (name, uri). Delete the file after use with delete_file(name)."""
    path = Path(path)
    size = path.stat().st_size
    start = json.dumps({"file": {"display_name": path.name}}).encode("utf-8")
    _, rh, _ = _http.request("POST", f"{BASE}/upload/v1beta/files", start, {
        **headers(), "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json"}, timeout=60, label="Gemini")
    url = rh.get("x-goog-upload-url") if rh is not None else None
    if not url:
        raise ProviderError("Gemini did not give an upload address — try again later.")
    # the upload address is Google's own one-time link: it needs no key
    _, _, raw = _http.request("POST", url, path.read_bytes(), {
        "Content-Length": str(size), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"},
        timeout=600, label="Gemini")
    f = _http._parse(raw, url, "Gemini").get("file") or {}
    name, uri = f.get("name"), f.get("uri")
    if not (name and uri):
        raise ProviderError("Gemini did not accept the audio upload.")
    for _ in range(90):                                 # wait until Google has processed the file
        state = str(f.get("state") or "ACTIVE").upper()
        if state != "PROCESSING":
            break
        _http._sleep(2)
        f = _http.get_json(f"{BASE}/v1beta/{name}", headers(), label="Gemini")
    if str(f.get("state") or "ACTIVE").upper() == "FAILED":
        delete_file(name)
        raise ProviderError("Gemini could not read the uploaded audio.")
    return name, uri


def delete_file(name):
    """Delete an uploaded file right after use (it would stay 48 h otherwise). Never fails."""
    try:
        _http.request("DELETE", f"{BASE}/v1beta/{name}", None, headers(), timeout=30, retries=0, label="Gemini")
    except ProviderError:
        log.info("Gemini: could not delete %s (it expires by itself after 48 h)", name)


TEST_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


class GeminiLLM(LLM):
    name = "gemini"

    def __init__(self, opts=None):
        self.opts = dict(opts or {})
        self.model = str(self.opts.get("model") or MODEL).strip()
        self.thinking = str(self.opts.get("thinking") or "low")
        self.vision = images_allowed(self.opts)   # Gemini models see pictures; sending them is the user's choice
        self._mode = 0                            # 0 schema, 1 JSON without schema, 2 plain text (after 400s)
        self._refused = set()                     # optional fields this model refused

    def _body(self, system, user, schema, imgs, gen):
        parts = [{"type": "text", "text": user}]
        parts += [{"type": "image", "data": b64(b), "mime_type": image_mime(b)} for b in imgs]
        body = {"model": self.model, "system_instruction": system, "input": parts, "generation_config": dict(gen)}
        if self._mode == 0:
            body["response_format"] = {"type": "text", "mime_type": "application/json", "schema": to_openapi(schema)}
        else:
            body["system_instruction"] = system + SCHEMA_HINT + json.dumps(schema, ensure_ascii=False)
            if self._mode == 1:
                body["response_format"] = {"type": "text", "mime_type": "application/json"}
        return body

    def _ask(self, system, user, schema, imgs, gen):
        """One answer -> (text, status). Each 400 changes one thing (drop a refused field / the pictures / the
        schema), so this ends."""
        for _ in range(10):
            body = self._body(system, user, schema, imgs, gen)
            try:
                r = interact(body)
                return output_text(r), str(r.get("status") or "")
            except HTTPError as ex:
                if ex.status != 400:
                    raise
                gone = drop_rejected(ex, body, [("generation_config", k) for k in
                                                ("temperature", "thinking_level", "max_output_tokens", "seed")])
                if gone:
                    gen.pop(gone, None)
                    self._refused.add(gone)
                    continue
                if imgs and "image" in ex.detail.lower():
                    log.warning("Gemini refused the pictures — asking with text only")
                    imgs = []
                    continue
                if self._mode < 2:
                    self._mode += 1
                    log.warning("Gemini refused the request (%s) — asking more simply (mode %d)", ex.detail[:150],
                                self._mode)
                    continue
                raise
        raise ProviderError("Gemini kept refusing the request.")

    def chat_json(self, system, user, schema, images=None, temperature=0.2, max_tokens=1500):
        headers()                                                 # a clear message when there is no key
        imgs = list(images or []) if self.vision else []
        if images and not imgs:
            log.info("Gemini: pictures not sent (Settings: sending pictures to cloud AI is off)")
        gen = {"temperature": temperature, "max_output_tokens": max(4096, int(max_tokens) * 4),
               "thinking_level": self.thinking}                   # max_output_tokens counts the thinking too
        for k in self._refused:
            gen.pop(k, None)
        for attempt in range(RETRIES + 1):
            text, status = self._ask(system, user, schema, imgs, gen)
            data = fit_schema(extract_json(text), schema)
            if data is not None:
                return data
            log.warning("Gemini's answer did not fit the schema (attempt %d, status %s): %s", attempt + 1,
                        status or "?", text[:200])
            if status.lower() == "incomplete" and "max_output_tokens" in gen:
                gen["max_output_tokens"] *= 2
        return None

    def test(self):
        ans = self.chat_json("You are a connection test.", "Answer with ok = true.", TEST_SCHEMA, max_tokens=64)
        if not ans:
            raise ProviderError(f"Gemini ({self.model}) answered, but not in the expected form.")
        return f"Ready: Gemini, model {self.model}"


def model_info(model):
    """GET /v1beta/models/<model> (free: uses none of the daily requests) — checks the key and the model name."""
    try:
        return _http.get_json(f"{BASE}/v1beta/models/{model}", headers(), retries=1, label="Gemini")
    except HTTPError as ex:
        if ex.status == 404:
            raise ProviderError(f"Gemini has no model named {model} — check the model name in Settings.") from None
        raise

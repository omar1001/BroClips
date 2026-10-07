"""OpenAI, or any OpenAI-compatible service (OpenRouter, Groq, LM Studio, …), as the Thinking AI (SPEC §4.1).

POST {base_url}/chat/completions with `Authorization: Bearer <key>`. JSON answers, most exact first — each step is
taken only when the server refuses the previous one with a 400, and the instance remembers what worked:
  1. response_format {"type": "json_schema", "json_schema": {"name": "answer", "schema": …, "strict": false}}
  2. response_format {"type": "json_object"} + the schema in the system prompt
  3. plain text + the schema in the prompt, the first JSON object taken from the answer (extract_json)
Before stepping down, a 400 that names an optional field is repaired instead: `temperature` dropped (reasoning
models allow only the default), `max_tokens` renamed to `max_completion_tokens` (newer OpenAI models), pictures
dropped (a model that cannot see). Pictures go as data URLs, only when allowed (SPEC §4.3) or when the server runs on
this PC / the home network. The key is config.secret(opts["key_name"] or "openai"); local servers need none."""
import json

from .. import config
from ..util import log
from . import _http
from ._http import (SCHEMA_HINT, HTTPError, b64, extract_json, fit_schema, image_mime, images_allowed,
                    is_local_url)
from .base import LLM, ProviderError

PRESETS = {"OpenAI": "https://api.openai.com/v1", "OpenRouter": "https://openrouter.ai/api/v1",
           "Groq": "https://api.groq.com/openai/v1", "LM Studio": "http://127.0.0.1:1234/v1"}
RETRIES = 2
TEST_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def service_label(base_url):
    """"OpenAI" / "Groq" / … for a preset base URL, else the host name (for messages)."""
    base = str(base_url or "").rstrip("/")
    for label, url in PRESETS.items():
        if base == url:
            return label
    return _http._where(base)[0] or "the AI service"


def auth_headers(base_url, key_name):
    """{"Authorization": "Bearer …"}; {} for a server on this PC / the home network without a key."""
    key = config.secret(key_name or "openai")
    if key:
        return {"Authorization": f"Bearer {key}"}
    if is_local_url(base_url):
        return {}
    raise ProviderError(f"No API key for {service_label(base_url)} yet — paste it in Settings.")


class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, opts=None):
        self.opts = dict(opts or {})
        self.base = str(self.opts.get("base_url") or PRESETS["OpenAI"]).strip().rstrip("/")
        self.model = str(self.opts.get("model") or "gpt-4o-mini").strip()
        self.key_name = str(self.opts.get("key_name") or "openai")
        self.local = is_local_url(self.base)
        self.label = service_label(self.base)
        self.vision = self.local or images_allowed(self.opts)    # may pictures be sent? (the model must see, too)
        self._mode = 0                                          # 0 json_schema, 1 json_object, 2 plain text
        self._extra = {}                                        # field renames / removals that worked

    def _body(self, system, user, schema, imgs, temperature, max_tokens):
        content = user
        if imgs:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": f"data:{image_mime(b)};base64,{b64(b)}"}} for b in imgs]
        sys_text = system if self._mode == 0 else system + SCHEMA_HINT + json.dumps(schema, ensure_ascii=False)
        body = {"model": self.model, "messages": [{"role": "system", "content": sys_text},
                                                  {"role": "user", "content": content}],
                "temperature": temperature, "max_tokens": int(max_tokens)}
        if self._extra.get("max_completion_tokens"):
            body["max_completion_tokens"] = body.pop("max_tokens")
        if self._extra.get("no_temperature"):
            body.pop("temperature")
        if self._mode == 0:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": schema, "strict": False}}
        elif self._mode == 1:
            body["response_format"] = {"type": "json_object"}
        return body

    def _complete(self, body):
        return _http.post_json(f"{self.base}/chat/completions", body, auth_headers(self.base, self.key_name),
                               timeout=600, label=self.label)

    def _ask(self, system, user, schema, imgs, temperature, max_tokens):
        """One answer -> (text, finish_reason). Every 400 changes one thing, so this ends."""
        for _ in range(10):
            body = self._body(system, user, schema, imgs, temperature, max_tokens)
            try:
                r = self._complete(body)
            except HTTPError as ex:
                if ex.status not in (400, 422):
                    raise
                d = ex.detail.lower()
                if "max_completion_tokens" in d and not self._extra.get("max_completion_tokens"):
                    self._extra["max_completion_tokens"] = True
                elif "temperature" in d and not self._extra.get("no_temperature"):
                    self._extra["no_temperature"] = True
                elif imgs and "image" in d:
                    log.warning("%s refused the pictures — asking with text only", self.label)
                    imgs = []
                elif self._mode < 2:
                    self._mode += 1
                    log.warning("%s refused the request (%s) — asking more simply (mode %d)", self.label,
                                ex.detail[:150], self._mode)
                else:
                    raise
                continue
            choice = (r.get("choices") or [{}])[0] or {}
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):                       # some servers send parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content or ""), str(choice.get("finish_reason") or "")
        raise ProviderError(f"{self.label} kept refusing the request.")

    def chat_json(self, system, user, schema, images=None, temperature=0.2, max_tokens=1500):
        auth_headers(self.base, self.key_name)                   # a clear message when there is no key
        imgs = list(images or []) if self.vision else []
        if images and not imgs:
            log.info("%s: pictures not sent (Settings: sending pictures to cloud AI is off)", self.label)
        budget = int(max_tokens)
        for attempt in range(RETRIES + 1):
            text, finish = self._ask(system, user, schema, imgs, temperature, budget)
            data = fit_schema(extract_json(text), schema)
            if data is not None:
                return data
            log.warning("%s's answer did not fit the schema (attempt %d, %s): %s", self.label, attempt + 1,
                        finish or "?", text[:200])
            if finish == "length":
                budget *= 2
        return None

    def list_models(self):
        """Model ids the service offers (for Settings)."""
        r = _http.get_json(f"{self.base}/models", auth_headers(self.base, self.key_name), label=self.label)
        return sorted(str(m.get("id")) for m in (r.get("data") or []) if isinstance(m, dict) and m.get("id"))

    def test(self):
        ans = self.chat_json("You are a connection test.", "Answer with ok = true.", TEST_SCHEMA, max_tokens=64)
        if not ans:
            raise ProviderError(f"{self.label} ({self.model}) answered, but not in the expected JSON form — try "
                                "another model.")
        return f"Ready: {self.label}, model {self.model}"

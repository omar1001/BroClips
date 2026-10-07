"""Anthropic Claude as the Thinking AI (SPEC §4.1): the Messages API with the standard library (no SDK, SPEC §3).

Shapes from the claude-api skill (model table cached 2026-09-25): POST https://api.anthropic.com/v1/messages, headers
`x-api-key` + `anthropic-version: 2023-06-01`. JSON answers use structured outputs —
output_config.format = {"type": "json_schema", "schema": …} — which every current model supports (Haiku 4.5 up to
Opus 5.5 / Fable 5.1). That schema dialect needs `additionalProperties: false` on every object and refuses
numeric / string-length / array-size limits, so those are removed here (fit_schema applies maxItems afterwards).
A forced tool (the SPEC's first idea) is only a fallback: Opus 5.5, Sonnet 5.5 and Fable 5.1 answer 400 to
tool_choice "tool". On a 400 that names `temperature` (Opus 4.7+, Sonnet 5.x allow only the default) it is dropped;
other 400s step down: structured output -> forced tool -> plain text + extract_json.
Default model claude-haiku-4-5: the cheapest current model ($1 / $5 per million tokens in/out), good at JSON, no
thinking unless asked (fast). Settings may name another, e.g. claude-sonnet-5-5 (smarter, 2x the price, thinks
first — hence the roomy max_tokens). A safety refusal (stop_reason "refusal") is treated as "no answer" (None)."""
import copy
import json

from .. import config
from ..util import log
from . import _http
from ._http import SCHEMA_HINT, HTTPError, b64, extract_json, fit_schema, image_mime, images_allowed
from .base import LLM, ProviderError

API = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MODEL = "claude-haiku-4-5"
RETRIES = 2
MIN_MAX_TOKENS = 8192            # room for thinking on models that think by default; unused tokens cost nothing
TEST_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
                "maxLength", "pattern", "maxItems", "uniqueItems", "minProperties", "maxProperties", "default",
                "examples", "$schema", "propertyOrdering", "nullable"}


def strict_schema(schema):
    """A copy of `schema` that structured outputs accept: additionalProperties false on every object, unsupported
    limits removed (minItems kept only as 0 or 1)."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in _UNSUPPORTED or (k == "minItems" and not (isinstance(v, int) and v <= 1)):
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {name: strict_schema(sub) for name, sub in v.items()}
        elif k == "items":
            out[k] = strict_schema(v)
        elif k in ("anyOf", "allOf") and isinstance(v, list):
            out[k] = [strict_schema(x) for x in v]
        elif k in ("$defs", "definitions") and isinstance(v, dict):
            out[k] = {name: strict_schema(sub) for name, sub in v.items()}
        else:
            out[k] = copy.deepcopy(v)
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
    return out


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, opts=None):
        self.opts = dict(opts or {})
        self.model = str(self.opts.get("model") or MODEL).strip()
        self.vision = images_allowed(self.opts)     # every current Claude model sees pictures; sending is a choice
        self._mode = 0                              # 0 structured output, 1 forced tool, 2 plain text
        self._no_temperature = False

    @staticmethod
    def _headers():
        key = config.secret("anthropic")
        if not key:
            raise ProviderError("No Anthropic API key yet — paste it in Settings (console.anthropic.com).")
        return {"x-api-key": key, "anthropic-version": API_VERSION}

    def _body(self, system, user, schema, imgs, temperature, max_tokens):
        content = [{"type": "image", "source": {"type": "base64", "media_type": image_mime(b), "data": b64(b)}}
                   for b in imgs]
        content.append({"type": "text", "text": user})
        body = {"model": self.model, "max_tokens": int(max_tokens), "system": system,
                "messages": [{"role": "user", "content": content}]}
        if not self._no_temperature:
            body["temperature"] = temperature
        if self._mode == 0:
            body["output_config"] = {"format": {"type": "json_schema", "schema": strict_schema(schema)}}
        elif self._mode == 1:
            body["tools"] = [{"name": "answer", "description": "Give the answer in exactly this shape.",
                              "input_schema": schema}]
            body["tool_choice"] = {"type": "tool", "name": "answer"}
        else:
            body["system"] = system + SCHEMA_HINT + json.dumps(schema, ensure_ascii=False)
        return body

    def _ask(self, system, user, schema, imgs, temperature, max_tokens):
        """One answer -> (parsed dict or None, stop_reason). Every 400 changes one thing, so this ends."""
        for _ in range(10):
            body = self._body(system, user, schema, imgs, temperature, max_tokens)
            try:
                r = _http.post_json(API, body, self._headers(), timeout=600, label="Anthropic")
            except HTTPError as ex:
                if ex.status != 400:
                    raise
                d = ex.detail.lower()
                if "temperature" in d and not self._no_temperature:
                    self._no_temperature = True
                elif "max_tokens" in d and max_tokens > 4096:          # an older model with a smaller output cap
                    max_tokens = 4096
                elif imgs and "image" in d:
                    log.warning("Anthropic refused the pictures — asking with text only")
                    imgs = []
                elif self._mode < 2:
                    self._mode += 1
                    log.warning("Anthropic refused the request (%s) — asking another way (mode %d)", ex.detail[:150],
                                self._mode)
                else:
                    raise
                continue
            stop = str(r.get("stop_reason") or "")
            blocks = [b for b in r.get("content") or [] if isinstance(b, dict)]
            if self._mode == 1:
                tool = next((b for b in blocks if b.get("type") == "tool_use"), None)
                return (tool.get("input") if tool and isinstance(tool.get("input"), dict) else None), stop
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            return extract_json(text), stop
        raise ProviderError("Anthropic kept refusing the request.")

    def chat_json(self, system, user, schema, images=None, temperature=0.2, max_tokens=1500):
        self._headers()                                           # a clear message when there is no key
        imgs = list(images or []) if self.vision else []
        if images and not imgs:
            log.info("Anthropic: pictures not sent (Settings: sending pictures to cloud AI is off)")
        budget = max(MIN_MAX_TOKENS, int(max_tokens))
        for attempt in range(RETRIES + 1):
            parsed, stop = self._ask(system, user, schema, imgs, temperature, budget)
            if stop == "refusal":
                log.warning("Claude declined to answer this question (safety refusal)")
                return None
            data = fit_schema(parsed, schema)
            if data is not None:
                return data
            log.warning("Claude's answer did not fit the schema (attempt %d, stop %s)", attempt + 1, stop or "?")
            if stop == "max_tokens":
                budget *= 2
        return None

    def test(self):
        ans = self.chat_json("You are a connection test.", "Answer with ok = true.", TEST_SCHEMA, max_tokens=64)
        if not ans:
            raise ProviderError(f"Claude ({self.model}) answered, but not in the expected form.")
        return f"Ready: Claude, model {self.model}"

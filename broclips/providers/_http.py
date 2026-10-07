"""Shared plumbing for the AI providers (SPEC §4): HTTP with the standard library only (no vendor SDKs, SPEC §3),
plain-English errors, and the small helpers every provider needs.

HTTP: `request` / `post_json` / `get_json` / `post_multipart`. Up to 2 retries with a growing wait for the problems
that pass by themselves (429 rate limit, 5xx server trouble, time-outs, a dropped connection); anything else fails at
once. Failures raise `HTTPError`: a ProviderError whose text the UI can show as is, plus `.status` (0 = no answer)
and `.detail` (the service's own words) so a provider can react — e.g. ask again more simply after a 400.
API keys travel only in the headers a provider passes in. Nothing here logs a header, a request body or a query
string, and long key-like strings are masked out of error texts.

Helpers: extract_json (the JSON object inside a chatty answer), fit_schema (cut an answer to its schema),
to_openapi (Gemini's schema subset), drop_rejected (remove an optional field a service refused), spread_words (word
times from segment times), image_mime / b64, is_local_url, images_allowed (SPEC §4.3), guess_language."""
import base64
import copy
import http.client
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlsplit

from .. import config
from ..util import log
from .base import ProviderError

RETRY_WAIT = (3.0, 10.0)        # seconds before the 1st / 2nd retry (a Retry-After header may ask for more, max 60 s)
_sleep = time.sleep             # tests replace it

KEY_REFUSED = "The API key was refused — check it in Settings"
RATE_LIMIT = "Rate limit or quota reached — wait or check your plan"
SERVER_TROUBLE = "The service had a problem — try again"
SCHEMA_HINT = ("\n\nAnswer with ONE JSON object only (no other text, no code fences). It must follow this JSON "
               "schema:\n")


class HTTPError(ProviderError):
    """An HTTP problem in plain English. `.status`: the HTTP status (0 = no answer at all); `.detail`: the service's
    own error words (key-like strings masked), for providers that react to a specific refusal."""

    def __init__(self, message, status=0, detail=""):
        super().__init__(message)
        self.status = status
        self.detail = detail or ""


# ---------- HTTP ----------
def _where(url):
    """(host[:port], path) — never the query string, which could hold a key."""
    p = urlsplit(url)
    return (p.netloc or url), p.path


_KEYLIKE = re.compile(r"[A-Za-z0-9_\-]{24,}")


def _mask(text):
    return _KEYLIKE.sub("…", str(text or ""))


def _detail(raw):
    """The service's own words from an error body: {"error": {"message"}} (OpenAI, Anthropic, Gemini),
    {"error": "..."} (Ollama), {"detail": ...} or plain text."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw or "")
    try:
        j = json.loads(text)
    except ValueError:
        return " ".join(text.split())[:400]
    if isinstance(j, list) and j:
        j = j[0]
    err = (j.get("error") or j.get("detail") or j) if isinstance(j, dict) else j
    if isinstance(err, dict):
        msg = err.get("message") or err.get("msg") or json.dumps(err, ensure_ascii=False)
        if err.get("param"):
            msg = f"{msg} (param: {err['param']})"
    else:
        msg = err
    return " ".join(str(msg).split())[:400]


def _message(status, label, detail):
    if status in (401, 403):
        text = KEY_REFUSED + (f" ({detail})" if detail else "")
    elif status == 429:
        text = RATE_LIMIT
    elif status >= 500:            # Ollama's 500s say useful things ("model requires more system memory ...")
        text = SERVER_TROUBLE + (f" ({detail[:200]})" if detail else "")
    else:
        text = f"the request was refused ({status})" + (f": {detail}" if detail else "")
    return f"{label}: {text}."


def _retry_after(headers):
    try:
        return min(60.0, float(headers.get("Retry-After")))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _read_error(e):
    try:
        return e.read()
    except Exception:
        return b""


def request(method, url, data=None, headers=None, timeout=120, retries=2, label=""):
    """One HTTP call -> (status, response headers, body bytes). Retries 429 / 5xx / time-outs / dropped connections;
    raises HTTPError (plain English) for everything else and after the last retry."""
    where, path = _where(url)
    who = label or where
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=dict(headers or {}), method=method)
        wait = RETRY_WAIT[min(attempt, len(RETRY_WAIT) - 1)]
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                log.debug("%s %s%s -> %s (%.1f s)", method, where, path, getattr(r, "status", 200),
                          time.monotonic() - t0)
                return getattr(r, "status", 200), r.headers, body
        except urllib.error.HTTPError as e:
            detail = _mask(_detail(_read_error(e)))
            last = HTTPError(_message(e.code, who, detail), e.code, detail)
            log.warning("%s %s%s -> %s %s", method, where, path, e.code, detail[:200])
            if not (e.code == 429 or e.code >= 500):
                raise last from None
            wait = max(wait, _retry_after(e.headers))
        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                last = HTTPError(f"{who} did not answer in time.", 0, "timeout")
            elif isinstance(reason, (ConnectionResetError, ConnectionAbortedError)):
                last = HTTPError(f"The connection to {who} broke.", 0, type(reason).__name__)
            else:
                dns = isinstance(reason, socket.gaierror)
                raise HTTPError(f"Cannot reach {where}" + (" (no internet?)" if dns else "") + ".", 0,
                                _mask(reason)) from None
        except (TimeoutError, socket.timeout):
            last = HTTPError(f"{who} did not answer in time.", 0, "timeout")
        except ConnectionRefusedError as e:
            raise HTTPError(f"Cannot reach {where}.", 0, type(e).__name__) from None
        except (ConnectionError, http.client.HTTPException) as e:
            last = HTTPError(f"The connection to {who} broke.", 0, type(e).__name__)
        if attempt < retries:
            log.warning("%s %s%s: %s Retry %d in %.0f s.", method, where, path, last, attempt + 1, wait)
            _sleep(wait)
    raise last


def _parse(raw, url, label=""):
    if not raw or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        where, _ = _where(url)
        raise HTTPError(f"{label or where} sent an answer that is not JSON.", 200,
                        _mask(raw[:200].decode("utf-8", "replace"))) from None


def post_json(url, body, headers=None, timeout=120, retries=2, label=""):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json", **(headers or {})}
    _, _, raw = request("POST", url, data, h, timeout, retries, label)
    return _parse(raw, url, label)


def get_json(url, headers=None, timeout=30, retries=2, label=""):
    _, _, raw = request("GET", url, None, headers, timeout, retries, label)
    return _parse(raw, url, label)


def post_multipart(url, fields, files, headers=None, timeout=600, retries=2, label=""):
    """multipart/form-data POST. fields: {name: value or [values]} (a list repeats the field, e.g.
    "timestamp_granularities[]"); files: {name: (filename, bytes, mime)}. -> parsed JSON answer."""
    boundary = "----BroClips" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        for v in (value if isinstance(value, (list, tuple)) else [value]):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{v}\r\n'
                         .encode("utf-8"))
    for name, (filename, content, mime) in files.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                     f"Content-Type: {mime}\r\n\r\n".encode("utf-8") + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    h = {"Content-Type": f"multipart/form-data; boundary={boundary}", **(headers or {})}
    _, _, raw = request("POST", url, b"".join(parts), h, timeout, retries, label)
    return _parse(raw, url, label)


# ---------- answers ----------
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def _balanced_end(s, i):
    """Index of the '}' closing the '{' at s[i] (strings and escapes respected), or -1."""
    depth, in_str, esc = 0, False, False
    for k in range(i, len(s)):
        ch = s[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return k
    return -1


def extract_json(text):
    """The JSON object in a model's answer: the whole text, else a ```json fence, else the first balanced {...}
    that parses. -> dict or None."""
    if isinstance(text, dict):
        return text
    s = str(text or "").strip()
    if not s:
        return None
    for cand in [s] + _FENCE.findall(s):
        try:
            v = json.loads(cand)
        except ValueError:
            continue
        if isinstance(v, dict):
            return v
    i = s.find("{")
    while i != -1:
        end = _balanced_end(s, i)
        if end != -1:
            try:
                v = json.loads(s[i:end + 1])
                if isinstance(v, dict):
                    return v
            except ValueError:
                pass
        i = s.find("{", i + 1)
    return None


_BAD = object()


def _fit(v, sc):
    if not isinstance(sc, dict) or not sc:
        return v
    if sc.get("anyOf"):
        for alt in sc["anyOf"]:
            r = _fit(v, alt)
            if r is not _BAD:
                return r
        return _BAD
    t = sc.get("type")
    if isinstance(t, list):
        if v is None and "null" in t:
            return None
        t = next((x for x in t if x != "null"), None)
    if v is None:
        return None if sc.get("nullable") else _BAD
    if t is None:
        t = "object" if "properties" in sc else "array" if "items" in sc else None
    if t == "object":
        if not isinstance(v, dict):
            return _BAD
        props, req = sc.get("properties") or {}, set(sc.get("required") or [])
        if any(k not in v for k in req):
            return _BAD
        out = dict(v)
        for k, sub in props.items():
            if k in out:
                r = _fit(out[k], sub)
                if r is _BAD:
                    if k in req:
                        return _BAD
                    del out[k]                      # an optional field that does not fit is left out
                else:
                    out[k] = r
        return out
    if t == "array":
        if not isinstance(v, list):
            return _BAD
        items = [r for r in (_fit(x, sc.get("items") or {}) for x in v) if r is not _BAD]
        if isinstance(sc.get("maxItems"), int):
            items = items[:sc["maxItems"]]
        return items
    if t in ("integer", "number"):
        if isinstance(v, bool):
            return _BAD
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                return _BAD
        if not isinstance(v, (int, float)):
            return _BAD
        if t == "integer":
            v = int(round(v))
    elif t == "boolean":
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        if not isinstance(v, bool):
            return _BAD
    elif t == "string":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = str(v)
        if not isinstance(v, str):
            return _BAD
    if sc.get("enum"):
        if v in sc["enum"]:
            return v
        if isinstance(v, str):                       # "Funny" for "funny"
            for e in sc["enum"]:
                if isinstance(e, str) and e.lower() == v.strip().lower():
                    return e
        return _BAD
    return v


def fit_schema(value, schema):
    """Make a parsed answer fit its schema where that is cheap and safe: arrays longer than maxItems are cut, array
    items (and optional fields) that do not fit are dropped, numbers sent as text become numbers, enum case is
    fixed. -> the fitted answer, or None when it cannot fit (not an object, a required key missing)."""
    if not isinstance(value, dict):
        return None
    out = _fit(value, schema or {})
    return None if out is _BAD or not isinstance(out, dict) else out


_OPENAPI_KEYS = {"type", "format", "description", "nullable", "enum", "items", "properties", "required",
                 "minItems", "maxItems", "minimum", "maximum", "anyOf", "propertyOrdering", "title"}


def to_openapi(schema):
    """A copy of a JSON schema in the subset Gemini's response schema accepts (OpenAPI 3.0 style): unknown keys such
    as additionalProperties, $schema, default or pattern are dropped, `const` becomes a one-value enum, a type list
    like ["string", "null"] becomes "string" + nullable, enums stay only on strings and `required` only names real
    properties."""
    if not isinstance(schema, dict):
        return schema
    s = copy.deepcopy(schema)
    if "const" in s and "enum" not in s:
        s["enum"] = [s["const"]]
    t = s.get("type")
    if isinstance(t, list):
        rest = [x for x in t if x != "null"]
        s["type"] = rest[0] if rest else "string"
        if "null" in t:
            s["nullable"] = True
    out = {}
    for k, v in s.items():
        if k not in _OPENAPI_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {name: to_openapi(sub) for name, sub in v.items()}
        elif k == "items":
            out[k] = to_openapi(v)
        elif k == "anyOf" and isinstance(v, list):
            out[k] = [to_openapi(x) for x in v]
        else:
            out[k] = v
    if "enum" in out and out.get("type", "string") != "string":
        out.pop("enum")
    if "required" in out:
        props = out.get("properties") or {}
        out["required"] = [r for r in out["required"] if r in props]
    return out


def drop_rejected(err, body, paths):
    """After a 400: remove the first optional field that the service's error text names — e.g. ("temperature",) or
    ("generation_config", "thinking_level") — so the same question can be asked again without it.
    -> the removed field's name, or "" when the error names none of them (a real error)."""
    text = (getattr(err, "detail", "") or str(err)).lower()
    for path in paths:
        parent = body
        for k in path[:-1]:
            parent = parent.get(k) if isinstance(parent, dict) else None
        if isinstance(parent, dict) and path[-1] in parent and path[-1].lower() in text:
            parent.pop(path[-1])
            log.info("the AI service refused '%s' — asking again without it", path[-1])
            return path[-1]
    return ""


# ---------- small shared helpers ----------
def spread_words(segments, p=1.0):
    """Word times when a service gives only segment times: [(text, start, end)] -> words, each segment's time shared
    out by character length (a long word takes longer to say)."""
    out = []
    for text, s, e in segments:
        toks = str(text or "").split()
        s, e = float(s), float(e)
        if not toks or e <= s:
            continue
        total = sum(len(t) + 1 for t in toks)
        t = s
        for tok in toks:
            d = (e - s) * (len(tok) + 1) / total
            out.append({"w": tok, "s": round(t, 2), "e": round(t + d, 2), "p": float(p)})
            t += d
    return out


def b64(data):
    return base64.b64encode(data).decode("ascii")


def image_mime(data):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def is_local_url(url):
    """True for a server on this PC or the home network (LM Studio, Ollama, a LAN box): no key needed, and pictures
    sent there do not leave the user's own machines."""
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def images_allowed(opts):
    """SPEC §4.3: may video frames be sent to a cloud AI? opts["allow_images"] wins, else the Settings switch
    (off by default)."""
    v = (opts or {}).get("allow_images")
    if v is None:
        v = config.settings().get("allow_images_to_cloud", False)
    return bool(v)


_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")


def guess_language(text):
    """A rough language code from the script when a service reports none: "ar" for mostly Arabic letters, else
    "en" (only the script matters for captions: right-to-left and the font)."""
    ar, la = len(_ARABIC.findall(text or "")), len(_LATIN.findall(text or ""))
    return "ar" if ar > la else "en"

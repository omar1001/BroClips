"""Pick the AI provider chosen in Settings. Each provider class takes its own settings sub-dict
(e.g. settings["thinking"]["ollama"]) in its constructor and reads API keys with config.secret(<name>)."""
import importlib

from .. import config
from .base import LLM, STT, ProviderError  # noqa: F401  (re-exported)

LLMS = {"ollama": "ollama:OllamaLLM", "gemini": "gemini:GeminiLLM", "openai": "openai_compat:OpenAILLM",
        "anthropic": "anthropic:AnthropicLLM", "fake": "fake:FakeLLM"}
STTS = {"local": "stt_local:LocalSTT", "openai": "stt_openai:OpenAISTT", "gemini": "stt_gemini:GeminiSTT",
        "fake": "fake:FakeSTT"}


def _cls(spec):
    mod, name = spec.split(":")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)


def get_llm(settings=None, provider=None):
    """The Thinking AI (an LLM instance) from Settings, or a given provider name with its saved options."""
    s = settings or config.settings()
    name = provider or s["thinking"]["provider"]
    if name not in LLMS:
        raise ProviderError(f"Unknown thinking AI '{name}'. Pick one in Settings.")
    return _cls(LLMS[name])(s["thinking"].get(name, {}))


def get_stt(settings=None, provider=None):
    """The Listening AI (speech-to-text) from Settings."""
    s = settings or config.settings()
    name = provider or s["listening"]["provider"]
    if name not in STTS:
        raise ProviderError(f"Unknown listening AI '{name}'. Pick one in Settings.")
    return _cls(STTS[name])(s["listening"].get(name, {}))

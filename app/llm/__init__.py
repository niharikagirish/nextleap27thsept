"""LLM backend selection (architecture.md ADR A7/A9).

``get_llm()`` resolves ``generation.provider`` from ``config.yaml`` into a live
client, and **never raises for a missing key** — it falls back to the ``echo``
retrieval-only backend and says so on stdout. That fallback is the reason the
demo cannot die on demo day: a missing, expired, or rate-limited key degrades the
answer quality visibly instead of taking the whole application down.

Selection order when ``provider: env`` (the default):

1. ``GROQ_API_KEY`` / ``GEMINI_API_KEY`` present -> :class:`FreeTierClient`
2. ``LLM_API_KEY`` present -> :class:`OpenAICompatClient`
3. nothing -> :class:`EchoClient` + a one-line notice

The result is cached, because constructing a client is cheap but the notice
should print once per process rather than once per query.
"""

from __future__ import annotations

from typing import Any

from ..config import SETTINGS
from ..disclaimers import NO_LLM_KEY
from .base import LLMClient, LLMError, ParseError
from .echo import EchoClient
from .free_tier import FreeTierClient, available as free_tier_available
from .openai_compat import OpenAICompatClient

_client: Any | None = None
_notice_shown = False


def get_llm(force: str | None = None) -> LLMClient:
    """Return the configured LLM client, falling back to ``echo`` if needed."""
    global _client, _notice_shown
    if _client is not None and force is None:
        return _client

    provider = (force or SETTINGS.get("generation.provider", "env") or "env").lower()
    client = _build(provider)

    if isinstance(client, EchoClient) and not _notice_shown:
        print(f"  NOTE: {NO_LLM_KEY}", flush=True)
        _notice_shown = True

    if force is None:
        _client = client
    return client


def _build(provider: str) -> LLMClient:
    if provider == "echo":
        return EchoClient(max_sentences=int(SETTINGS.get("generation.max_sentences", 3)))

    if provider == "free_tier":
        try:
            return FreeTierClient()
        except LLMError as exc:
            print(f"  free_tier unavailable ({exc}); using retrieval-only mode.",
                  flush=True)
            return EchoClient(
                max_sentences=int(SETTINGS.get("generation.max_sentences", 3))
            )

    if provider == "openai_compat":
        try:
            return OpenAICompatClient(
                api_key=SETTINGS.llm_api_key or "",
                model=SETTINGS.get("generation.model"),
            )
        except LLMError as exc:
            print(f"  openai_compat unavailable ({exc}); using retrieval-only mode.",
                  flush=True)
            return EchoClient(
                max_sentences=int(SETTINGS.get("generation.max_sentences", 3))
            )

    if provider != "env":
        raise LLMError(
            f"unknown generation.provider {provider!r}; expected one of "
            "env, echo, free_tier, openai_compat"
        )

    # provider == "env": sniff the environment, cheapest-and-most-likely first.
    if free_tier_available():
        try:
            return FreeTierClient()
        except LLMError:
            pass
    if SETTINGS.llm_api_key:
        try:
            return OpenAICompatClient(
                api_key=SETTINGS.llm_api_key or "",
                model=SETTINGS.get("generation.model"),
            )
        except LLMError as exc:
            print(f"  LLM_API_KEY set but unusable ({exc}); "
                  f"using retrieval-only mode.", flush=True)
    return EchoClient(max_sentences=int(SETTINGS.get("generation.max_sentences", 3)))


def describe() -> str:
    """One line naming the active backend, for ``--debug`` output and the UI."""
    client = get_llm()
    extra = ""
    if isinstance(client, OpenAICompatClient):
        extra = f" model={client.model}"
    return f"{client.name}{extra}"


def reset_cache() -> None:
    """Drop the cached client. Tests only."""
    global _client, _notice_shown
    _client = None
    _notice_shown = False


__all__ = [
    "get_llm", "describe", "reset_cache", "LLMClient", "LLMError", "ParseError",
    "EchoClient", "OpenAICompatClient", "FreeTierClient",
]

"""Free-tier chat backends (Groq, and anything else key-only).

A thin, opinionated layer over :mod:`app.llm.openai_compat` that fills in the
base URL and a sensible default model for the free tiers a student project can
actually sign up for in two minutes. If ``generation.provider: free_tier`` is
configured but no known provider key is present, this module explains exactly
which env var is missing instead of letting a 401 surface from the adapter.

Kept deliberately separate from ``openai_compat`` so that pointing the project at
a paid endpoint later is a ``config.yaml`` change, not a code change.

Model selection, highest precedence first:

1. the ``model=`` constructor argument
2. the provider's own env var - ``GROQ_MODEL`` / ``GEMINI_MODEL``
3. the hardcoded default in :data:`FREE_TIER_PROVIDERS`
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMError
from .openai_compat import OpenAICompatClient

# provider key env var -> (base url, default model)
#
# The defaults are verified working, not aspirational. Groq retired the
# llama-3.1 and llama-3.3 chat models (they now answer 404 model_not_found), so
# a stale entry here means "paste a Groq key and it just works" quietly stops
# working. Verified 2026-09-27 against a live key: openai/gpt-oss-120b and
# openai/gpt-oss-20b both respond; llama-3.3-70b-versatile does not.
#
# Override per provider with the matching *_MODEL env var, or globally with
# LLM_MODEL on the generic OpenAI-compatible path.
FREE_TIER_PROVIDERS: dict[str, tuple[str, str]] = {
    "GROQ_API_KEY": (
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
    ),
    "GEMINI_API_KEY": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.0-flash",
    ),
}


def model_env_var(key_env_name: str) -> str:
    """The per-provider model override env var for a key env var.

    ``GROQ_API_KEY`` -> ``GROQ_MODEL``, ``GEMINI_API_KEY`` -> ``GEMINI_MODEL``.

    Lets a user pin an exact model per provider - including one this table does
    not know about yet - without editing code. Unknown names are mapped by the
    same rule rather than rejected, so adding a provider above is the only step
    needed to make its model configurable too.
    """
    suffix = "_API_KEY"
    if key_env_name.endswith(suffix):
        return f"{key_env_name[: -len(suffix)]}_MODEL"
    return f"{key_env_name}_MODEL"


class FreeTierClient(OpenAICompatClient):
    """OpenAI-compatible client with a free-tier default filled in."""

    name = "free_tier"

    def __init__(self, api_key: str | None = None, provider: str | None = None,
                 model: str | None = None) -> None:
        chosen = provider or os.getenv("LLM_PROVIDER") or ""
        key = api_key
        base_url = None
        default_model = model

        if not chosen:
            for env_name, (url, _fallback) in FREE_TIER_PROVIDERS.items():
                if os.getenv(env_name):
                    chosen = env_name
                    key = key or os.getenv(env_name, "")
                    base_url = url
                    break

        if not chosen:
            raise LLMError(
                "no free-tier provider key found. Set one of "
                f"{', '.join(sorted(FREE_TIER_PROVIDERS))} in .env, or set "
                "LLM_API_KEY + LLM_BASE_URL for any OpenAI-compatible endpoint.\n"
                "With no key at all the app still runs in retrieval-only mode."
            )

        if chosen in FREE_TIER_PROVIDERS:
            url, fallback_model = FREE_TIER_PROVIDERS[chosen]
            base_url = base_url or url
            # Precedence: explicit argument, then the provider's own
            # <PROVIDER>_MODEL env var, then the hardcoded free-tier default.
            #
            # The env var has to be read *here* rather than in the parent class.
            # OpenAICompatClient resolves `model or os.getenv("LLM_MODEL")`, but
            # by the time it runs, `default_model` is already the non-empty
            # hardcoded fallback - so LLM_MODEL would always lose, and a user who
            # deliberately picked a specific model would silently get another one
            # with no warning anywhere.
            default_model = (
                default_model
                or os.getenv(model_env_var(chosen), "")
                or fallback_model
            )
            key = key or os.getenv(chosen, "")
        else:
            key = key or os.getenv(chosen, "")

        super().__init__(
            api_key=key or "",
            base_url=base_url,
            model=default_model,
        )
        self.name = f"free_tier:{chosen}"
        self.provider = chosen


def available() -> list[str]:
    """Which configured free-tier providers have a key present, for diagnostics."""
    return [name for name in sorted(FREE_TIER_PROVIDERS) if os.getenv(name)]


__all__ = [
    "FreeTierClient",
    "FREE_TIER_PROVIDERS",
    "available",
    "model_env_var",
]

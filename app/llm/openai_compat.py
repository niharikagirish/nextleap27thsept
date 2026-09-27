"""OpenAI-compatible ``/chat/completions`` adapter.

Covers Groq, Together, OpenRouter, a local Ollama, LM Studio, vLLM, llama.cpp's
server, and anything else that speaks the OpenAI wire format. One adapter, many
backends — the base URL is the only thing that changes.

Implemented with ``httpx`` rather than the ``openai`` SDK on purpose: the SDK
pins a provider-shaped interface, pulls a dependency, and offers nothing here
that a POST does not. A 20-line request against a documented HTTP endpoint is
also far easier to debug when a demo fails on a locked-down network.

Configuration comes from the environment, never ``config.yaml``:

* ``LLM_BASE_URL`` — e.g. ``https://api.groq.com/openai/v1``. Optional for the
  real OpenAI endpoint.
* ``LLM_API_KEY``  — required.
* ``LLM_MODEL``    — required; each provider names its models differently.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
TIMEOUT_S = 30.0


class OpenAICompatClient:
    """Any endpoint implementing ``POST {base_url}/chat/completions``."""

    name = "openai_compat"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = TIMEOUT_S,
    ) -> None:
        if not api_key:
            raise LLMError(
                "no API key. Set LLM_API_KEY in .env, or unset it to fall back "
                "to the retrieval-only echo backend."
            )
        self.api_key = api_key
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or ""
        self.timeout_s = float(timeout_s)
        if not self.model:
            raise LLMError(
                "LLM_MODEL is not set. Each provider names its models differently "
                "(e.g. Groq: llama-3.3-70b-versatile). Add it to .env."
            )

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 250,
    ) -> str:
        import httpx

        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        try:
            response = httpx.post(url, json=payload, headers=headers,
                                  timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"could not reach {url}: {exc}",
                code="llm_unreachable",
                remedy="check network access and LLM_BASE_URL. Unset LLM_API_KEY "
                       "to use the retrieval-only echo backend instead.",
            ) from exc

        if response.status_code == 401:
            raise LLMError(
                f"{self.base_url} rejected the API key (401).",
                code="llm_auth_failed",
                remedy="the API key was rejected. Check LLM_API_KEY (or "
                       "GROQ_API_KEY / GEMINI_API_KEY).",
            )
        if response.status_code == 429:
            raise LLMError(
                f"rate limited by {self.base_url} (429).",
                code="llm_rate_limited",
                remedy="the free tier is throttling. Wait a few seconds and "
                       "retry; the key and model are fine.",
            )
        if response.status_code in (403, 404):
            # 404 is the common one on Groq: a retired or misspelled model name
            # is reported as "does not exist or you do not have access to it",
            # which reads like a permissions problem and sends people hunting
            # through the wrong key.
            raise LLMError(
                f"{self.base_url} returned {response.status_code}: "
                f"{response.text[:300]}",
                code="llm_model_not_found",
                remedy=f"the provider does not recognise model "
                       f"'{self.model}'. Set it to a current model id.",
            )
        if response.status_code >= 400:
            raise LLMError(
                f"{self.base_url} returned {response.status_code}: "
                f"{response.text[:300]}",
                code="llm_bad_request" if response.status_code < 500
                     else "llm_server_error",
                remedy="the provider rejected the request. Re-run with --debug "
                       "for the full response.",
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"unexpected response shape from {self.base_url}: "
                f"{response.text[:300]}",
                code="llm_bad_response",
                remedy="the provider replied in an unexpected shape. Re-run "
                       "with --debug, or switch model.",
            ) from exc

        if not isinstance(content, str):
            raise LLMError(
                f"expected a text completion, got {type(content).__name__}.",
                code="llm_unsupported",
                remedy="this model does not return text at this endpoint. "
                       "Pick a chat-completions model.",
            )
        return content


__all__ = ["OpenAICompatClient", "DEFAULT_BASE_URL"]

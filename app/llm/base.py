"""The LLM interface (architecture.md §3, ADR A7).

One narrow method, so the backend can change without touching a line of the
pipeline. The orchestrator only ever sees an :class:`LLMClient`; whether the
answer came from Groq, a local Ollama, or the ``echo`` retrieval-only stub is
invisible above this line. That is the whole point of the abstraction: PRD Q1
(left the backend undecided) should not have been a blocker, and with this seam
in place it never is.

Every adapter must honour the same contract:

* ``temperature`` and ``max_tokens`` are passed through, never overridden;
* the returned string is raw model output — **JSON parsing is not this layer's
  job**. ``app/pipeline/generator.py`` owns that, so a backend that returns prose
  gets the same single retry as any other.
* failures raise :class:`LLMError` with a remedy attached, never a bare stack
  trace, because the most likely failure on demo day is a missing or exhausted
  key.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


class LLMError(RuntimeError):
    """The LLM could not be reached or refused the request, with a remedy.

    ``code`` exists because "it didn't work" is not an actionable symptom. A dead
    model name, a bad key, a rate limit and a dead network need four different
    fixes, and the caller only saw one flat string. Callers that want to branch
    (the orchestrator setting ``reason``, the CLI choosing what to advise) should
    use ``code``; ``str(exc)`` stays the human-readable detail, and ``remedy`` is
    the short actionable line safe to show a user who has not passed --debug.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "llm_unavailable",
        remedy: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.remedy = remedy


class ParseError(RuntimeError):
    """Model output could not be parsed as the required JSON, after a retry."""


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat interface."""

    name: str

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 250,
    ) -> str:
        """Return the assistant's raw text for ``messages``."""
        ...


__all__ = ["LLMClient", "LLMError", "ParseError"]

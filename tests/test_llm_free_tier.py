"""Free-tier backend selection tests (ADR A7/A9).

The bug this file exists to prevent: a user sets ``GROQ_MODEL`` to a specific
model, the app runs happily against a *different* hardcoded one, and nothing
anywhere says so. A wrong-but-valid model is the worst failure mode here, because
every other signal - latency, cost, a 200 response - looks perfectly healthy.

So the assertions are about which model gets *chosen*, not about whether a
request succeeds. No network, no key, no dependency on the developer's real
``.env``: the environment is rebuilt per test so a locally-configured key cannot
change what these tests mean.
"""

from __future__ import annotations

import pytest

from app.llm import free_tier
from app.llm.base import LLMError
from app.llm.free_tier import FREE_TIER_PROVIDERS, FreeTierClient, model_env_var

_LLM_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from the real environment and the developer's ``.env``.

    ``.env`` has already been loaded into ``os.environ`` by ``app.config`` at
    import time, so simply not setting a var is not enough - a real
    ``GROQ_API_KEY`` on the machine would otherwise decide these outcomes.
    """
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# model_env_var - the name mapping
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("key_env", "expected"),
    [
        ("GROQ_API_KEY", "GROQ_MODEL"),
        ("GEMINI_API_KEY", "GEMINI_MODEL"),
    ],
)
def test_model_env_var_maps_key_to_model(key_env: str, expected: str) -> None:
    assert model_env_var(key_env) == expected


def test_model_env_var_handles_unknown_provider() -> None:
    """A provider not in the table still gets a predictable name.

    Guards the extension path: adding an entry to ``FREE_TIER_PROVIDERS`` must be
    the only change needed to make its model configurable.
    """
    assert model_env_var("SOME_NEW_API_KEY") == "SOME_NEW_MODEL"
    assert model_env_var("BARE_NAME") == "BARE_NAME_MODEL"


# --------------------------------------------------------------------------- #
# Model precedence: argument > <PROVIDER>_MODEL > hardcoded default
# --------------------------------------------------------------------------- #

def test_provider_model_env_var_wins_over_hardcoded_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression test: a pinned model must not be silently replaced."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")

    client = FreeTierClient()

    assert client.model == "llama-3.1-8b-instant"
    assert client.model != FREE_TIER_PROVIDERS["GROQ_API_KEY"][1]


def test_explicit_argument_wins_over_provider_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "from-env")

    client = FreeTierClient(model="from-argument")

    assert client.model == "from-argument"


def test_falls_back_to_hardcoded_default_when_env_var_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    client = FreeTierClient()

    assert client.model == FREE_TIER_PROVIDERS["GROQ_API_KEY"][1]


def test_blank_provider_model_env_var_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``GROQ_MODEL=`` line must not resolve to a model named "".

    ``.env`` files routinely ship keys with blank values, and an empty string is
    falsy but still "present" - treating it as a value would produce a request
    with ``"model": ""`` and a confusing 400 from the provider.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "")

    client = FreeTierClient()

    assert client.model == FREE_TIER_PROVIDERS["GROQ_API_KEY"][1]


def test_generic_llm_model_loses_to_provider_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents the one genuinely surprising interaction.

    ``LLM_MODEL`` is the documented way to pin a model on the generic
    OpenAI-compatible path, but it does **not** apply on the free-tier path: the
    provider table supplies a model first, so a stray ``LLM_MODEL`` left over from
    another provider is ignored instead of being sent to Groq, where it would 400.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "some-other-providers-model")

    client = FreeTierClient()

    assert client.model == FREE_TIER_PROVIDERS["GROQ_API_KEY"][1]


# --------------------------------------------------------------------------- #
# Provider selection and base URL
# --------------------------------------------------------------------------- #

def test_sniffs_first_provider_with_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "some-model")

    client = FreeTierClient()

    assert client.provider == "GROQ_API_KEY"
    assert client.base_url == FREE_TIER_PROVIDERS["GROQ_API_KEY"][0]
    assert client.model == "some-model"


def test_gemini_model_env_var_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")

    client = FreeTierClient()

    assert client.provider == "GEMINI_API_KEY"
    assert client.base_url == FREE_TIER_PROVIDERS["GEMINI_API_KEY"][0]
    assert client.model == "gemini-custom"


def test_explicit_provider_argument_is_respected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LLM_PROVIDER`` picks the winner when both keys are set."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "GEMINI_API_KEY")

    client = FreeTierClient()

    assert client.provider == "GEMINI_API_KEY"


def test_raises_actionable_error_with_no_key() -> None:
    with pytest.raises(LLMError) as excinfo:
        FreeTierClient()

    message = str(excinfo.value)
    assert "GROQ_API_KEY" in message
    assert "GEMINI_API_KEY" in message
    # The remedy must name the file to edit, not just the variables.
    assert ".env" in message


def test_available_lists_only_providers_with_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert free_tier.available() == []

    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    assert free_tier.available() == ["GROQ_API_KEY"]

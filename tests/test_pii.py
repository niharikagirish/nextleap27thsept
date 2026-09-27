"""PII guard tests (FR-10, FR-11, C3) — Phase 6.

The load-bearing assertions here are the **negative controls**. A PII guard that
catches everything is trivially easy to write and useless in a demo: if
"What is the minimum SIP amount, Rs 500?" gets blocked, the bot looks broken and
a reviewer will assume the guard is a stub. So ``pii-benign-*`` probes are
asserted to pass clean.

The other invariant asserted throughout: ``scan`` returns **category names only**.
Several tests assert the matched value never appears in the result, because a
guard that echoes what it matched is a second copy of the secret.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import disclaimers
from app.guards import pii

PROBE_PATH = Path(__file__).resolve().parent.parent / "eval" / "pii_probes.json"


def _probes() -> list[dict]:
    return json.loads(PROBE_PATH.read_text(encoding="utf-8"))["probes"]


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("My PAN is ABCDE1234F", ["pan"]),
        ("pan: abcde1234f", ["pan"]),                    # lowercase
        ("PAN ABC DE 1234 F", ["pan"]),                   # spaced formatting
        ("Aadhaar 9999 9999 9999", ["aadhaar"]),
        ("aadhaar is 999999999999", ["aadhaar"]),
        ("account number 123456789012", ["account"]),
        ("call 9999999999", ["phone"]),
        ("+91 99999 99999", ["phone"]),
        ("mail me at test@example.com", ["email"]),
        ("the OTP is 482913", ["otp"]),
        ("my ifsc is HDFC0123456", ["ifsc"]),
    ],
)
def test_detects_each_category(text: str, expected: list[str]) -> None:
    assert pii.scan(text) == expected


def test_specific_category_wins_over_overlapping_generic_one() -> None:
    """A PAN is 10 chars of digits; the report must not also call it an account."""
    assert pii.scan("ABCDE1234F") == ["pan"]
    # Aadhaar (14 digits) contains a 10-digit run; only the specific label counts.
    assert pii.scan("9999 9999 9999") == ["aadhaar"]
    # A 10-digit number starting 6-9 matches both account and phone patterns.
    assert pii.scan("9876543210") == ["account"]


def test_otp_with_separator_is_detected() -> None:
    assert "otp" in pii.scan("your one time password is 482913")


def test_multiple_categories_are_all_reported() -> None:
    found = pii.scan("PAN ABCDE1234F and phone 9999999999")
    assert set(found) == {"pan", "phone"}


# --------------------------------------------------------------------------- #
# The core security invariant: names, never values
# --------------------------------------------------------------------------- #


def test_scan_never_returns_the_matched_value() -> None:
    secret = "ABCDE1234F"
    result = pii.scan(f"My PAN is {secret}")
    assert secret not in json.dumps(result)
    assert all(not item[0].isdigit() for item in result)


def test_categories_string_carries_no_values() -> None:
    text = "PAN ABCDE1234F, account 123456789012, email a@b.com"
    joined = pii.categories(text)
    for secret in ("ABCDE1234F", "123456789012", "a@b.com"):
        assert secret not in joined


def test_describe_categories_is_human_readable() -> None:
    described = pii.describe_categories(["pan", "otp"])
    assert "PAN" in described and "OTP" in described


# --------------------------------------------------------------------------- #
# Negative controls - the guard must not over-block
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question",
    [
        "What is the expense ratio of the HDFC Large Cap Fund?",
        "What is the minimum SIP amount, is it Rs 500?",
        "What is the lock-in period of HDFC ELSS Tax Saver Fund?",
        "What is the exit load after 1 year?",
        "Who manages the HDFC Flexi Cap Fund?",
        "What is the benchmark of HDFC Balanced Advantage Fund?",
    ],
)
def test_benign_questions_are_not_blocked(question: str) -> None:
    assert pii.contains_pii(question) is False


def test_three_digit_amount_does_not_trip_the_account_rule() -> None:
    """The specific regression: \\b[0-9]{10,}\\b vs 'Rs 500' and '1 year'."""
    assert "account" not in pii.scan("The minimum SIP is Rs 500 per month")
    assert "account" not in pii.scan("Exit load is 1% after 1 year")


def test_empty_and_none_safe() -> None:
    assert pii.scan("") == []
    assert pii.contains_pii("") is False


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_redact_replaces_the_value_but_keeps_the_category() -> None:
    redacted, hits = pii.redact("Your PAN is ABCDE1234F, thanks.")
    assert hits == ["pan"]
    assert "ABCDE1234F" not in redacted
    assert "[REDACTED:pan]" in redacted


def test_redact_leaves_clean_text_untouched() -> None:
    text = "The expense ratio is 0.87%."
    redacted, hits = pii.redact(text)
    assert redacted == text and hits == []


# --------------------------------------------------------------------------- #
# Count-only metric (C3)
# --------------------------------------------------------------------------- #


def test_block_counter_is_count_only() -> None:
    pii.reset_counter()
    assert pii.blocks_total() == 0
    pii.note_block()
    pii.note_block(2)
    assert pii.blocks_total() == 3
    pii.reset_counter()
    assert pii.blocks_total() == 0


def test_counter_value_never_contains_a_secret() -> None:
    pii.reset_counter()
    total = pii.note_block()
    assert isinstance(total, int)
    assert "ABCDE1234F" not in str(total)


# --------------------------------------------------------------------------- #
# Probe file drives the suite
# --------------------------------------------------------------------------- #


def test_probe_file_is_wellformed() -> None:
    probes = _probes()
    assert len(probes) >= 10
    seen = set()
    for probe in probes:
        assert {"id", "question", "expect_kind", "pii_categories"} <= set(probe)
        assert probe["id"] not in seen, f"duplicate probe id {probe['id']}"
        seen.add(probe["id"])


@pytest.mark.parametrize("probe", _probes(), ids=lambda p: p["id"])
def test_probe_expectations_hold(probe: dict) -> None:
    found = pii.scan(probe["question"])
    assert found == probe["pii_categories"], (
        f"{probe['id']}: expected {probe['pii_categories']}, scan found {found}"
    )
    must_not_echo = probe.get("must_not_echo")
    if must_not_echo:
        assert must_not_echo not in disclaimers.PII_NOTICE


def test_every_blocking_probe_expectation_is_pii_notice() -> None:
    for probe in _probes():
        if probe["pii_categories"]:
            assert probe["expect_kind"] == "pii_notice"


def test_negative_controls_are_present_and_clean() -> None:
    """Without these the suite proves nothing about precision."""
    controls = [p for p in _probes() if p["id"].startswith("pii-benign")]
    assert len(controls) >= 3, "need negative controls to assert the guard's precision"
    for control in controls:
        assert pii.contains_pii(control["question"]) is False

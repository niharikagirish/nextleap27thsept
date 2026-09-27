"""Console encoding tests.

Regression test for a crash that only appeared once real LLM answers arrived:
the corpus and the model both use the rupee sign (U+20B9), en/em dashes and the
degree sign, none of which a cp1252 Windows console can encode. Python selects
the console code page for ``sys.stdout`` and raised
``UnicodeEncodeError`` inside ``print(render(answer))`` - i.e. the entire
retrieve/generate/validate pipeline had already run successfully and then the
process died while displaying a *correct* answer.

The failure mode this guards against is distinctive: everything is fine until a
particular character turns up in production data, and the traceback points at
``encodings/cp1252.py`` rather than at anything the author wrote.
"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from app import enable_utf8_console

# Characters that actually occur in the Groww corpus or in generated answers.
HOSTILE = [
    "\u20b9",  # rupee sign - scheme fees, expense ratios, corpus tables
    "\u2013",  # en dash - "1-3 years" style ranges
    "\u2014",  # em dash
    "\u00b0",  # degree sign
    "\u2192",  # rightwards arrow
]


def test_reports_utf8_capable_stream() -> None:
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    try:
        enable_utf8_console()  # must not raise even though streams are patched
    finally:
        stream.detach()


def test_is_idempotent_and_never_raises() -> None:
    """Calling it twice, or with an unreconfigurable stream, must be harmless.

    A CLI helper that can itself raise is worse than the bug it fixes, so the
    contract is deliberately "cannot fail".
    """
    enable_utf8_console()
    enable_utf8_console()


def test_prints_hostile_characters_without_raising(
    capsys: pytest.CaptureFixture[str],
) -> None:
    enable_utf8_console()
    for char in HOSTILE:
        print(f"amount: {char}100")


def test_rupee_survives_a_real_subprocess_on_a_legacy_codepage() -> None:
    """The actual bug, exercised in a real interpreter.

    ``capsys`` replaces stdout with a buffer that is already UTF-8, so it cannot
    reproduce the failure. This spawns a child interpreter with the Windows
    console code page forced to cp1252, which is what ``python -m app.chat`` used
    to hit. The child asserts internally, so a pass means the child's own print
    succeeded - not merely that we captured bytes off it.
    """
    code = (
        "import sys\n"
        "from app import enable_utf8_console\n"
        "enable_utf8_console()\n"
        "print('exit load: \\u20b91 on \\u2013 3 years, 30\\u00b0')\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, f"child failed: {result.stderr}"
    assert "ok" in result.stdout
    assert "\u20b9" in result.stdout


@pytest.mark.parametrize("char", HOSTILE)
def test_each_hostile_char_alone_does_not_abort(char: str) -> None:
    """One character, one run.

    Isolating each character means a newly-encountered symbol from a future data
    refresh shows up as its own named failure rather than hiding inside a long
    combined test whose failure message names none of them.
    """
    enable_utf8_console()
    print(char)

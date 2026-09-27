"""HDFC Mutual Fund FAQ RAG Chatbot."""

from __future__ import annotations

import sys

__version__ = "1.0.0"


def enable_utf8_console() -> None:
    """Make ``print`` of non-ASCII text safe on a legacy Windows console.

    The corpus is full of characters a cp1252 terminal cannot represent at all -
    the rupee sign (U+20B9), en/em dashes, the degree sign. Python picks the
    console code page for ``sys.stdout`` and then raises
    ``UnicodeEncodeError`` on the first one it meets, so a correct answer
    containing a single rupee sign crashed the CLI *after* the whole pipeline had
    already run and validated successfully.

    Reconfiguring to UTF-8 with ``errors="replace"`` fixes the real text and
    guarantees no character can ever abort a run again: anything still
    unrepresentable degrades to a placeholder instead of a traceback.

    Called at the top of each ``main()`` rather than at import time, so importing
    the package stays side-effect free for tests and the Streamlit app - both of
    which run under a UTF-8 environment already and must not have their stdout
    reconfigured underneath them.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            # Captured by pytest's capsys / StringIO, which has no reconfigure.
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # A stream that refuses reconfiguration is not worth failing over.
            pass


__all__ = ["enable_utf8_console", "__version__"]

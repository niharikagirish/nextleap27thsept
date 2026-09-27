"""RAG pipeline stages.

One module per stage (constraint C10). Phases 1-4 provide ``loader``,
``chunker``, ``embedder``, ``store`` and ``retriever``; Phase 5 adds ``prompt``,
``generator`` and ``orchestrator``; Phase 6 adds ``validators``.

Modules in this package may import ``app.models`` and ``app.config``, plus
stdlib and third-party packages. ``orchestrator`` and ``validators`` additionally
import ``app.guards.*``; the dependency is one-way, because the guards import
nothing from here. See ``app/guards/__init__.py`` for the reasoning.

The other deliberate exception is ``retriever`` -> ``store``. Retrieval is the
*client* of the store, not a second stage that re-implements it, and it touches
only the store's public API (``get_collection``, ``from_metadata``,
``chunk_id_for``). ``search(..., collection=...)`` accepts an injected
collection, so the retrieval logic is still testable with no disk and no
ChromaDB, which is what C10 asks for.
"""

from . import (  # noqa: F401
    chunker,
    embedder,
    generator,
    loader,
    orchestrator,
    prompt,
    retriever,
    store,
    validators,
)

__all__ = [
    "loader", "chunker", "embedder", "store", "retriever",
    "prompt", "generator", "orchestrator", "validators",
]

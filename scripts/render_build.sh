#!/usr/bin/env bash
# Build the retrieval index from the committed artifacts.
#
# Why this exists as a build step: `chroma_db/` and `cache/` are gitignored, so a
# fresh clone has no index at all. Without this step the deployed app starts,
# loads, and then answers every question with "the retrieval index is empty".
#
# `--offline` replays the committed `artifacts/raw_docs.jsonl` instead of
# re-fetching the five Groww pages. That makes the build deterministic and
# hermetic: a Groww redesign, an outage, or a rate limit cannot produce a
# different index on the server than the one the retrieval report was measured
# against. To deliberately re-scrape, run `python -m app.ingest --stage all`
# (no --offline) and commit the refreshed artifacts.
#
# Cold build is ~30 s of compute plus the one-time ~90 MB model download.
set -euo pipefail

echo "==> Installing CPU-only torch first"
# The default PyPI torch wheel drags in multi-GB CUDA dependencies, which
# breaches the size budget and can exhaust a Render build's memory. Installing
# from the CPU index first means pip never resolves the CUDA extra.
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing project requirements"
pip install --no-cache-dir -r requirements.txt

echo "==> Building the retrieval index (offline replay, no network)"
# --reset drops and recreates the collection. Without it, an upsert into an
# existing HNSW index can leave stale links behind, and the failure surfaces
# later as a query-time
#   RuntimeError: Cannot return the results in a contigious 2D array
# from inside hnswlib — long after the build looked like it succeeded.
# A clean build must start from a clean index.
python -m app.ingest --stage all --offline --reset

echo "==> Pre-warming the embedding model into the image"
# Pulls all-MiniLM-L6-v2 into the build image so the first user request does not
# pay the download.
python -c "from app.embedder_singleton import get_model; get_model()"

echo "==> Verifying the index is queryable"
python - <<'PY'
from app.pipeline.store import count
n = count()
assert n > 0, "index is empty - the build did not produce a usable corpus"
print(f"index ready: {n} chunks")
PY

# The start command is the one thing the build cannot execute for real, because
# it needs a port and a long-lived process. So check the part that is checkable:
# that the interpreter which ran this script can import streamlit, and report
# where its console script went. A deploy that dies at startup with
# "streamlit: command not found" is entirely preventable here, and this printout
# is the first thing to read if it ever happens again.
echo "==> Verifying the start command's entry point"
python - <<'PY'
import shutil
import sys

import streamlit

print(f"interpreter      : {sys.executable}")
print(f"streamlit module : {streamlit.__file__}")
print(f"streamlit version: {streamlit.__version__}")

script = shutil.which("streamlit")
print(f"'streamlit' on PATH: {script or 'NO'}")
if not script:
    print(
        "  note: the console script is not on PATH, which is why the start "
        "command uses 'python -m streamlit' rather than bare 'streamlit'."
    )
PY

echo "==> Build complete"

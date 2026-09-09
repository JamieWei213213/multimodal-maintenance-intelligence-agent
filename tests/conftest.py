from pathlib import Path

import pytest

from maintenance_intelligence.retrieval import load_chunks

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CHUNKS = Path(__file__).resolve().parent / "fixtures" / "sample_chunks.jsonl"
CORPUS_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"


@pytest.fixture(scope="session")
def fixture_chunks():
    """Small synthetic corpus that mirrors the real chunk schema.

    The text is paraphrased fixture content, not excerpts from the ABB manuals, so
    it can be versioned and run in CI without the gitignored ``data/processed``
    directory.
    """
    return load_chunks(FIXTURE_CHUNKS)


@pytest.fixture(scope="session")
def corpus_chunks():
    """Full ingested corpus; only available after ``maintenance-ingest`` has run."""
    if not CORPUS_CHUNKS.exists():
        pytest.skip("Full corpus not ingested (run maintenance-ingest); fixture tests still cover this path")
    return load_chunks(CORPUS_CHUNKS)

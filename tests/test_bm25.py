from pathlib import Path

from maintenance_intelligence.retrieval import BM25Retriever, load_chunks


def test_bm25_returns_unique_pages_with_positive_scores():
    chunks = load_chunks(Path("data/processed/chunks.jsonl"))
    results = BM25Retriever(chunks).search("A2B1 overcurrent motor cable", limit=5)

    assert results
    assert all(result.score > 0 for result in results)
    citations = [(result.chunk.source_filename, result.chunk.pdf_page) for result in results]
    assert len(citations) == len(set(citations))
    assert any("A2B1" in result.chunk.text for result in results)


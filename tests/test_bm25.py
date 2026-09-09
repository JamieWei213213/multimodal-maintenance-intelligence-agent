from maintenance_intelligence.retrieval import BM25Retriever


def _assert_unique_pages_with_positive_scores(results):
    assert results
    assert all(result.score > 0 for result in results)
    citations = [(result.chunk.source_filename, result.chunk.pdf_page) for result in results]
    assert len(citations) == len(set(citations))
    assert any("A2B1" in result.chunk.text for result in results)


def test_bm25_returns_unique_pages_with_positive_scores(fixture_chunks):
    results = BM25Retriever(fixture_chunks).search("A2B1 overcurrent motor cable", limit=5)
    _assert_unique_pages_with_positive_scores(results)


def test_bm25_ignores_unrelated_pages(fixture_chunks):
    results = BM25Retriever(fixture_chunks).search("A2B1", limit=5)
    assert all("A2B1" in result.chunk.text for result in results)


def test_bm25_on_full_corpus(corpus_chunks):
    """Regression check against the real ingested corpus; skipped when it is absent."""
    results = BM25Retriever(corpus_chunks).search("A2B1 overcurrent motor cable", limit=5)
    _assert_unique_pages_with_positive_scores(results)

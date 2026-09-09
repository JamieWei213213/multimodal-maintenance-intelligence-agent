from maintenance_intelligence.retrieval import (
    KeywordRetriever,
    is_technical_identifier,
    tokenize,
)

ACCEPTED_A2B1_CITATIONS = {
    ("abb_acs580_01_quickstart_3AXD50001317799_revB.pdf", 22),
    ("abb_acs580_firmware_3AXD50000016097_revJ.pdf", 538),
}


def test_technical_identifier_detection_preserves_mixed_codes():
    assert is_technical_identifier("a5a0")
    assert is_technical_identifier("ip55")
    assert not is_technical_identifier("warning")
    assert not is_technical_identifier("5091")


def _assert_a2b1_evidence(results):
    assert results
    assert "A2B1" in results[0].chunk.text
    assert (results[0].chunk.source_filename, results[0].chunk.pdf_page) in ACCEPTED_A2B1_CITATIONS


def _assert_pages_deduplicated(results):
    citations = [(result.chunk.source_filename, result.chunk.pdf_page) for result in results]
    assert len(citations) == len(set(citations))


def test_keyword_search_returns_a2b1_evidence(fixture_chunks):
    _assert_a2b1_evidence(KeywordRetriever(fixture_chunks).search("A2B1", limit=5))


def test_keyword_search_deduplicates_document_pages(fixture_chunks):
    # The fixture has two A2B1 chunks on quickstart page 22 and two on firmware page 538;
    # only one result per (document, page) may be returned.
    results = KeywordRetriever(fixture_chunks).search("A2B1", limit=5)
    _assert_pages_deduplicated(results)
    assert len(results) == 2


def test_keyword_search_on_full_corpus(corpus_chunks):
    """Regression check against the real ingested corpus; skipped when it is absent."""
    results = KeywordRetriever(corpus_chunks).search("A2B1", limit=5)
    _assert_a2b1_evidence(results)
    _assert_pages_deduplicated(results)


def test_tokenize_removes_stop_words_and_preserves_fault_codes():
    tokens = tokenize("What should the technician check for warning A2B1?")

    assert tokens == ["technician", "check", "warning", "a2b1"]

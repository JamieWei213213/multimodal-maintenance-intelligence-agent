from pathlib import Path

from maintenance_intelligence.retrieval import (
    KeywordRetriever,
    is_technical_identifier,
    load_chunks,
    tokenize,
)


def test_technical_identifier_detection_preserves_mixed_codes():
    assert is_technical_identifier("a5a0")
    assert is_technical_identifier("ip55")
    assert not is_technical_identifier("warning")
    assert not is_technical_identifier("5091")


def test_keyword_search_returns_a2b1_evidence():
    chunks = load_chunks(Path("data/processed/chunks.jsonl"))
    retriever = KeywordRetriever(chunks)

    results = retriever.search("A2B1", limit=5)

    accepted_citations = {
        (
            "abb_acs580_01_quickstart_3AXD50001317799_revB.pdf",
            22,
        ),
        (
            "abb_acs580_firmware_3AXD50000016097_revJ.pdf",
            538,
        ),
    }

    assert results
    assert "A2B1" in results[0].chunk.text
    assert (
        results[0].chunk.source_filename,
        results[0].chunk.pdf_page,
    ) in accepted_citations


def test_keyword_search_deduplicates_document_pages():
    chunks = load_chunks(Path("data/processed/chunks.jsonl"))
    retriever = KeywordRetriever(chunks)

    results = retriever.search("A2B1", limit=5)

    citations = [
        (result.chunk.source_filename, result.chunk.pdf_page)
        for result in results
    ]

    assert len(citations) == len(set(citations))


def test_tokenize_removes_stop_words_and_preserves_fault_codes():
    tokens = tokenize(
        "What should the technician check for warning A2B1?"
    )

    assert tokens == [
        "technician",
        "check",
        "warning",
        "a2b1",
    ]

from maintenance_intelligence.chunking import chunk_page
from maintenance_intelligence.models import Block, PageRecord


def make_page() -> PageRecord:
    return PageRecord(
        document_id="doc_abc",
        source_filename="manual.pdf",
        source_sha256="ABC",
        pdf_page=24,
        printed_page="24",
        width=600,
        height=800,
        rotation=0,
        extraction_method="native",
        text_char_count=120,
        blocks=[
            Block("b1", "heading", "Electrical safety", [10, 10, 200, 30], 16, "Bold", True, 1),
            Block("b2", "text", "WARNING Injury or death can occur. Wait five minutes before continuing.", [10, 40, 500, 80]),
        ],
    )


def test_chunk_preserves_page_provenance_and_safety_warning() -> None:
    chunks = chunk_page(make_page(), max_chars=200, overlap_chars=20)
    assert len(chunks) == 1
    assert chunks[0].pdf_page == 24
    assert chunks[0].printed_page == "24"
    assert chunks[0].section_path == ["Electrical safety"]
    assert chunks[0].block_ids == ["b2"]
    assert chunks[0].contains_safety_warning is True


def test_chunk_ids_are_deterministic() -> None:
    first = chunk_page(make_page())[0].chunk_id
    second = chunk_page(make_page())[0].chunk_id
    assert first == second


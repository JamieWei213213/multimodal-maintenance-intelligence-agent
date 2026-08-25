from maintenance_intelligence.citations import make_citation, validate_citation
from maintenance_intelligence.answering import GroundedAnswerer, _sentences
from maintenance_intelligence.hybrid import HybridResult
from maintenance_intelligence.models import ChunkRecord
from maintenance_intelligence.retrieval import SearchResult


def sample_chunk() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chk_test",
        document_id="doc_test",
        source_filename="manual.pdf",
        source_sha256="ABC",
        pdf_page=24,
        printed_page="24",
        chunk_index=0,
        kind="text",
        section_path=["Safety"],
        text="WARNING Wait five minutes before electrical work.",
        block_ids=["b1"],
        bboxes=[[10, 20, 300, 60]],
        extraction_method="native",
        contains_safety_warning=True,
    )


def test_citation_round_trip_is_valid():
    chunk = sample_chunk()
    citation = make_citation(chunk)
    assert validate_citation(citation, {chunk.chunk_id: chunk})


def test_changed_evidence_is_rejected():
    chunk = sample_chunk()
    citation = make_citation(chunk)
    changed = citation.__class__(**{**citation.__dict__, "evidence": "invented procedure"})
    assert not validate_citation(changed, {chunk.chunk_id: chunk})


def test_pdf_line_breaks_do_not_create_sentence_fragments():
    assert _sentences("Output current has\nexceeded the limit. Check\nthe motor cable.") == [
        "Output current has exceeded the limit.",
        "Check the motor cable.",
    ]


class _FakeHybrid:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks

    def search(self, query: str, limit: int = 5):
        return [
            HybridResult(SearchResult(chunk=chunk, score=1.0, rank=rank), 0.03, rank, rank)
            for rank, chunk in enumerate(self.chunks[:limit], start=1)
        ]


def test_procedure_terms_must_be_supported_together():
    bearing = sample_chunk().__class__(
        **{**sample_chunk().to_dict(), "chunk_id": "bearing", "text": "Use an insulated motor bearing."}
    )
    torque = sample_chunk().__class__(
        **{**sample_chunk().to_dict(), "chunk_id": "torque", "text": "Tighten a cover screw to the specified torque."}
    )
    chunks = {chunk.chunk_id: chunk for chunk in (bearing, torque)}
    response = GroundedAnswerer(_FakeHybrid([bearing, torque]), chunks).answer(
        "Give the bearing replacement procedure and tightening torque."
    )
    assert response["abstained"] is True


def test_adjacent_chunks_expand_context_without_crossing_pages():
    first = sample_chunk().__class__(
        **{
            **sample_chunk().to_dict(),
            "chunk_id": "first",
            "chunk_index": 0,
            "text": "A2B1 means overcurrent.",
        }
    )
    continuation = sample_chunk().__class__(
        **{
            **sample_chunk().to_dict(),
            "chunk_id": "continuation",
            "chunk_index": 1,
            "text": "Check the motor load and motor cable.",
        }
    )
    other_page = sample_chunk().__class__(
        **{
            **sample_chunk().to_dict(),
            "chunk_id": "other-page",
            "pdf_page": 25,
            "printed_page": "25",
            "chunk_index": 0,
            "text": "This text must not be included.",
        }
    )
    chunks = {chunk.chunk_id: chunk for chunk in (first, continuation, other_page)}
    response = GroundedAnswerer(_FakeHybrid([first]), chunks, adjacent_chunk_radius=1).answer(
        "What does A2B1 mean and what should I check?"
    )
    assert "Check the motor load and motor cable" in response["answer"]
    assert "must not be included" not in response["answer"]
    assert {citation["chunk_id"] for citation in response["citations"]} == {
        "first",
        "continuation",
    }


def test_site_specific_network_request_abstains_without_joint_support():
    network = sample_chunk().__class__(
        **{
            **sample_chunk().to_dict(),
            "chunk_id": "network",
            "text": "The fieldbus adapter communicates with the drive.",
        }
    )
    chunks = {network.chunk_id: network}
    response = GroundedAnswerer(_FakeHybrid([network]), chunks).answer(
        "What static IP address should be assigned to this site?"
    )
    assert response["abstained"] is True

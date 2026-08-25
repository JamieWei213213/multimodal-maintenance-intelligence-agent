from __future__ import annotations

from dataclasses import dataclass

from maintenance_intelligence.models import ChunkRecord


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source_filename: str
    pdf_page: int
    printed_page: str
    evidence: str
    bboxes: list[list[float]]


def make_citation(chunk: ChunkRecord, *, max_chars: int = 700) -> Citation:
    evidence = chunk.text.strip()[:max_chars]
    return Citation(
        chunk_id=chunk.chunk_id,
        source_filename=chunk.source_filename,
        pdf_page=chunk.pdf_page,
        printed_page=chunk.printed_page,
        evidence=evidence,
        bboxes=chunk.bboxes,
    )


def validate_citation(citation: Citation, chunks_by_id: dict[str, ChunkRecord]) -> bool:
    chunk = chunks_by_id.get(citation.chunk_id)
    if chunk is None:
        return False
    return (
        citation.source_filename == chunk.source_filename
        and citation.pdf_page == chunk.pdf_page
        and citation.printed_page == chunk.printed_page
        and bool(citation.evidence)
        and citation.evidence in chunk.text
    )


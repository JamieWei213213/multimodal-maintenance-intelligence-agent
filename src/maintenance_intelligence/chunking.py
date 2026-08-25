from __future__ import annotations

import hashlib
import re

from .extract import SAFETY_PATTERN
from .models import Block, ChunkRecord, PageRecord


def _chunk_id(document_id: str, pdf_page: int, chunk_index: int, text: str) -> str:
    payload = f"{document_id}|{pdf_page}|{chunk_index}|{text}".encode("utf-8")
    return f"chk_{hashlib.sha256(payload).hexdigest()[:24]}"


def _paragraphs(block: Block) -> list[str]:
    pieces = re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z0-9•])", block.text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > max_chars:
            parts.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += extra
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_page(page: PageRecord, *, max_chars: int = 1400, overlap_chars: int = 180) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    section_stack: list[str] = []
    pending: list[tuple[str, Block]] = []

    def emit(kind: str = "text") -> None:
        nonlocal pending
        if not pending:
            return
        text = "\n\n".join(piece for piece, _ in pending).strip()
        source_blocks = list(dict.fromkeys(block.block_id for _, block in pending))
        bboxes: list[list[float]] = []
        seen_bboxes: set[tuple[float, ...]] = set()
        for _, block in pending:
            key = tuple(block.bbox)
            if key not in seen_bboxes:
                seen_bboxes.add(key)
                bboxes.append(block.bbox)
        index = len(chunks)
        chunks.append(ChunkRecord(
            chunk_id=_chunk_id(page.document_id, page.pdf_page, index, text),
            document_id=page.document_id,
            source_filename=page.source_filename,
            source_sha256=page.source_sha256,
            pdf_page=page.pdf_page,
            printed_page=page.printed_page,
            chunk_index=index,
            kind=kind,
            section_path=list(section_stack),
            text=text,
            block_ids=source_blocks,
            bboxes=bboxes,
            extraction_method=page.extraction_method,
            contains_safety_warning=bool(SAFETY_PATTERN.search(text)),
        ))
        overlap: list[tuple[str, Block]] = []
        chars = 0
        for item in reversed(pending):
            if chars + len(item[0]) > overlap_chars:
                break
            overlap.insert(0, item)
            chars += len(item[0])
        pending = overlap

    for block in page.blocks:
        if block.is_heading:
            emit()
            level = block.heading_level or 3
            section_stack[:] = section_stack[: level - 1]
            section_stack.append(block.text.replace("\n", " "))
            continue
        if block.kind == "table":
            emit()
            for part in _split_long(block.text, max_chars):
                pending = [(part, block)]
                emit(kind="table")
            pending = []
            continue
        for paragraph in _paragraphs(block):
            for part in _split_long(paragraph, max_chars):
                proposed = sum(len(text) for text, _ in pending) + 2 * max(0, len(pending) - 1) + len(part)
                if pending and proposed > max_chars:
                    emit()
                pending.append((part, block))
    emit()
    return chunks

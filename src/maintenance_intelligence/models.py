from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Block:
    block_id: str
    kind: str
    text: str
    bbox: list[float]
    font_size: float | None = None
    font_name: str | None = None
    is_heading: bool = False
    heading_level: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageRecord:
    document_id: str
    source_filename: str
    source_sha256: str
    pdf_page: int
    printed_page: str
    width: float
    height: float
    rotation: int
    extraction_method: str
    text_char_count: int
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["blocks"] = [block.to_dict() for block in self.blocks]
        return result


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    source_filename: str
    source_sha256: str
    pdf_page: int
    printed_page: str
    chunk_index: int
    kind: str
    section_path: list[str]
    text: str
    block_ids: list[str]
    bboxes: list[list[float]]
    extraction_method: str
    contains_safety_warning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


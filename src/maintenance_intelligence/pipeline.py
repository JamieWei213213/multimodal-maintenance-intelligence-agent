from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pymupdf as fitz

from .chunking import chunk_page
from .extract import extract_document, render_page_image
from .hashing import validate_corpus


@dataclass(frozen=True)
class IngestionConfig:
    min_text_chars: int = 40
    ocr_dpi: int = 300
    max_chunk_chars: int = 1400
    overlap_chars: int = 180
    render_pages: bool = False
    render_dpi: int = 144


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def run_ingestion(
    manual_dir: Path,
    checksum_csv: Path,
    output_dir: Path,
    *,
    config: IngestionConfig | None = None,
) -> dict[str, object]:
    config = config or IngestionConfig()
    validated = validate_corpus(manual_dir, checksum_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "page_images"

    documents: list[dict[str, object]] = []
    page_records: list[dict[str, object]] = []
    chunk_records: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    ocr_pages = 0

    for source in validated:
        pdf_path = manual_dir / str(source["filename"])
        document, pages = extract_document(
            pdf_path,
            str(source["sha256"]),
            min_text_chars=config.min_text_chars,
            ocr_dpi=config.ocr_dpi,
        )
        documents.append(document)
        for page in pages:
            if page.extraction_method == "ocr":
                ocr_pages += 1
            if page.warnings:
                warnings.append({"source_filename": page.source_filename, "pdf_page": page.pdf_page, "warnings": page.warnings})
            page_records.append(page.to_dict())
            chunk_records.extend(
                chunk.to_dict()
                for chunk in chunk_page(page, max_chars=config.max_chunk_chars, overlap_chars=config.overlap_chars)
            )
            if config.render_pages:
                image_path = image_dir / page.document_id / f"page_{page.pdf_page:04d}.png"
                render_page_image(pdf_path, page.pdf_page, image_path, dpi=config.render_dpi)

    _write_jsonl(output_dir / "documents.jsonl", documents)
    _write_jsonl(output_dir / "pages.jsonl", page_records)
    _write_jsonl(output_dir / "chunks.jsonl", chunk_records)

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "pymupdf": fitz.VersionBind},
        "sources": validated,
        "counts": {
            "documents": len(documents),
            "pages": len(page_records),
            "chunks": len(chunk_records),
            "ocr_pages": ocr_pages,
            "warning_pages": len(warnings),
            "safety_chunks": sum(bool(chunk["contains_safety_warning"]) for chunk in chunk_records),
            "table_chunks": sum(chunk["kind"] == "table" for chunk in chunk_records),
        },
        "warnings": warnings,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest

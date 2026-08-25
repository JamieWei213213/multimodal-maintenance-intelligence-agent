from __future__ import annotations

import hashlib
import re
import statistics
from pathlib import Path

import pymupdf as fitz

from .models import Block, PageRecord


SAFETY_PATTERN = re.compile(r"\b(warning|danger|caution|electric shock|injury or death)\b", re.I)


def document_id(source_sha256: str) -> str:
    return f"doc_{source_sha256[:16].lower()}"


def _clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u0000", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _block_id(doc_id: str, pdf_page: int, kind: str, index: int, text: str) -> str:
    payload = f"{doc_id}|{pdf_page}|{kind}|{index}|{text}".encode("utf-8")
    return f"blk_{hashlib.sha256(payload).hexdigest()[:20]}"


def _text_blocks(page: fitz.Page, textpage: fitz.TextPage | None = None) -> list[Block]:
    data = page.get_text("dict", textpage=textpage, sort=True)
    candidates: list[dict[str, object]] = []
    body_sizes: list[float] = []
    for raw in data.get("blocks", []):
        if raw.get("type") != 0:
            continue
        lines = raw.get("lines", [])
        spans = [span for line in lines for span in line.get("spans", []) if span.get("text", "").strip()]
        text = _clean_text("\n".join("".join(span.get("text", "") for span in line.get("spans", [])) for line in lines))
        if not text or not spans:
            continue
        sizes = [float(span.get("size", 0.0)) for span in spans if span.get("size")]
        size = max(sizes) if sizes else None
        if size:
            body_sizes.extend(sizes)
        font_name = str(spans[0].get("font", "")) or None
        candidates.append({"text": text, "bbox": [round(float(v), 2) for v in raw["bbox"]], "size": size, "font": font_name})

    median_size = statistics.median(body_sizes) if body_sizes else 10.0
    blocks: list[Block] = []
    for index, item in enumerate(candidates):
        text = str(item["text"])
        size = item["size"]
        font = item["font"]
        bold = bool(font and any(token in str(font).lower() for token in ("bold", "black", "heavy")))
        short = len(text) <= 140 and text.count("\n") <= 2
        prose_like = text.endswith((".", "!", "?", ":"))
        warning_lead = bool(re.match(r"^(warning|danger|caution)\b", text, re.I))
        is_heading = short and not prose_like and not warning_lead and bool(
            (size and float(size) >= median_size * 1.16) or bold
        )
        level = None
        if is_heading and size:
            ratio = float(size) / median_size
            level = 1 if ratio >= 1.65 else 2 if ratio >= 1.35 else 3
        blocks.append(Block(
            block_id="",
            kind="heading" if is_heading else "text",
            text=text,
            bbox=item["bbox"],
            font_size=round(float(size), 2) if size else None,
            font_name=str(font) if font else None,
            is_heading=is_heading,
            heading_level=level,
        ))
    return blocks


def _table_blocks(page: fitz.Page) -> list[Block]:
    blocks: list[Block] = []
    try:
        finder = page.find_tables()
    except Exception:
        return blocks
    for table in finder.tables:
        rows = table.extract()
        clean_rows = [[_clean_text(cell or "") for cell in row] for row in rows]
        clean_rows = [row for row in clean_rows if any(row)]
        if not clean_rows:
            continue
        text = "\n".join(" | ".join(row) for row in clean_rows)
        blocks.append(Block(block_id="", kind="table", text=text, bbox=[round(float(v), 2) for v in table.bbox]))
    return blocks


def extract_page(
    page: fitz.Page,
    *,
    doc_id: str,
    filename: str,
    source_sha256: str,
    min_text_chars: int,
    ocr_dpi: int,
) -> PageRecord:
    warnings: list[str] = []
    extraction_method = "native"
    blocks = _text_blocks(page)
    native_text_chars = sum(len(block.text) for block in blocks)
    if native_text_chars < min_text_chars:
        try:
            textpage = page.get_textpage_ocr(dpi=ocr_dpi, full=True)
            blocks = _text_blocks(page, textpage=textpage)
            extraction_method = "ocr"
        except Exception as exc:
            extraction_method = "native_ocr_unavailable"
            warnings.append(f"OCR requested but unavailable: {type(exc).__name__}: {exc}")

    blocks.extend(_table_blocks(page))
    blocks.sort(key=lambda block: (round(block.bbox[1], 1), round(block.bbox[0], 1), block.kind))
    for index, block in enumerate(blocks):
        block.block_id = _block_id(doc_id, page.number + 1, block.kind, index, block.text)

    label = page.get_label() or str(page.number + 1)
    return PageRecord(
        document_id=doc_id,
        source_filename=filename,
        source_sha256=source_sha256,
        pdf_page=page.number + 1,
        printed_page=label,
        width=round(page.rect.width, 2),
        height=round(page.rect.height, 2),
        rotation=page.rotation,
        extraction_method=extraction_method,
        text_char_count=sum(len(block.text) for block in blocks if block.kind != "table"),
        blocks=blocks,
        warnings=warnings,
    )


def extract_document(
    path: Path,
    source_sha256: str,
    *,
    min_text_chars: int = 40,
    ocr_dpi: int = 300,
) -> tuple[dict[str, object], list[PageRecord]]:
    doc_id = document_id(source_sha256)
    with fitz.open(path) as pdf:
        metadata = {key: value for key, value in pdf.metadata.items() if value}
        pages = [
            extract_page(
                page,
                doc_id=doc_id,
                filename=path.name,
                source_sha256=source_sha256,
                min_text_chars=min_text_chars,
                ocr_dpi=ocr_dpi,
            )
            for page in pdf
        ]
        document = {
            "document_id": doc_id,
            "source_filename": path.name,
            "source_sha256": source_sha256,
            "file_bytes": path.stat().st_size,
            "pdf_page_count": len(pdf),
            "metadata": metadata,
        }
    return document, pages


def render_page_image(pdf_path: Path, pdf_page: int, output_path: Path, dpi: int = 144) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as pdf:
        page = pdf[pdf_page - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(output_path)

import csv
from pathlib import Path

import pymupdf as fitz

from maintenance_intelligence.hashing import sha256_file
from maintenance_intelligence.pipeline import IngestionConfig, run_ingestion


def test_pipeline_writes_provenance_artifacts(tmp_path: Path) -> None:
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    pdf_path = manuals / "sample.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "WARNING Disconnect power before maintenance.", fontsize=12)
    pdf.set_metadata({"title": "Sample manual"})
    pdf.save(pdf_path)
    pdf.close()

    checksum_path = tmp_path / "checksums.csv"
    with checksum_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "sha256", "bytes"])
        writer.writeheader()
        writer.writerow({"filename": pdf_path.name, "sha256": sha256_file(pdf_path), "bytes": pdf_path.stat().st_size})

    output = tmp_path / "processed"
    manifest = run_ingestion(
        manuals,
        checksum_path,
        output,
        config=IngestionConfig(min_text_chars=10, max_chunk_chars=300, overlap_chars=20),
    )

    assert manifest["counts"]["documents"] == 1
    assert manifest["counts"]["pages"] == 1
    assert manifest["counts"]["chunks"] >= 1
    assert manifest["counts"]["safety_chunks"] >= 1
    assert (output / "documents.jsonl").exists()
    assert (output / "pages.jsonl").exists()
    assert (output / "chunks.jsonl").exists()

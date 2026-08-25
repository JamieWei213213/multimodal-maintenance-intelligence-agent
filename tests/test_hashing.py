from pathlib import Path

import pytest

from maintenance_intelligence.hashing import sha256_file, validate_corpus


def test_validate_corpus_accepts_matching_file(tmp_path: Path) -> None:
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    pdf = manuals / "manual.pdf"
    pdf.write_bytes(b"locked manual")
    digest = sha256_file(pdf)
    checksums = tmp_path / "checksums.csv"
    checksums.write_text(f"filename,sha256,bytes\nmanual.pdf,{digest},{pdf.stat().st_size}\n", encoding="utf-8")

    result = validate_corpus(manuals, checksums)

    assert result == [{"filename": "manual.pdf", "sha256": digest, "bytes": 13}]


def test_validate_corpus_rejects_changed_file(tmp_path: Path) -> None:
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    pdf = manuals / "manual.pdf"
    pdf.write_bytes(b"changed")
    checksums = tmp_path / "checksums.csv"
    checksums.write_text("filename,sha256,bytes\nmanual.pdf,ABC,7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_corpus(manuals, checksums)


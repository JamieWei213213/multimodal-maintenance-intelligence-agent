from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpectedFile:
    filename: str
    sha256: str
    bytes: int


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def load_checksum_manifest(path: Path) -> dict[str, ExpectedFile]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "sha256", "bytes"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Checksum CSV must contain columns: {sorted(required)}")
    return {
        row["filename"]: ExpectedFile(
            filename=row["filename"],
            sha256=row["sha256"].strip().upper(),
            bytes=int(row["bytes"]),
        )
        for row in rows
    }


def validate_corpus(manual_dir: Path, checksum_csv: Path) -> list[dict[str, object]]:
    expected = load_checksum_manifest(checksum_csv)
    actual_names = {path.name for path in manual_dir.glob("*.pdf")}
    missing = sorted(set(expected) - actual_names)
    unexpected = sorted(actual_names - set(expected))
    if missing or unexpected:
        raise ValueError(f"Corpus mismatch. Missing={missing}; unexpected={unexpected}")

    validated: list[dict[str, object]] = []
    for filename, record in sorted(expected.items()):
        path = manual_dir / filename
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != record.bytes:
            raise ValueError(f"Size mismatch for {filename}: expected {record.bytes}, got {size}")
        if digest != record.sha256:
            raise ValueError(f"SHA-256 mismatch for {filename}: expected {record.sha256}, got {digest}")
        validated.append({"filename": filename, "sha256": digest, "bytes": size})
    return validated


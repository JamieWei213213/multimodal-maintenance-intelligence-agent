from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from maintenance_intelligence.visuals import VisualAssetRecord


class VisualEvidenceIndex:
    """Read-only lookup for visuals that share a page with validated text citations."""

    def __init__(self, records: Iterable[VisualAssetRecord]) -> None:
        self.by_id: dict[str, VisualAssetRecord] = {}
        self.by_page: dict[tuple[str, int], list[VisualAssetRecord]] = {}
        for record in records:
            if record.asset_id in self.by_id:
                raise ValueError(f"Duplicate visual asset ID: {record.asset_id}")
            self.by_id[record.asset_id] = record
            self.by_page.setdefault((record.source_filename, record.pdf_page), []).append(record)

    @classmethod
    def load(cls, path: Path) -> "VisualEvidenceIndex":
        if not path.exists():
            return cls([])
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(VisualAssetRecord(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid visual record at line {line_number}") from exc
        return cls(records)

    def for_citations(self, citations: Iterable[dict[str, object]], limit: int = 4) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        selected: list[VisualAssetRecord] = []
        seen: set[str] = set()
        for citation in citations:
            key = (str(citation["source_filename"]), int(citation["pdf_page"]))
            for record in self.by_page.get(key, []):
                if record.asset_id in seen:
                    continue
                seen.add(record.asset_id)
                selected.append(record)
                if len(selected) == limit:
                    return [self._public_record(item) for item in selected]
        return [self._public_record(item) for item in selected]

    @staticmethod
    def _public_record(record: VisualAssetRecord) -> dict[str, object]:
        return {
            "asset_id": record.asset_id,
            "source_filename": record.source_filename,
            "pdf_page": record.pdf_page,
            "asset_type": record.asset_type,
            "bbox": record.bbox,
            "width": record.width,
            "height": record.height,
            "nearby_text": record.nearby_text,
        }

    def get(self, asset_id: str) -> VisualAssetRecord | None:
        return self.by_id.get(asset_id)

    def __len__(self) -> int:
        return len(self.by_id)

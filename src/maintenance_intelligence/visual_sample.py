from __future__ import annotations

import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path

import pymupdf


ANNOTATION_FIELDS = [
    "asset_id",
    "source_filename",
    "pdf_page",
    "asset_type",
    "bbox",
    "nearby_text",
    "crop_path",
    "is_meaningful_visual",
    "visual_category",
    "label_notes",
]


def _load_inventory(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratified_sample(
    records: list[dict[str, object]], sample_size: int, *, seed: int = 580
) -> list[dict[str, object]]:
    """Round-robin across document/type groups with deterministic within-group order."""
    if sample_size <= 0:
        return []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["source_filename"]), str(record["asset_type"]))].append(record)
    randomizer = random.Random(seed)
    queues: list[deque[dict[str, object]]] = []
    for key in sorted(grouped):
        group = grouped[key]
        randomizer.shuffle(group)
        queues.append(deque(group))
    selected: list[dict[str, object]] = []
    while queues and len(selected) < min(sample_size, len(records)):
        remaining: list[deque[dict[str, object]]] = []
        for queue in queues:
            if queue and len(selected) < sample_size:
                selected.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def _context_rect(page: pymupdf.Page, bbox: list[float], padding: float = 36.0) -> pymupdf.Rect:
    rect = pymupdf.Rect(bbox)
    expanded = pymupdf.Rect(
        rect.x0 - padding,
        rect.y0 - padding,
        rect.x1 + padding,
        rect.y1 + padding,
    )
    return expanded & page.rect


def create_annotation_sample(
    inventory_path: Path,
    manual_dir: Path,
    output_csv: Path,
    render_dir: Path,
    *,
    sample_size: int = 24,
    seed: int = 580,
) -> dict[str, int]:
    records = stratified_sample(_load_inventory(inventory_path), sample_size, seed=seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    documents: dict[str, pymupdf.Document] = {}
    rows: list[dict[str, object]] = []
    try:
        for index, record in enumerate(records, start=1):
            filename = str(record["source_filename"])
            document = documents.get(filename)
            if document is None:
                document = pymupdf.open(manual_dir / filename)
                documents[filename] = document
            page = document[int(record["pdf_page"]) - 1]
            destination = render_dir / f"{index:03d}_{record['asset_id']}.png"
            clip = _context_rect(page, list(record["bbox"]))
            page.get_pixmap(clip=clip, dpi=144, alpha=False).save(destination)
            rows.append(
                {
                    **{field: record.get(field, "") for field in ANNOTATION_FIELDS[:6]},
                    "bbox": json.dumps(record["bbox"]),
                    "crop_path": destination.as_posix(),
                    "is_meaningful_visual": "",
                    "visual_category": "",
                    "label_notes": "",
                }
            )
    finally:
        for document in documents.values():
            document.close()
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"sampled_candidates": len(rows), "seed": seed}

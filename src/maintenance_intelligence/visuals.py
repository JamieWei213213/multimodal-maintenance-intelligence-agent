from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class VisualAssetRecord:
    asset_id: str
    source_filename: str
    pdf_page: int
    asset_type: str
    bbox: list[float]
    width: int | None
    height: int | None
    nearby_text: str
    rendered_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _asset_id(filename: str, page: int, asset_type: str, bbox: list[float], ordinal: int) -> str:
    identity = f"{filename}|{page}|{asset_type}|{bbox}|{ordinal}"
    return "vis_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _nearby_text(page: pymupdf.Page, bbox: pymupdf.Rect, margin: float = 90.0) -> str:
    candidates: list[tuple[float, str]] = []
    for block in page.get_text("blocks"):
        block_rect = pymupdf.Rect(block[:4])
        text = " ".join(str(block[4]).split())
        if not text:
            continue
        vertical_distance = min(abs(block_rect.y0 - bbox.y1), abs(bbox.y0 - block_rect.y1))
        horizontal_overlap = max(0.0, min(block_rect.x1, bbox.x1) - max(block_rect.x0, bbox.x0))
        if vertical_distance <= margin and horizontal_overlap > 0:
            candidates.append((vertical_distance, text))
    candidates.sort(key=lambda item: item[0])
    return " ".join(text for _, text in candidates[:3])[:1000]


def inventory_pdf(
    path: Path,
    *,
    render_dir: Path | None = None,
    min_vector_drawings: int = 20,
) -> list[VisualAssetRecord]:
    records: list[VisualAssetRecord] = []
    with pymupdf.open(path) as document:
        for pdf_page, page in enumerate(document, start=1):
            ordinal = 0
            seen_rects: set[tuple[float, float, float, float]] = set()
            for image in page.get_images(full=True):
                xref = int(image[0])
                for rect in page.get_image_rects(xref):
                    rounded = tuple(round(value, 2) for value in (rect.x0, rect.y0, rect.x1, rect.y1))
                    if rounded in seen_rects or rect.width < 20 or rect.height < 20:
                        continue
                    seen_rects.add(rounded)
                    bbox = list(rounded)
                    rendered_path = None
                    asset_id = _asset_id(path.name, pdf_page, "raster_image", bbox, ordinal)
                    if render_dir is not None:
                        destination = render_dir / path.stem / f"{asset_id}.png"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        pixmap = page.get_pixmap(clip=rect, dpi=144, alpha=False)
                        pixmap.save(destination)
                        rendered_path = destination.as_posix()
                    records.append(
                        VisualAssetRecord(
                            asset_id=asset_id,
                            source_filename=path.name,
                            pdf_page=pdf_page,
                            asset_type="raster_image",
                            bbox=bbox,
                            width=int(image[2]),
                            height=int(image[3]),
                            nearby_text=_nearby_text(page, rect),
                            rendered_path=rendered_path,
                        )
                    )
                    ordinal += 1

            drawings = page.get_drawings()
            substantial = [drawing["rect"] for drawing in drawings if drawing["rect"].get_area() >= 25]
            if len(substantial) >= min_vector_drawings:
                union = substantial[0]
                for rect in substantial[1:]:
                    union |= rect
                bbox = [round(value, 2) for value in (union.x0, union.y0, union.x1, union.y1)]
                asset_id = _asset_id(path.name, pdf_page, "vector_drawing_region", bbox, ordinal)
                rendered_path = None
                if render_dir is not None:
                    destination = render_dir / path.stem / f"{asset_id}.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    page.get_pixmap(clip=union, dpi=144, alpha=False).save(destination)
                    rendered_path = destination.as_posix()
                records.append(
                    VisualAssetRecord(
                        asset_id=asset_id,
                        source_filename=path.name,
                        pdf_page=pdf_page,
                        asset_type="vector_drawing_region",
                        bbox=bbox,
                        width=None,
                        height=None,
                        nearby_text=_nearby_text(page, union),
                        rendered_path=rendered_path,
                    )
                )
    return records


def build_visual_inventory(
    manual_dir: Path,
    output_path: Path,
    *,
    render_dir: Path | None = None,
) -> dict[str, int]:
    records: list[VisualAssetRecord] = []
    for path in sorted(manual_dir.glob("*.pdf")):
        records.extend(inventory_pdf(path, render_dir=render_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "visual_assets": len(records),
        "raster_images": sum(record.asset_type == "raster_image" for record in records),
        "vector_regions": sum(record.asset_type == "vector_drawing_region" for record in records),
    }

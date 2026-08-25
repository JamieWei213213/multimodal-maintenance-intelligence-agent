from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pymupdf

from maintenance_intelligence.visual_retrieval import VisualSearchResult
from maintenance_intelligence.visuals import VisualAssetRecord


CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
CLIP_IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision"


def load_labeled_asset_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row["asset_id"] for row in csv.DictReader(handle)]


def render_asset_crops(
    assets: list[VisualAssetRecord],
    manual_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    padding: float = 24.0,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents: dict[str, pymupdf.Document] = {}
    paths: dict[str, Path] = {}
    try:
        for asset in assets:
            document = documents.get(asset.source_filename)
            if document is None:
                document = pymupdf.open(manual_dir / asset.source_filename)
                documents[asset.source_filename] = document
            page = document[asset.pdf_page - 1]
            rect = pymupdf.Rect(asset.bbox)
            clip = pymupdf.Rect(
                rect.x0 - padding,
                rect.y0 - padding,
                rect.x1 + padding,
                rect.y1 + padding,
            ) & page.rect
            destination = output_dir / f"{asset.asset_id}.png"
            page.get_pixmap(clip=clip, dpi=dpi, alpha=False).save(destination)
            paths[asset.asset_id] = destination
    finally:
        for document in documents.values():
            document.close()
    return paths


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(values)
    return values / norm if norm else values


class VisualVectorRetriever:
    def __init__(
        self,
        assets: list[VisualAssetRecord],
        image_vectors: Iterable[np.ndarray],
        query_embedder: Callable[[str], np.ndarray],
    ) -> None:
        self.assets = assets
        vectors = [_normalized(vector) for vector in image_vectors]
        if len(self.assets) != len(vectors):
            raise ValueError("Asset and image vector counts do not match.")
        self.image_vectors = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
        self.query_embedder = query_embedder

    def search(self, query: str, limit: int = 10) -> list[VisualSearchResult]:
        if not query.strip() or limit <= 0 or not self.assets:
            return []
        query_vector = _normalized(self.query_embedder(query))
        scores = self.image_vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            VisualSearchResult(
                asset=self.assets[int(index)],
                score=float(scores[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        ]


def build_clip_retriever(
    assets: list[VisualAssetRecord],
    crop_paths: dict[str, Path],
    cache_dir: Path,
) -> VisualVectorRetriever:
    from fastembed import ImageEmbedding, TextEmbedding

    image_model = ImageEmbedding(CLIP_IMAGE_MODEL, cache_dir=str(cache_dir))
    text_model = TextEmbedding(CLIP_TEXT_MODEL, cache_dir=str(cache_dir))
    image_vectors = list(image_model.embed([crop_paths[asset.asset_id] for asset in assets]))

    def embed_query(query: str) -> np.ndarray:
        return next(iter(text_model.query_embed(query)))

    return VisualVectorRetriever(assets, image_vectors, embed_query)

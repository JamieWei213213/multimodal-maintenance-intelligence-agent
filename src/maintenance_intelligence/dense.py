from __future__ import annotations

import uuid
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from maintenance_intelligence.models import ChunkRecord
from maintenance_intelligence.retrieval import SearchResult, _deduplicate_pages


VECTOR_NAME = "dense"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"maintenance-intelligence:{chunk_id}"))


class DenseRetriever:
    def __init__(
        self,
        *,
        collection_name: str,
        model_name: str,
        cache_dir: Path,
        url: str | None = None,
        path: Path | str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.model_name = model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        if url:
            self.client = QdrantClient(url=url)
        else:
            location = str(path or ":memory:")
            if location != ":memory:":
                Path(location).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=location) if location != ":memory:" else QdrantClient(":memory:")

    def _vector_size(self) -> int:
        return len(next(iter(self.model.embed(["dimension probe"]))))

    def ensure_collection(self, *, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if recreate and exists:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    VECTOR_NAME: models.VectorParams(
                        size=self._vector_size(),
                        distance=models.Distance.COSINE,
                    )
                },
            )

    def index(self, chunks: list[ChunkRecord], *, recreate: bool = False, batch_size: int = 128) -> int:
        self.ensure_collection(recreate=recreate)
        indexed = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = list(self.model.embed([chunk.text for chunk in batch], batch_size=batch_size))
            points = [
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector={VECTOR_NAME: vector.tolist()},
                    payload=chunk.to_dict(),
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            indexed += len(points)
        return indexed

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if limit <= 0 or not query.strip() or not self.client.collection_exists(self.collection_name):
            return []
        vector = next(iter(self.model.query_embed(query))).tolist()
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            using=VECTOR_NAME,
            limit=max(limit * 4, limit),
            with_payload=True,
        ).points
        results = [
            SearchResult(chunk=ChunkRecord(**dict(point.payload or {})), score=float(point.score))
            for point in points
            if point.payload
        ]
        return _deduplicate_pages(results, limit)


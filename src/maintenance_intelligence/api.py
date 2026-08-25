from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pymupdf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from maintenance_intelligence.answering import GroundedAnswerer
from maintenance_intelligence.config import Settings
from maintenance_intelligence.database import QueryRepository
from maintenance_intelligence.dense import DenseRetriever
from maintenance_intelligence.hybrid import HybridRetriever
from maintenance_intelligence.observability import (
    ApiMetrics,
    monotonic_time,
    new_request_id,
    request_log,
)
from maintenance_intelligence.retrieval import BM25Retriever, load_chunks
from maintenance_intelligence.visual_evidence import VisualEvidenceIndex


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


@dataclass
class AppRuntime:
    answerer: Any
    chunks_by_id: dict[str, Any]
    dense: Any
    repository: Any
    visual_index: Any | None = None

    def close(self) -> None:
        self.dense.client.close()


def build_runtime(settings: Settings) -> AppRuntime:
    chunks = load_chunks(settings.chunks_path)
    dense = DenseRetriever(
        collection_name=settings.qdrant_collection,
        model_name=settings.embedding_model,
        cache_dir=settings.model_cache,
        url=settings.qdrant_url,
        path=settings.qdrant_path,
    )
    lexical = BM25Retriever(chunks)
    hybrid = HybridRetriever(
        lexical,
        dense,
        lexical_weight=settings.lexical_rrf_weight,
        dense_weight=settings.dense_rrf_weight,
    )
    repository = QueryRepository(settings.database_url)
    repository.initialize()
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    visual_index = VisualEvidenceIndex.load(settings.visual_inventory_path)
    return AppRuntime(
        answerer=GroundedAnswerer(
            hybrid,
            chunks_by_id,
            min_rrf=settings.abstention_min_rrf,
            evidence_limit=settings.answer_evidence_limit,
            adjacent_chunk_radius=settings.adjacent_chunk_radius,
        ),
        chunks_by_id=chunks_by_id,
        dense=dense,
        repository=repository,
        visual_index=visual_index,
    )


def create_app(
    settings: Settings | None = None,
    runtime_factory: Callable[[Settings], AppRuntime] = build_runtime,
) -> FastAPI:
    settings = settings or Settings()
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory(settings)
        app.state.answerer = runtime.answerer
        app.state.chunks_by_id = runtime.chunks_by_id
        app.state.dense = runtime.dense
        app.state.repository = runtime.repository
        app.state.visual_index = runtime.visual_index or VisualEvidenceIndex([])
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="Maintenance Intelligence API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )

    @app.middleware("http")
    async def observe_request(request, call_next):
        request_id = new_request_id(request.headers.get("x-request-id"))
        started = monotonic_time()
        status = 500
        try:
            protected = (
                request.url.path == "/ask"
                or request.url.path.startswith("/documents/")
                or request.url.path.startswith("/evidence/")
                or request.url.path.startswith("/visuals/")
            )
            supplied_key = request.headers.get("x-api-key", "")
            if settings.api_key and protected and not secrets.compare_digest(
                supplied_key,
                settings.api_key,
            ):
                response = JSONResponse(status_code=401, content={"detail": "Invalid API key"})
            else:
                response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            response.headers["x-content-type-options"] = "nosniff"
            response.headers["x-frame-options"] = "DENY"
            response.headers["referrer-policy"] = "no-referrer"
            if request.url.path == "/ask":
                response.headers["cache-control"] = "no-store"
            return response
        finally:
            duration = monotonic_time() - started
            route_object = request.scope.get("route")
            route = getattr(route_object, "path", request.url.path)
            metrics.requests.labels(request.method, route, str(status)).inc()
            metrics.duration.labels(request.method, route).observe(duration)
            request_log(
                request_id=request_id,
                method=request.method,
                route=route,
                status=status,
                duration_seconds=duration,
            )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, object]:
        dense = app.state.dense
        if not dense.client.collection_exists(settings.qdrant_collection):
            raise HTTPException(status_code=503, detail="Qdrant collection is unavailable")
        point_count = dense.client.count(settings.qdrant_collection, exact=True).count
        if point_count <= 0 or not app.state.chunks_by_id:
            raise HTTPException(status_code=503, detail="Retrieval index is empty")
        return {
            "status": "ready",
            "chunks": len(app.state.chunks_by_id),
            "vectors": point_count,
            "collection": settings.qdrant_collection,
            "visual_candidates": len(app.state.visual_index),
        }

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/evaluation")
    def evaluation_summary() -> dict[str, object]:
        try:
            data = json.loads(settings.evaluation_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=503, detail="Evaluation summary is unavailable") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=503, detail="Evaluation summary is invalid")
        return data

    @app.post("/ask")
    def ask(request: QuestionRequest) -> dict[str, object]:
        response = app.state.answerer.answer(request.question)
        response["visual_evidence"] = (
            app.state.visual_index.for_citations(response.get("citations", []))
            if not response["abstained"]
            else []
        )
        metrics.answers.labels(str(response["abstained"]).lower()).inc()
        app.state.repository.record(request.question, response)
        return response

    @app.get("/documents/{filename}/pages/{pdf_page}/image")
    def page_image(filename: str, pdf_page: int) -> Response:
        safe_name = Path(filename).name
        path = settings.manuals_path / safe_name
        if safe_name != filename or not path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        with pymupdf.open(path) as document:
            if pdf_page < 1 or pdf_page > len(document):
                raise HTTPException(status_code=404, detail="Page not found")
            pixmap = document[pdf_page - 1].get_pixmap(dpi=144, alpha=False)
            return Response(content=pixmap.tobytes("png"), media_type="image/png")

    @app.get("/evidence/{chunk_id}/image")
    def evidence_image(chunk_id: str) -> Response:
        chunk = app.state.chunks_by_id.get(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Evidence chunk not found")
        path = settings.manuals_path / Path(chunk.source_filename).name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        with pymupdf.open(path) as document:
            page = document[chunk.pdf_page - 1]
            for bbox in chunk.bboxes:
                page.draw_rect(
                    pymupdf.Rect(bbox),
                    color=(0.1, 0.55, 0.35),
                    fill=(0.75, 1.0, 0.82),
                    fill_opacity=0.22,
                    width=1.5,
                    overlay=True,
                )
            pixmap = page.get_pixmap(dpi=144, alpha=False)
            return Response(content=pixmap.tobytes("png"), media_type="image/png")

    @app.get("/visuals/{asset_id}/image")
    def visual_image(asset_id: str) -> Response:
        asset = app.state.visual_index.get(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Visual candidate not found")
        path = settings.manuals_path / Path(asset.source_filename).name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        with pymupdf.open(path) as document:
            if asset.pdf_page < 1 or asset.pdf_page > len(document):
                raise HTTPException(status_code=404, detail="Page not found")
            page = document[asset.pdf_page - 1]
            bbox = pymupdf.Rect(asset.bbox)
            clip = pymupdf.Rect(bbox.x0 - 24, bbox.y0 - 24, bbox.x1 + 24, bbox.y1 + 24) & page.rect
            if clip.is_empty or clip.is_infinite:
                raise HTTPException(status_code=422, detail="Visual candidate has an invalid bounding box")
            page.draw_rect(bbox, color=(0.12, 0.68, 0.78), width=2, overlay=True)
            pixmap = page.get_pixmap(clip=clip, dpi=144, alpha=False)
            return Response(content=pixmap.tobytes("png"), media_type="image/png")

    return app


app = create_app()

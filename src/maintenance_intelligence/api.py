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

from maintenance_intelligence.answering import INSUFFICIENT, GroundedAnswerer
from maintenance_intelligence.agent import MaintenanceAgent
from maintenance_intelligence.config import Settings
from maintenance_intelligence.database import QueryRepository
from maintenance_intelligence.dense import DenseRetriever
from maintenance_intelligence.guardrails import apply_output_guard, check_input
from maintenance_intelligence.hybrid import HybridRetriever
from maintenance_intelligence.llm import UnavailableLLM, build_llm_client
from maintenance_intelligence.observability import (
    ApiMetrics,
    monotonic_time,
    new_request_id,
    request_log,
)
from maintenance_intelligence.prompts import list_prompts
from maintenance_intelligence.retrieval import BM25Retriever, load_chunks
from maintenance_intelligence.schemas import AnswerResponse, GuardrailRejection, QuestionRequest
from maintenance_intelligence.tracing import configure_tracing, instrument_fastapi, span
from maintenance_intelligence.visual_evidence import VisualEvidenceIndex


@dataclass
class AppRuntime:
    answerer: Any
    chunks_by_id: dict[str, Any]
    dense: Any
    repository: Any
    visual_index: Any | None = None
    agent: Any | None = None
    llm: Any | None = None

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
    answerer = GroundedAnswerer(
        hybrid,
        chunks_by_id,
        min_rrf=settings.abstention_min_rrf,
        evidence_limit=settings.answer_evidence_limit,
        adjacent_chunk_radius=settings.adjacent_chunk_radius,
    )
    return AppRuntime(
        answerer=answerer,
        chunks_by_id=chunks_by_id,
        dense=dense,
        repository=repository,
        visual_index=visual_index,
        agent=MaintenanceAgent(answerer, chunks_by_id, visual_index),
        llm=build_llm_client(settings),
    )


def create_app(
    settings: Settings | None = None,
    runtime_factory: Callable[[Settings], AppRuntime] = build_runtime,
) -> FastAPI:
    settings = settings or Settings()
    metrics = ApiMetrics()
    tracing_backend = configure_tracing(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory(settings)
        app.state.answerer = runtime.answerer
        app.state.chunks_by_id = runtime.chunks_by_id
        app.state.dense = runtime.dense
        app.state.repository = runtime.repository
        app.state.visual_index = runtime.visual_index or VisualEvidenceIndex([])
        app.state.agent = runtime.agent or MaintenanceAgent(
            runtime.answerer, runtime.chunks_by_id, app.state.visual_index
        )
        app.state.llm = runtime.llm or UnavailableLLM()
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="Maintenance Intelligence API", version=settings.version, lifespan=lifespan)
    instrument_fastapi(app)
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
                request.url.path in {"/ask", "/agent/query", "/config"}
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
            if request.url.path in {"/ask", "/agent/query"}:
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
        return {"status": "ok", "version": settings.version}

    @app.get("/config")
    def runtime_config() -> dict[str, object]:
        """Non-secret runtime configuration, so a reviewer can see what produced an answer."""
        llm = getattr(app.state, "llm", None)
        return {
            "version": settings.version,
            "llm_provider": getattr(llm, "provider", "none"),
            "llm_model": getattr(llm, "default_model", "none"),
            "tracing_exporter": tracing_backend,
            "embedding_model": settings.embedding_model,
            "lexical_rrf_weight": settings.lexical_rrf_weight,
            "dense_rrf_weight": settings.dense_rrf_weight,
            "abstention_min_rrf": settings.abstention_min_rrf,
            "prompt_versions": list_prompts(),
        }

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

    def _guarded_answer(question: str, mode: str, produce) -> AnswerResponse:
        with span("guardrails.input", question_length=len(question)) as guard_span:
            decision = check_input(question)
            guard_span.set_attribute("guardrails.flags", ",".join(decision.flags) or "none")
            if not decision.allowed:
                metrics.guardrail_rejections.labels(decision.flags[0] if decision.flags else "unknown").inc()
                raise HTTPException(
                    status_code=422,
                    detail=GuardrailRejection(detail=decision.reason or "Rejected", flags=decision.flags).model_dump(),
                )
        with span(f"answer.{mode}") as answer_span:
            response = produce(question)
            answer_span.set_attribute("answer.abstained", bool(response.get("abstained")))
            answer_span.set_attribute("answer.citations", len(response.get("citations", [])))
        with span("guardrails.output") as output_span:
            response = apply_output_guard(
                response,
                app.state.chunks_by_id,
                input_decision=decision,
                insufficient_text=INSUFFICIENT,
            )
            output_span.set_attribute(
                "guardrails.grounded_sentence_ratio",
                float(response["guardrails"]["grounded_sentence_ratio"]),
            )
        metrics.answers.labels(str(response["abstained"]).lower()).inc()
        app.state.repository.record(question, response)
        return AnswerResponse.from_payload(response, mode=mode)

    @app.post("/ask", response_model=AnswerResponse, responses={422: {"model": GuardrailRejection}})
    def ask(request: QuestionRequest) -> AnswerResponse:
        def produce(question: str) -> dict[str, object]:
            response = app.state.answerer.answer(question)
            response["visual_evidence"] = (
                app.state.visual_index.for_citations(response.get("citations", []))
                if not response["abstained"]
                else []
            )
            return response

        return _guarded_answer(request.question, "extractive", produce)

    @app.post("/agent/query", response_model=AnswerResponse, responses={422: {"model": GuardrailRejection}})
    def agent_query(request: QuestionRequest) -> AnswerResponse:
        return _guarded_answer(request.question, "agent", app.state.agent.run)

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

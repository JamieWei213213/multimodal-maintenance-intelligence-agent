from types import SimpleNamespace
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from maintenance_intelligence.api import AppRuntime, create_app
from maintenance_intelligence.config import Settings
from maintenance_intelligence.visual_evidence import VisualEvidenceIndex
from maintenance_intelligence.visuals import VisualAssetRecord


class FakeAnswerer:
    def answer(self, question: str):
        return {
            "answer": f"Supported: {question}",
            "abstained": False,
            "confidence": 0.9,
            "citations": [],
        }


class FakeClient:
    def collection_exists(self, collection: str) -> bool:
        return collection == "test_chunks"

    def count(self, collection: str, exact: bool = True):
        return SimpleNamespace(count=2)

    def close(self) -> None:
        pass


class FakeRepository:
    def __init__(self) -> None:
        self.events = []

    def record(self, question, response) -> None:
        self.events.append((question, response))


class FakeVisualIndex:
    def __len__(self):
        return 1

    def for_citations(self, citations, limit=4):
        return [{"asset_id": "vis_test", "source_filename": "manual.pdf", "pdf_page": 1}]

    def get(self, asset_id):
        return None


def make_client(
    *,
    api_key: str | None = None,
    manuals_path: Path = Path("data/manuals"),
    visual_index=None,
):
    repository = FakeRepository()
    runtime = AppRuntime(
        answerer=FakeAnswerer(),
        chunks_by_id={"one": object(), "two": object()},
        dense=SimpleNamespace(client=FakeClient()),
        repository=repository,
        visual_index=visual_index or FakeVisualIndex(),
    )
    settings = Settings(
        qdrant_collection="test_chunks",
        api_key=api_key,
        manuals_path=manuals_path,
    )
    return TestClient(create_app(settings, runtime_factory=lambda _: runtime)), repository


def test_health_readiness_request_id_security_headers_and_metrics():
    client, repository = make_client()
    with client:
        health = client.get("/health", headers={"x-request-id": "integration-123"})
        assert health.status_code == 200
        assert health.headers["x-request-id"] == "integration-123"
        assert health.headers["x-content-type-options"] == "nosniff"
        assert health.headers["x-frame-options"] == "DENY"

        ready = client.get("/ready")
        assert ready.json() == {
            "status": "ready",
            "chunks": 2,
            "vectors": 2,
            "collection": "test_chunks",
            "visual_candidates": 1,
        }

        evaluation = client.get("/evaluation")
        assert evaluation.status_code == 200
        assert evaluation.json()["corpus"]["chunks"] == 11888

        answer = client.post("/ask", json={"question": "What is A2B1?"})
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "no-store"
        assert answer.json()["visual_evidence"][0]["asset_id"] == "vis_test"
        assert repository.events[0][0] == "What is A2B1?"

        invalid = client.post("/ask", json={"question": "x"})
        assert invalid.status_code == 422

        metrics = client.get("/metrics")
        assert "maintenance_api_requests_total" in metrics.text
        assert "maintenance_answers_total" in metrics.text


def test_configured_api_key_protects_maintenance_routes():
    client, _ = make_client(api_key="secret-value")
    with client:
        denied = client.post("/ask", json={"question": "What is A2B1?"})
        assert denied.status_code == 401
        allowed = client.post(
            "/ask",
            json={"question": "What is A2B1?"},
            headers={"x-api-key": "secret-value"},
        )
        assert allowed.status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/visuals/unknown/image").status_code == 401


def test_visual_candidate_endpoint_renders_provenance_crop(tmp_path):
    manual = tmp_path / "manual.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(40, 40, 160, 160))
    document.save(manual)
    document.close()
    index = VisualEvidenceIndex(
        [
            VisualAssetRecord(
                asset_id="vis_crop",
                source_filename="manual.pdf",
                pdf_page=1,
                asset_type="vector_drawing_region",
                bbox=[40, 40, 160, 160],
                width=None,
                height=None,
                nearby_text="Figure 1",
            )
        ]
    )
    client, _ = make_client(manuals_path=tmp_path, visual_index=index)
    with client:
        response = client.get("/visuals/vis_crop/image")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")

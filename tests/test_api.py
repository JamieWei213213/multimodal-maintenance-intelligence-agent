from types import SimpleNamespace

from fastapi.testclient import TestClient

from maintenance_intelligence.api import AppRuntime, create_app
from maintenance_intelligence.config import Settings


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


def make_client(*, api_key: str | None = None):
    repository = FakeRepository()
    runtime = AppRuntime(
        answerer=FakeAnswerer(),
        chunks_by_id={"one": object(), "two": object()},
        dense=SimpleNamespace(client=FakeClient()),
        repository=repository,
    )
    settings = Settings(qdrant_collection="test_chunks", api_key=api_key)
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
        }

        evaluation = client.get("/evaluation")
        assert evaluation.status_code == 200
        assert evaluation.json()["corpus"]["chunks"] == 11888

        answer = client.post("/ask", json={"question": "What is A2B1?"})
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "no-store"
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

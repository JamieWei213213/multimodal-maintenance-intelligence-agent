from __future__ import annotations

import json
import logging
import time
import uuid

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


logger = logging.getLogger("maintenance_intelligence.api")


class ApiMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "maintenance_api_requests_total",
            "HTTP requests handled by the maintenance API.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.duration = Histogram(
            "maintenance_api_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("method", "route"),
            registry=self.registry,
        )
        self.answers = Counter(
            "maintenance_answers_total",
            "Maintenance answers grouped by abstention outcome.",
            ("abstained",),
            registry=self.registry,
        )
        self.guardrail_rejections = Counter(
            "maintenance_guardrail_rejections_total",
            "Questions rejected by the input guardrail, by first flag.",
            ("flag",),
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "maintenance_llm_tokens_total",
            "LLM tokens consumed, by model and direction.",
            ("model", "direction"),
            registry=self.registry,
        )
        self.llm_cost = Counter(
            "maintenance_llm_cost_usd_total",
            "Estimated LLM spend in USD, by model.",
            ("model",),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)


def request_log(
    *,
    request_id: str,
    method: str,
    route: str,
    status: int,
    duration_seconds: float,
) -> None:
    logger.info(
        json.dumps(
            {
                "event": "http_request",
                "request_id": request_id,
                "method": method,
                "route": route,
                "status": status,
                "duration_ms": round(duration_seconds * 1000, 2),
            },
            separators=(",", ":"),
        )
    )


def new_request_id(incoming: str | None) -> str:
    return incoming.strip()[:128] if incoming and incoming.strip() else str(uuid.uuid4())


def monotonic_time() -> float:
    return time.perf_counter()

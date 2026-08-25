from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg


SCHEMA = """
CREATE TABLE IF NOT EXISTS query_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    question TEXT NOT NULL,
    abstained BOOLEAN NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    citations JSONB NOT NULL
);
"""


class QueryRepository:
    def __init__(self, database_url: str | None) -> None:
        self.database_url = database_url

    @contextmanager
    def connect(self):
        if not self.database_url:
            yield None
            return
        with psycopg.connect(self.database_url) as connection:
            yield connection

    def initialize(self) -> None:
        with self.connect() as connection:
            if connection is not None:
                connection.execute(SCHEMA)
                connection.commit()

    def record(self, question: str, response: dict[str, object]) -> None:
        with self.connect() as connection:
            if connection is None:
                return
            connection.execute(
                "INSERT INTO query_events (created_at, question, abstained, confidence, citations) VALUES (%s, %s, %s, %s, %s)",
                (
                    datetime.now(timezone.utc),
                    question,
                    bool(response["abstained"]),
                    float(response["confidence"]),
                    json.dumps(response["citations"]),
                ),
            )
            connection.commit()


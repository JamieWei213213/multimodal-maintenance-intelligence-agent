from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    chunks_path: Path = Path(os.getenv("CHUNKS_PATH", "data/processed/chunks.jsonl"))
    manuals_path: Path = Path(os.getenv("MANUALS_PATH", "data/manuals"))
    visual_inventory_path: Path = Path(
        os.getenv("VISUAL_INVENTORY_PATH", "data/processed/visual_assets.jsonl")
    )
    evaluation_summary_path: Path = Path(
        os.getenv("EVALUATION_SUMMARY_PATH", "evaluation/baseline_metrics.json")
    )
    qdrant_url: str | None = os.getenv("QDRANT_URL")
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", "data/qdrant"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "maintenance_chunks")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    model_cache: Path = Path(os.getenv("MODEL_CACHE", "data/models"))
    database_url: str | None = os.getenv("DATABASE_URL")
    api_key: str | None = os.getenv("API_KEY")
    dense_candidate_limit: int = int(os.getenv("DENSE_CANDIDATE_LIMIT", "20"))
    lexical_candidate_limit: int = int(os.getenv("LEXICAL_CANDIDATE_LIMIT", "20"))
    lexical_rrf_weight: float = _float("LEXICAL_RRF_WEIGHT", 16.0)
    dense_rrf_weight: float = _float("DENSE_RRF_WEIGHT", 1.0)
    answer_evidence_limit: int = int(os.getenv("ANSWER_EVIDENCE_LIMIT", "10"))
    adjacent_chunk_radius: int = int(os.getenv("ADJACENT_CHUNK_RADIUS", "1"))
    abstention_min_rrf: float = _float("ABSTENTION_MIN_RRF", 0.016)
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    )

from __future__ import annotations

import argparse
from dataclasses import replace

from maintenance_intelligence.config import Settings
from maintenance_intelligence.dense import DenseRetriever
from maintenance_intelligence.retrieval import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Index maintenance chunks into Qdrant.")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    chunks = load_chunks(settings.chunks_path)
    dense = DenseRetriever(
        collection_name=settings.qdrant_collection,
        model_name=settings.embedding_model,
        cache_dir=settings.model_cache,
        url=settings.qdrant_url,
        path=settings.qdrant_path,
    )
    count = dense.index(chunks, recreate=args.recreate)
    print(f"Indexed {count} chunks into {settings.qdrant_collection}.")


from __future__ import annotations

import argparse
import json
from pathlib import Path

from maintenance_intelligence.answering import GroundedAnswerer
from maintenance_intelligence.config import Settings
from maintenance_intelligence.dense import DenseRetriever
from maintenance_intelligence.experiments import ExperimentTracker
from maintenance_intelligence.evaluation import (
    evaluate_retriever,
    evaluate_answerer,
    load_golden_questions,
    save_evaluation_report,
)
from maintenance_intelligence.hybrid import HybridRetriever
from maintenance_intelligence.retrieval import BM25Retriever, SearchResult, load_chunks


class _HybridEvaluationAdapter:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(chunk=item.result.chunk, score=item.rrf_score, rank=rank)
            for rank, item in enumerate(self.retriever.search(query, limit=limit), start=1)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense and hybrid retrieval.")
    parser.add_argument("--questions", type=Path, default=Path("evaluation/golden_questions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--run-name", default="evaluate-full", help="MLflow run name (logged when MLFLOW_TRACKING_URI is set)")
    args = parser.parse_args()

    settings = Settings()
    chunks = load_chunks(settings.chunks_path)
    questions = load_golden_questions(args.questions)
    dense = DenseRetriever(
        collection_name=settings.qdrant_collection,
        model_name=settings.embedding_model,
        cache_dir=settings.model_cache,
        url=settings.qdrant_url,
        path=settings.qdrant_path,
    )
    try:
        hybrid = _HybridEvaluationAdapter(HybridRetriever(
            BM25Retriever(chunks),
            dense,
            lexical_weight=settings.lexical_rrf_weight,
            dense_weight=settings.dense_rrf_weight,
        ))
        reports = {
            "dense": evaluate_retriever(dense, questions, limit=args.limit),
            "hybrid": evaluate_retriever(hybrid, questions, limit=args.limit),
        }
        for name, report in reports.items():
            save_evaluation_report(report, args.output_dir / f"{name}.json")
        answerer = GroundedAnswerer(
            hybrid.retriever,
            {chunk.chunk_id: chunk for chunk in chunks},
            min_rrf=settings.abstention_min_rrf,
            evidence_limit=settings.answer_evidence_limit,
            adjacent_chunk_radius=settings.adjacent_chunk_radius,
        )
        answering_report = evaluate_answerer(answerer, questions)
        save_evaluation_report(answering_report, args.output_dir / "answering.json")
        summary = {
            **{name: report["metrics"] for name, report in reports.items()},
            "answering": answering_report["metrics"],
        }
        run = ExperimentTracker(settings.mlflow_tracking_uri).log_run(
            args.run_name,
            params={
                "mode": "extractive",
                "questions": str(args.questions),
                "limit": args.limit,
                "embedding_model": settings.embedding_model,
                "lexical_rrf_weight": settings.lexical_rrf_weight,
                "dense_rrf_weight": settings.dense_rrf_weight,
                "abstention_min_rrf": settings.abstention_min_rrf,
                "adjacent_chunk_radius": settings.adjacent_chunk_radius,
            },
            metrics=summary,
            artifacts=[args.output_dir / f"{name}.json" for name in (*reports, "answering")],
        )
        print(json.dumps({**summary, "experiment": {"backend": run.backend, "run_id": run.run_id}}, indent=2))
    finally:
        dense.client.close()


if __name__ == "__main__":
    main()

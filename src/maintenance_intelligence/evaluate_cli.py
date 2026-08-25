from __future__ import annotations

import argparse
import json
from pathlib import Path

from maintenance_intelligence.evaluation import (
    evaluate_retriever,
    load_golden_questions,
    save_evaluation_report,
)
from maintenance_intelligence.retrieval import BM25Retriever, KeywordRetriever, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lexical retrieval against the golden question set.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/golden_questions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    questions = load_golden_questions(args.questions)
    retrievers = {
        "term_frequency": KeywordRetriever(chunks),
        "bm25": BM25Retriever(chunks),
    }
    summary: dict[str, object] = {}
    for name, retriever in retrievers.items():
        report = evaluate_retriever(retriever, questions, limit=args.limit)
        save_evaluation_report(report, args.output_dir / f"{name}.json")
        summary[name] = report["metrics"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path

from maintenance_intelligence.retrieval import load_chunks
from maintenance_intelligence.visual_retrieval import (
    VisualBM25Retriever,
    evaluate_visual_retriever,
    load_visual_assets,
    load_visual_questions,
)
from maintenance_intelligence.visual_clip import load_labeled_asset_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text-linked visual candidate retrieval.")
    parser.add_argument("--inventory", type=Path, default=Path("data/processed/visual_assets.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/visual_retrieval_questions.csv"))
    parser.add_argument("--labels", type=Path, default=Path("evaluation/visual_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/visual_retrieval_metrics.json"))
    args = parser.parse_args()
    labeled_ids = set(load_labeled_asset_ids(args.labels))
    assets = [asset for asset in load_visual_assets(args.inventory) if asset.asset_id in labeled_ids]
    if len(assets) != len(labeled_ids):
        raise ValueError("One or more labeled assets are missing from the visual inventory.")
    retriever = VisualBM25Retriever(assets, load_chunks(args.chunks))
    report = evaluate_visual_retriever(retriever, load_visual_questions(args.questions))
    report["candidate_pool_size"] = len(assets)
    report["limitations"] = [
        "The candidate pool contains only 24 manually inspected regions from one product family.",
        "The 12 queries were written after visual inspection and have not been independently authored.",
        "Page-text retrieval does not demonstrate diagram comprehension.",
    ]
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

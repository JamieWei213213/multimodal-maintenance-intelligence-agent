from __future__ import annotations

import argparse
import json
from pathlib import Path

from maintenance_intelligence.retrieval import load_chunks
from maintenance_intelligence.visual_clip import (
    build_clip_retriever,
    load_labeled_asset_ids,
    render_asset_crops,
)
from maintenance_intelligence.visual_retrieval import (
    VisualBM25Retriever,
    VisualRRF,
    evaluate_visual_retriever,
    load_visual_assets,
    load_visual_questions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP text-to-image retrieval on labeled candidates.")
    parser.add_argument("--inventory", type=Path, default=Path("data/processed/visual_assets.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("evaluation/visual_labels.csv"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/visual_retrieval_questions.csv"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--manual-dir", type=Path, default=Path("data/manuals"))
    parser.add_argument("--crop-dir", type=Path, default=Path("evaluation/visual_clip_samples"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/visual_clip_metrics.json"))
    parser.add_argument("--hybrid-output", type=Path, default=Path("evaluation/visual_hybrid_metrics.json"))
    args = parser.parse_args()
    inventory = {asset.asset_id: asset for asset in load_visual_assets(args.inventory)}
    labeled_ids = load_labeled_asset_ids(args.labels)
    missing = [asset_id for asset_id in labeled_ids if asset_id not in inventory]
    if missing:
        raise ValueError(f"Labeled assets missing from inventory: {missing}")
    assets = [inventory[asset_id] for asset_id in labeled_ids]
    crops = render_asset_crops(assets, args.manual_dir, args.crop_dir)
    retriever = build_clip_retriever(assets, crops, args.model_cache)
    questions = load_visual_questions(args.questions)
    report = evaluate_visual_retriever(retriever, questions)
    report["method"] = "Paired CLIP text and pixel embeddings with cosine similarity over 24 manually inspected candidates."
    report["candidate_pool_size"] = len(assets)
    report["limitations"] = [
        "The candidate pool contains only 24 manually inspected regions from one product family.",
        "The 12 queries were written after visual inspection and have not been independently authored.",
        "Retrieval does not demonstrate visual question answering or safe procedure comprehension.",
    ]
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lexical = VisualBM25Retriever(assets, load_chunks(args.chunks))
    hybrid_report = evaluate_visual_retriever(VisualRRF([lexical, retriever]), questions)
    hybrid_report["method"] = "Untuned 1:1 reciprocal-rank fusion of page-text BM25 and CLIP pixel retrieval."
    hybrid_report["candidate_pool_size"] = len(assets)
    hybrid_report["limitations"] = report["limitations"] + [
        "The fusion weights were fixed at 1:1 and have not been tuned on a held-out set."
    ]
    args.hybrid_output.write_text(json.dumps(hybrid_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clip": report["metrics"], "hybrid": hybrid_report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

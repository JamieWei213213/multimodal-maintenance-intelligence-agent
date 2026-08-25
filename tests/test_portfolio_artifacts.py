import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _csv_rows(relative_path: str):
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_published_metrics_match_versioned_evaluation_evidence():
    baseline = _json("evaluation/baseline_metrics.json")
    golden = _csv_rows("evaluation/golden_questions.csv")
    answerable = [row for row in golden if row["answerability"] == "answerable"]
    unanswerable = [row for row in golden if row["answerability"] != "answerable"]
    assert baseline["golden_set"]["answerable_questions"] == len(answerable)
    assert baseline["golden_set"]["unanswerable_questions"] == len(unanswerable)

    labels = _csv_rows("evaluation/visual_labels.csv")
    visual_questions = _csv_rows("evaluation/visual_retrieval_questions.csv")
    assert baseline["visual_candidate_evaluation"]["sample_size"] == len(labels)
    assert baseline["visual_retrieval"]["candidate_pool_size"] == len(labels)
    assert baseline["visual_retrieval"]["questions"] == len(visual_questions)

    lexical = _json("evaluation/visual_retrieval_metrics.json")
    clip = _json("evaluation/visual_clip_metrics.json")
    hybrid = _json("evaluation/visual_hybrid_metrics.json")
    published = baseline["visual_retrieval"]
    assert published["lexical_asset_recall_at_10"] == pytest.approx(lexical["metrics"]["asset_recall_at_10"])
    assert published["clip_asset_recall_at_10"] == pytest.approx(clip["metrics"]["asset_recall_at_10"])
    assert published["hybrid_asset_recall_at_10"] == pytest.approx(hybrid["metrics"]["asset_recall_at_10"])
    assert published["clip_asset_recall_at_3"] == pytest.approx(clip["metrics"]["asset_recall_at_3"])
    assert published["hybrid_asset_recall_at_3"] == pytest.approx(hybrid["metrics"]["asset_recall_at_3"])


def test_corpus_manifest_counts_match_published_snapshot():
    baseline = _json("evaluation/baseline_metrics.json")
    manifest = _json("evaluation/corpus_manifest.json")
    checksums = _csv_rows("evaluation/corpus_checksums.csv")
    assert baseline["corpus"]["documents"] == len(checksums)
    assert baseline["corpus"]["documents"] == manifest["documents"]
    assert baseline["corpus"]["pages"] == manifest["pages"]
    assert baseline["corpus"]["chunks"] == manifest["chunks"]
    assert baseline["corpus"]["table_chunks"] == manifest["table_chunks"]
    assert baseline["corpus"]["safety_chunks"] == manifest["safety_chunks"]

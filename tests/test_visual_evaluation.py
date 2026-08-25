import csv

import pytest

from maintenance_intelligence.visual_evaluation import evaluate_visual_labels


def _write_labels(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "asset_id",
                "source_filename",
                "pdf_page",
                "asset_type",
                "is_meaningful_visual",
                "visual_category",
                "label_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_visual_evaluation_reports_precision_and_strata(tmp_path):
    labels = tmp_path / "labels.csv"
    _write_labels(
        labels,
        [
            {"asset_id": "a", "source_filename": "a.pdf", "pdf_page": 1, "asset_type": "raster_image", "is_meaningful_visual": "yes", "visual_category": "photo", "label_notes": ""},
            {"asset_id": "b", "source_filename": "a.pdf", "pdf_page": 2, "asset_type": "vector_drawing_region", "is_meaningful_visual": "no", "visual_category": "decorative", "label_notes": ""},
        ],
    )
    report = evaluate_visual_labels(labels)
    assert report["sample_size"] == 2
    assert report["sampled_candidate_precision"] == 0.5
    assert len(report["strata"]) == 2


def test_visual_evaluation_rejects_incomplete_labels(tmp_path):
    labels = tmp_path / "labels.csv"
    _write_labels(
        labels,
        [{"asset_id": "a", "source_filename": "a.pdf", "pdf_page": 1, "asset_type": "raster_image", "is_meaningful_visual": "", "visual_category": "", "label_notes": ""}],
    )
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_visual_labels(labels)

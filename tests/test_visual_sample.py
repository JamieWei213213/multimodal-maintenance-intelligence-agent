import csv
import json

import pymupdf

from maintenance_intelligence.visual_sample import create_annotation_sample, stratified_sample


def test_stratified_sample_is_deterministic_and_covers_groups():
    records = [
        {"asset_id": f"{document}-{kind}-{index}", "source_filename": document, "asset_type": kind}
        for document in ("a.pdf", "b.pdf")
        for kind in ("raster_image", "vector_drawing_region")
        for index in range(3)
    ]
    first = stratified_sample(records, 4, seed=580)
    second = stratified_sample(records, 4, seed=580)
    assert first == second
    assert {(row["source_filename"], row["asset_type"]) for row in first} == {
        ("a.pdf", "raster_image"),
        ("a.pdf", "vector_drawing_region"),
        ("b.pdf", "raster_image"),
        ("b.pdf", "vector_drawing_region"),
    }


def test_annotation_sample_renders_crop_and_blank_labels(tmp_path):
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    pdf_path = manuals / "manual.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(40, 40, 160, 160))
    document.save(pdf_path)
    document.close()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "asset_id": "vis_test",
                "source_filename": "manual.pdf",
                "pdf_page": 1,
                "asset_type": "vector_drawing_region",
                "bbox": [40, 40, 160, 160],
                "nearby_text": "Figure 1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "annotations.csv"
    result = create_annotation_sample(inventory, manuals, output, tmp_path / "renders", sample_size=1)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert result["sampled_candidates"] == 1
    assert rows[0]["is_meaningful_visual"] == ""
    assert rows[0]["crop_path"].endswith("001_vis_test.png")
    assert (tmp_path / "renders" / "001_vis_test.png").exists()

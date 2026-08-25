import base64

import pymupdf

from maintenance_intelligence.visuals import build_visual_inventory, inventory_pdf


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_visual_inventory_preserves_page_bbox_and_nearby_text(tmp_path):
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    path = manual_dir / "manual.pdf"
    document = pymupdf.open()
    page = document.new_page(width=300, height=300)
    page.insert_text((40, 40), "Figure 1 Motor wiring diagram")
    page.insert_image(pymupdf.Rect(40, 60, 160, 180), stream=PNG)
    document.save(path)
    document.close()

    records = inventory_pdf(path)
    raster = next(record for record in records if record.asset_type == "raster_image")
    assert raster.source_filename == "manual.pdf"
    assert raster.pdf_page == 1
    assert raster.bbox == [40.0, 60.0, 160.0, 180.0]
    assert "Motor wiring diagram" in raster.nearby_text

    output = tmp_path / "visual_assets.jsonl"
    counts = build_visual_inventory(manual_dir, output)
    assert counts["raster_images"] == 1
    assert output.read_text(encoding="utf-8").count("\n") >= 1

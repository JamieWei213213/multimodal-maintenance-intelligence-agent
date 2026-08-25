import json

from maintenance_intelligence.visual_evidence import VisualEvidenceIndex
from maintenance_intelligence.visuals import VisualAssetRecord


def _record(asset_id="vis_one", page=7):
    return VisualAssetRecord(
        asset_id=asset_id,
        source_filename="manual.pdf",
        pdf_page=page,
        asset_type="raster_image",
        bbox=[10.0, 20.0, 100.0, 120.0],
        width=100,
        height=100,
        nearby_text="Figure 1",
    )


def test_visual_index_returns_only_candidates_on_cited_pages(tmp_path):
    path = tmp_path / "visuals.jsonl"
    path.write_text(json.dumps(_record().to_dict()) + "\n", encoding="utf-8")
    index = VisualEvidenceIndex.load(path)
    result = index.for_citations(
        [{"source_filename": "manual.pdf", "pdf_page": 7}], limit=2
    )
    assert len(index) == 1
    assert result[0]["asset_id"] == "vis_one"
    assert "rendered_path" not in result[0]
    assert index.for_citations([{"source_filename": "manual.pdf", "pdf_page": 8}]) == []


def test_visual_index_deduplicates_repeated_citation_pages():
    index = VisualEvidenceIndex([_record()])
    citations = [
        {"source_filename": "manual.pdf", "pdf_page": 7},
        {"source_filename": "manual.pdf", "pdf_page": 7},
    ]
    assert len(index.for_citations(citations)) == 1

from maintenance_intelligence.models import ChunkRecord
from maintenance_intelligence.visual_retrieval import (
    VisualBM25Retriever,
    VisualGoldenQuestion,
    VisualRRF,
    evaluate_visual_retriever,
)
from maintenance_intelligence.visuals import VisualAssetRecord


def _chunk(page, text):
    return ChunkRecord("c" + str(page), "d", "manual.pdf", "sha", page, str(page), 0, "text", [], text, [], [], "native", False)


def _asset(asset_id, page, caption=""):
    return VisualAssetRecord(asset_id, "manual.pdf", page, "vector_drawing_region", [0, 0, 100, 100], None, None, caption)


def test_visual_retriever_uses_caption_and_same_page_text():
    retriever = VisualBM25Retriever(
        [_asset("fan", 4, "cooling fan"), _asset("wiring", 7)],
        [_chunk(4, "Replace the main cooling fan"), _chunk(7, "Terminal wiring diagram")],
    )
    assert retriever.search("main cooling fan")[0].asset.asset_id == "fan"
    assert retriever.search("terminal wiring")[0].asset.asset_id == "wiring"


def test_visual_evaluation_separates_asset_and_page_recall():
    retriever = VisualBM25Retriever([_asset("expected", 2)], [_chunk(2, "brake parameter table")])
    question = VisualGoldenQuestion("V1", "brake table", "expected", "manual.pdf", 2)
    report = evaluate_visual_retriever(retriever, [question])
    assert report["metrics"]["asset_recall_at_1"] == 1.0
    assert report["metrics"]["page_recall_at_1"] == 1.0


def test_visual_rrf_combines_ranked_results_without_raw_score_calibration():
    class StaticRetriever:
        def __init__(self, ordered):
            self.ordered = ordered

        def search(self, query, limit=10):
            from maintenance_intelligence.visual_retrieval import VisualSearchResult

            return [VisualSearchResult(asset, 999.0 - rank, rank) for rank, asset in enumerate(self.ordered[:limit], 1)]

    first, second, third = _asset("first", 1), _asset("second", 2), _asset("third", 3)
    fused = VisualRRF([StaticRetriever([first, second]), StaticRetriever([second, third])])
    assert fused.search("query")[0].asset.asset_id == "second"

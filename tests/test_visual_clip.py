import numpy as np

from maintenance_intelligence.visual_clip import VisualVectorRetriever
from maintenance_intelligence.visuals import VisualAssetRecord


def _asset(asset_id):
    return VisualAssetRecord(asset_id, "manual.pdf", 1, "raster_image", [0, 0, 10, 10], 10, 10, "")


def test_visual_vector_retriever_ranks_by_cosine_similarity():
    retriever = VisualVectorRetriever(
        [_asset("diagram"), _asset("photo")],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        lambda query: np.array([1.0, 0.0]) if "diagram" in query else np.array([0.0, 1.0]),
    )
    assert retriever.search("wiring diagram")[0].asset.asset_id == "diagram"
    assert retriever.search("product photo")[0].asset.asset_id == "photo"


def test_visual_vector_retriever_rejects_mismatched_vectors():
    try:
        VisualVectorRetriever([_asset("one")], [], lambda _: np.array([1.0]))
    except ValueError as exc:
        assert "counts" in str(exc)
    else:
        raise AssertionError("Expected mismatched vector count to fail")

from maintenance_intelligence.hybrid import HybridRetriever
from maintenance_intelligence.retrieval import SearchResult
from test_grounding import sample_chunk


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, limit=5):
        return self.results[:limit]


def test_rrf_rewards_chunks_found_by_both_retrievers():
    shared = sample_chunk()
    lexical = FakeRetriever([SearchResult(shared, 4.0, 1)])
    dense = FakeRetriever([SearchResult(shared, 0.8, 1)])
    results = HybridRetriever(lexical, dense).search("warning", limit=1)
    assert results[0].lexical_rank == 1
    assert results[0].dense_rank == 1
    assert results[0].rrf_score == 2 / 61


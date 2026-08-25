from __future__ import annotations

from dataclasses import dataclass

from maintenance_intelligence.retrieval import SearchResult


@dataclass(frozen=True)
class HybridResult:
    result: SearchResult
    rrf_score: float
    lexical_rank: int | None
    dense_rank: int | None


class HybridRetriever:
    def __init__(
        self,
        lexical,
        dense,
        *,
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight

    def search(self, query: str, *, limit: int = 5, lexical_limit: int = 20, dense_limit: int = 20) -> list[HybridResult]:
        lexical = self.lexical.search(query, limit=lexical_limit)
        dense = self.dense.search(query, limit=dense_limit)
        combined: dict[str, dict[str, object]] = {}
        for source, label, weight in (
            (lexical, "lexical", self.lexical_weight),
            (dense, "dense", self.dense_weight),
        ):
            for result in source:
                entry = combined.setdefault(
                    result.chunk.chunk_id,
                    {"result": result, "rrf": 0.0, "lexical": None, "dense": None},
                )
                entry["rrf"] = float(entry["rrf"]) + weight / (self.rrf_k + result.rank)
                entry[label] = result.rank
        ranked = sorted(combined.values(), key=lambda entry: -float(entry["rrf"]))
        output: list[HybridResult] = []
        seen_pages: set[tuple[str, int]] = set()
        for entry in ranked:
            result = entry["result"]
            assert isinstance(result, SearchResult)
            citation = (result.chunk.source_filename, result.chunk.pdf_page)
            if citation in seen_pages:
                continue
            seen_pages.add(citation)
            output.append(
                HybridResult(
                    result=result,
                    rrf_score=float(entry["rrf"]),
                    lexical_rank=entry["lexical"] if isinstance(entry["lexical"], int) else None,
                    dense_rank=entry["dense"] if isinstance(entry["dense"], int) else None,
                )
            )
            if len(output) == limit:
                break
        return output

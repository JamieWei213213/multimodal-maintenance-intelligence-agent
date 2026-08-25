from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from maintenance_intelligence.models import ChunkRecord
from maintenance_intelligence.retrieval import is_technical_identifier, tokenize
from maintenance_intelligence.visuals import VisualAssetRecord


@dataclass(frozen=True)
class VisualSearchResult:
    asset: VisualAssetRecord
    score: float
    rank: int


@dataclass(frozen=True)
class VisualGoldenQuestion:
    question_id: str
    question: str
    expected_asset_id: str
    source_document: str
    pdf_page: int


class VisualBM25Retriever:
    """BM25 over caption and same-page text; it never examines visual pixels."""

    def __init__(
        self,
        assets: list[VisualAssetRecord],
        chunks: list[ChunkRecord],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        identifier_boost: float = 2.0,
    ) -> None:
        self.assets = assets
        self.k1 = k1
        self.b = b
        self.identifier_boost = identifier_boost
        page_chunks: dict[tuple[str, int], list[ChunkRecord]] = defaultdict(list)
        for chunk in chunks:
            page_chunks[(chunk.source_filename, chunk.pdf_page)].append(chunk)
        documents = []
        for asset in assets:
            page_text = " ".join(
                chunk.text
                for chunk in sorted(
                    page_chunks.get((asset.source_filename, asset.pdf_page), []),
                    key=lambda item: item.chunk_index,
                )
            )
            documents.append(f"{asset.nearby_text} {page_text} {asset.asset_type.replace('_', ' ')}")
        self._counts = [Counter(tokenize(document)) for document in documents]
        self._lengths = [sum(counts.values()) for counts in self._counts]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._document_frequency: Counter[str] = Counter()
        for counts in self._counts:
            self._document_frequency.update(counts)

    def _idf(self, token: str) -> float:
        total = len(self.assets)
        frequency = self._document_frequency[token]
        return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, limit: int = 10) -> list[VisualSearchResult]:
        query_tokens = list(dict.fromkeys(tokenize(query)))
        if not query_tokens or limit <= 0:
            return []
        scored: list[tuple[VisualAssetRecord, float]] = []
        for asset, counts, length in zip(self.assets, self._counts, self._lengths, strict=True):
            score = 0.0
            for token in query_tokens:
                frequency = counts[token]
                if not frequency:
                    continue
                ratio = length / self._average_length if self._average_length else 0.0
                denominator = frequency + self.k1 * (1 - self.b + self.b * ratio)
                score += self._idf(token) * frequency * (self.k1 + 1) / denominator
                if is_technical_identifier(token):
                    score += self.identifier_boost * self._idf(token)
            if score > 0:
                scored.append((asset, score))
        scored.sort(key=lambda item: (-item[1], item[0].source_filename, item[0].pdf_page, item[0].asset_id))
        return [
            VisualSearchResult(asset=asset, score=score, rank=rank)
            for rank, (asset, score) in enumerate(scored[:limit], start=1)
        ]


class VisualRRF:
    """Reciprocal-rank fusion for retrievers whose raw scores are not comparable."""

    def __init__(self, retrievers: list[object], *, rank_constant: int = 60) -> None:
        if not retrievers:
            raise ValueError("At least one visual retriever is required.")
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive.")
        self.retrievers = retrievers
        self.rank_constant = rank_constant

    def search(self, query: str, limit: int = 10) -> list[VisualSearchResult]:
        if limit <= 0:
            return []
        scores: dict[str, float] = defaultdict(float)
        assets: dict[str, VisualAssetRecord] = {}
        candidate_limit = max(50, limit)
        for retriever in self.retrievers:
            for result in retriever.search(query, limit=candidate_limit):
                scores[result.asset.asset_id] += 1.0 / (self.rank_constant + result.rank)
                assets[result.asset.asset_id] = result.asset
        ordered = sorted(
            scores,
            key=lambda asset_id: (
                -scores[asset_id],
                assets[asset_id].source_filename,
                assets[asset_id].pdf_page,
                asset_id,
            ),
        )[:limit]
        return [
            VisualSearchResult(assets[asset_id], scores[asset_id], rank)
            for rank, asset_id in enumerate(ordered, start=1)
        ]


def load_visual_assets(path: Path) -> list[VisualAssetRecord]:
    return [
        VisualAssetRecord(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_visual_questions(path: Path) -> list[VisualGoldenQuestion]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            VisualGoldenQuestion(
                question_id=row["id"],
                question=row["question"],
                expected_asset_id=row["expected_asset_id"],
                source_document=row["source_document"],
                pdf_page=int(row["pdf_page"]),
            )
            for row in csv.DictReader(handle)
        ]


def evaluate_visual_retriever(
    retriever: VisualBM25Retriever,
    questions: list[VisualGoldenQuestion],
    *,
    limit: int = 10,
) -> dict[str, object]:
    asset_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    page_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_rank = 0.0
    rows = []
    for question in questions:
        results = retriever.search(question.question, limit=limit)
        asset_rank = next((item.rank for item in results if item.asset.asset_id == question.expected_asset_id), None)
        page_rank = next(
            (
                item.rank
                for item in results
                if (item.asset.source_filename, item.asset.pdf_page)
                == (question.source_document, question.pdf_page)
            ),
            None,
        )
        if asset_rank is not None:
            reciprocal_rank += 1 / asset_rank
        for cutoff in asset_hits:
            asset_hits[cutoff] += int(asset_rank is not None and asset_rank <= cutoff)
            page_hits[cutoff] += int(page_rank is not None and page_rank <= cutoff)
        rows.append(
            {
                "id": question.question_id,
                "question": question.question,
                "expected_asset_id": question.expected_asset_id,
                "asset_rank": asset_rank,
                "page_rank": page_rank,
                "top_results": [
                    {
                        "rank": item.rank,
                        "score": round(item.score, 6),
                        "asset_id": item.asset.asset_id,
                        "source_document": item.asset.source_filename,
                        "pdf_page": item.asset.pdf_page,
                        "asset_type": item.asset.asset_type,
                    }
                    for item in results
                ],
            }
        )
    denominator = len(questions) or 1
    return {
        "questions": len(questions),
        "metrics": {
            "asset_mrr": reciprocal_rank / denominator,
            **{f"asset_recall_at_{cutoff}": asset_hits[cutoff] / denominator for cutoff in asset_hits},
            **{f"page_recall_at_{cutoff}": page_hits[cutoff] / denominator for cutoff in page_hits},
        },
        "method": "BM25 over nearby caption text and same-page extracted text; no pixel embeddings or visual reasoning.",
        "limitations": [
            "The 12 questions target a manually inspected subset from one product family.",
            "Queries were written after visual inspection and have not been independently authored.",
            "Page-text retrieval does not demonstrate diagram comprehension.",
        ],
        "results": rows,
    }

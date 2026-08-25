from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from maintenance_intelligence.models import ChunkRecord


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "should", "the", "to", "what", "when",
    "where", "which", "with",
}


@dataclass(frozen=True)
class SearchResult:
    chunk: ChunkRecord
    score: float
    rank: int = 0


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


def load_chunks(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(ChunkRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid chunk record at {path}:{line_number}: {exc}") from exc
    return chunks


def _deduplicate_pages(results: list[SearchResult], limit: int) -> list[SearchResult]:
    deduplicated: list[SearchResult] = []
    seen_pages: set[tuple[str, int]] = set()
    for result in results:
        citation = (result.chunk.source_filename, result.chunk.pdf_page)
        if citation in seen_pages:
            continue
        seen_pages.add(citation)
        deduplicated.append(
            SearchResult(chunk=result.chunk, score=result.score, rank=len(deduplicated) + 1)
        )
        if len(deduplicated) == limit:
            break
    return deduplicated


class KeywordRetriever:
    """Simple term-frequency baseline retained for comparison."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self._token_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens or limit <= 0:
            return []
        results = [
            SearchResult(chunk=chunk, score=float(sum(counts[token] for token in query_tokens)))
            for chunk, counts in zip(self.chunks, self._token_counts, strict=True)
            if any(counts[token] for token in query_tokens)
        ]
        results.sort(
            key=lambda result: (
                -result.score,
                result.chunk.source_filename,
                result.chunk.pdf_page,
                result.chunk.chunk_index,
            )
        )
        return _deduplicate_pages(results, limit)


class BM25Retriever:
    """In-memory BM25 baseline with page-level result deduplication."""

    def __init__(
        self,
        chunks: list[ChunkRecord],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._token_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._lengths = [sum(counts.values()) for counts in self._token_counts]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._document_frequency: Counter[str] = Counter()
        for counts in self._token_counts:
            self._document_frequency.update(counts.keys())

    def _idf(self, token: str) -> float:
        total = len(self.chunks)
        frequency = self._document_frequency[token]
        return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_tokens = list(dict.fromkeys(tokenize(query)))
        if not query_tokens or limit <= 0 or not self.chunks:
            return []

        results: list[SearchResult] = []
        for chunk, counts, length in zip(
            self.chunks,
            self._token_counts,
            self._lengths,
            strict=True,
        ):
            score = 0.0
            for token in query_tokens:
                term_frequency = counts[token]
                if term_frequency == 0:
                    continue
                length_ratio = length / self._average_length if self._average_length else 0.0
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self._idf(token) * (term_frequency * (self.k1 + 1)) / denominator
            if score > 0:
                results.append(SearchResult(chunk=chunk, score=score))

        results.sort(
            key=lambda result: (
                -result.score,
                result.chunk.source_filename,
                result.chunk.pdf_page,
                result.chunk.chunk_index,
            )
        )
        return _deduplicate_pages(results, limit)


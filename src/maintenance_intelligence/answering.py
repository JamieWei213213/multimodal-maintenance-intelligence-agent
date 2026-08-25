from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from dataclasses import asdict

from maintenance_intelligence.citations import make_citation, validate_citation
from maintenance_intelligence.retrieval import tokenize


INSUFFICIENT = (
    "Insufficient evidence in the indexed manuals. Do not infer or perform a maintenance "
    "procedure without the applicable manufacturer documentation and site safety process."
)

CONJUNCTIVE_PROCEDURE_TERMS = {
    "bearing",
    "replacement",
    "procedure",
    "tightening",
    "torque",
}

CONJUNCTIVE_REQUEST_GROUPS = (
    {"plc", "ladder", "logic", "program"},
    {"hydraulic", "pressure"},
    {"encoder", "connector", "pinout"},
    {"static", "ip", "address"},
    {"replacement", "brake", "resistor", "part", "number"},
    {"bearing", "lubricated", "lubrication", "grease"},
    {"arc", "flash", "ppe", "category"},
)

ENCLOSURE_CODES = {"ip21", "ip55", "ip66"}


def _sentences(text: str) -> list[str]:
    # PDF table cells and narrow columns often insert line breaks mid-sentence.
    normalized = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _excerpt(text: str, query_tokens: set[str], limit: int = 800) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    technical = sorted(
        (token for token in query_tokens if any(char.isdigit() for char in token)),
        key=len,
        reverse=True,
    )
    anchors = technical + sorted(query_tokens, key=len, reverse=True)
    lowered = normalized.lower()
    position = next((lowered.find(token) for token in anchors if lowered.find(token) >= 0), 0)
    start = max(0, position - 180)
    end = min(len(normalized), start + limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(normalized) else ""
    return f"{prefix}{normalized[start:end].strip()}{suffix}"


class GroundedAnswerer:
    def __init__(
        self,
        hybrid_retriever,
        chunks_by_id,
        *,
        min_rrf: float = 0.016,
        evidence_limit: int = 5,
        adjacent_chunk_radius: int = 1,
    ) -> None:
        self.hybrid = hybrid_retriever
        self.chunks_by_id = chunks_by_id
        self.min_rrf = min_rrf
        self.evidence_limit = evidence_limit
        self.adjacent_chunk_radius = max(0, adjacent_chunk_radius)
        page_chunks = defaultdict(list)
        for chunk in chunks_by_id.values():
            page_chunks[(chunk.source_filename, chunk.pdf_page)].append(chunk)
        self.page_chunks = {
            key: sorted(chunks, key=lambda chunk: chunk.chunk_index)
            for key, chunks in page_chunks.items()
        }
        self.document_frequency: Counter[str] = Counter()
        for chunk in chunks_by_id.values():
            self.document_frequency.update(set(tokenize(chunk.text)))
        self.corpus_size = max(1, len(chunks_by_id))

    def _support_score(self, context: str, query_tokens: set[str]) -> float:
        context_tokens = set(tokenize(context))
        overlap = query_tokens.intersection(context_tokens)
        weighted_overlap = sum(
            math.log((self.corpus_size + 1) / (self.document_frequency[token] + 1)) + 1.0
            for token in overlap
        )
        return weighted_overlap / math.sqrt(max(1, len(context_tokens)))

    def _selection_score(
        self,
        context: str,
        query_tokens: set[str],
        source_filename: str,
        pdf_page: int,
    ) -> float:
        score = self._support_score(context, query_tokens)
        lowered = context.lower()
        if "quick" in query_tokens and "start" in query_tokens and "quickstart" in source_filename.lower():
            score += 3.0
        if "firmware" in query_tokens and "firmware" in source_filename.lower():
            score += 3.0
        if "hardware" in query_tokens and "hardware" in source_filename.lower():
            score += 3.0
        if "introduction to the manual" in lowered or "related manuals" in lowered:
            score *= 0.5
        if pdf_page <= 3 and not query_tokens.intersection({"title", "document", "revision"}):
            score *= 0.35
        non_ascii = sum(ord(character) > 127 for character in context)
        if context and non_ascii / len(context) > 0.08:
            score *= 0.2
        requested_enclosures = query_tokens.intersection(ENCLOSURE_CODES)
        context_enclosures = set(tokenize(context)).intersection(ENCLOSURE_CODES)
        if requested_enclosures and context_enclosures.difference(requested_enclosures):
            score *= 0.5
        metric_sizes = [token for token in query_tokens if re.fullmatch(r"m\d+", token)]
        if "torque" in query_tokens and metric_sizes:
            normalized = re.sub(r"\s+", " ", lowered)
            for size in metric_sizes:
                near_torque = re.search(
                    rf"(?:{size}.{{0,100}}\d+(?:\.\d+)?\s*n[·. ]?m|"
                    rf"\d+(?:\.\d+)?\s*n[·. ]?m.{{0,100}}{size})",
                    normalized,
                )
                score += 5.0 if near_torque else -2.0
        return score

    def _context_chunks(self, chunk):
        page = self.page_chunks.get((chunk.source_filename, chunk.pdf_page), [chunk])
        position = next(
            (index for index, candidate in enumerate(page) if candidate.chunk_id == chunk.chunk_id),
            0,
        )
        start = max(0, position - self.adjacent_chunk_radius)
        end = min(len(page), position + self.adjacent_chunk_radius + 1)
        return page[start:end]

    def answer(self, question: str) -> dict[str, object]:
        results = self.hybrid.search(question, limit=self.evidence_limit)
        query_tokens = set(tokenize(question))
        if not results or results[0].rrf_score < self.min_rrf:
            return {"answer": INSUFFICIENT, "abstained": True, "citations": [], "confidence": 0.0}

        context_groups = [self._context_chunks(item.result.chunk) for item in results]
        requested_procedure_terms = query_tokens.intersection(CONJUNCTIVE_PROCEDURE_TERMS)
        if len(requested_procedure_terms) >= 2 and not any(
            requested_procedure_terms.issubset(
                set(tokenize(" ".join(chunk.text for chunk in context_chunks)))
            )
            for context_chunks in context_groups
        ):
            return {"answer": INSUFFICIENT, "abstained": True, "citations": [], "confidence": 0.0}

        for group in CONJUNCTIVE_REQUEST_GROUPS:
            requested = query_tokens.intersection(group)
            if len(requested) >= 2 and not any(
                requested.issubset(set(tokenize(" ".join(chunk.text for chunk in context_chunks))))
                for context_chunks in context_groups
            ):
                return {"answer": INSUFFICIENT, "abstained": True, "citations": [], "confidence": 0.0}

        supported = []
        for position, (result, context_chunks) in enumerate(zip(results, context_groups, strict=True)):
            context = " ".join(chunk.text for chunk in context_chunks)
            supported.append((
                self._selection_score(
                    context,
                    query_tokens,
                    result.result.chunk.source_filename,
                    result.result.chunk.pdf_page,
                ),
                position,
                context,
                context_chunks,
            ))
        overlap, _, primary_context, primary_chunks = max(
            supported,
            key=lambda item: (item[0], -item[1]),
        )
        if overlap == 0:
            return {"answer": INSUFFICIENT, "abstained": True, "citations": [], "confidence": 0.0}
        citations = [make_citation(chunk) for chunk in primary_chunks]
        citations = [citation for citation in citations if validate_citation(citation, self.chunks_by_id)]
        if not citations:
            return {"answer": INSUFFICIENT, "abstained": True, "citations": [], "confidence": 0.0}
        answer = "Evidence from the manual: " + _excerpt(primary_context, query_tokens)
        safety = any(chunk.contains_safety_warning for chunk in primary_chunks)
        if safety:
            answer = "Safety-critical evidence applies. " + answer
        confidence = min(1.0, results[0].rrf_score / max(self.min_rrf * 2, 1e-9))
        return {
            "answer": answer,
            "abstained": False,
            "confidence": round(confidence, 3),
            "citations": [asdict(citation) for citation in citations],
        }

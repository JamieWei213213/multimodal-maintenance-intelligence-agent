from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from maintenance_intelligence.answering import INSUFFICIENT


FAULT_CODE = re.compile(r"\b(?=[A-Z0-9]{3,8}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]+\b", re.I)


@dataclass(frozen=True)
class AgentPolicy:
    """Hard limits for the read-only troubleshooting controller."""

    max_tool_calls: int = 5
    max_searches: int = 3


class MaintenanceAgent:
    """A bounded controller over grounded retrieval tools.

    The controller has no equipment-control capability and never generates a new
    maintenance procedure. It can only select searches, combine extractive
    evidence, validate citations, preserve safety flags, or abstain.
    """

    def __init__(
        self,
        answerer: Any,
        chunks_by_id: dict[str, Any],
        visual_index: Any,
        *,
        policy: AgentPolicy | None = None,
    ) -> None:
        self.answerer = answerer
        self.chunks_by_id = chunks_by_id
        self.visual_index = visual_index
        self.policy = policy or AgentPolicy()

    @staticmethod
    def _classify(question: str) -> tuple[str, list[str]]:
        lowered = question.lower()
        codes = FAULT_CODE.findall(question)
        if any(term in lowered for term in ("compare", "difference", "across", "both manuals")):
            return "multi_document_comparison", [
                question,
                f"hardware manual {question}",
                f"firmware manual {question}",
            ]
        if any(term in lowered for term in ("diagram", "drawing", "schematic", "table")):
            return "visual_evidence", [question]
        if codes:
            return "fault_code", [question]
        return "manual_lookup", [question]

    def _valid_citations(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for citation in citations:
            chunk = self.chunks_by_id.get(citation.get("chunk_id"))
            key = (
                str(citation.get("source_filename", "")),
                int(citation.get("pdf_page", 0)),
                str(citation.get("chunk_id", "")),
            )
            if (
                chunk is not None
                and citation.get("source_filename") == chunk.source_filename
                and citation.get("pdf_page") == chunk.pdf_page
                and citation.get("printed_page") == chunk.printed_page
                and bool(citation.get("evidence"))
                and str(citation["evidence"]) in chunk.text
                and key not in seen
            ):
                seen.add(key)
                valid.append(citation)
        return valid

    def run(self, question: str) -> dict[str, object]:
        task_type, queries = self._classify(question)
        queries = queries[: self.policy.max_searches]
        trace: list[dict[str, object]] = []
        supported: list[dict[str, object]] = []

        for query in queries:
            if len(trace) >= self.policy.max_tool_calls:
                break
            response = self.answerer.answer(query)
            trace.append(
                {
                    "tool": "search_manuals",
                    "input": {"query": query},
                    "outcome": "abstained" if response.get("abstained") else "evidence_found",
                    "citation_count": len(response.get("citations", [])),
                }
            )
            if not response.get("abstained"):
                supported.append(response)

        citations = self._valid_citations(
            [
                citation
                for response in supported
                for citation in response.get("citations", [])
            ]
        )
        validated_chunk_ids = {citation["chunk_id"] for citation in citations}
        # A response only stays "supported" if at least one of its own citations
        # survived validation; otherwise its answer text has no verified evidence
        # behind it and must not reach the final answer.
        supported = [
            response
            for response in supported
            if any(
                citation.get("chunk_id") in validated_chunk_ids
                for citation in response.get("citations", [])
            )
        ]
        if len(trace) < self.policy.max_tool_calls:
            trace.append({
                "tool": "validate_citations",
                "outcome": "passed" if citations else "failed",
                "validated_count": len(citations),
            })

        safety_applies = any(
            self.chunks_by_id[citation["chunk_id"]].contains_safety_warning
            for citation in citations
        )
        if len(trace) < self.policy.max_tool_calls:
            trace.append({
                "tool": "check_safety_warnings",
                "outcome": "safety_critical" if safety_applies else "no_embedded_warning",
            })

        if not supported or not citations:
            return {
                "answer": INSUFFICIENT,
                "abstained": True,
                "confidence": 0.0,
                "citations": [],
                "visual_evidence": [],
                "agent": {
                    "task_type": task_type,
                    "policy": "bounded_read_only",
                    "tool_calls": len(trace),
                    "trace": trace,
                },
            }

        # Answers are extractive outputs from the grounded answerer. For a comparison,
        # retain evidence from distinct documents without synthesizing an unsupported procedure.
        selected: list[dict[str, object]] = []
        seen_documents: set[str] = set()
        for response in sorted(
            supported,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        ):
            documents = {item["source_filename"] for item in response.get("citations", [])}
            if not selected or documents.difference(seen_documents):
                selected.append(response)
                seen_documents.update(documents)
            if task_type != "multi_document_comparison" or len(selected) == 2:
                break
        answer = "\n\n".join(str(response["answer"]) for response in selected)
        if safety_applies and not answer.startswith("Safety-critical evidence applies."):
            answer = "Safety-critical evidence applies. " + answer
        confidence = min(float(response.get("confidence", 0.0)) for response in selected)
        visual_evidence = self.visual_index.for_citations(citations) if task_type == "visual_evidence" else []
        return {
            "answer": answer,
            "abstained": False,
            "confidence": round(confidence, 3),
            "citations": citations,
            "visual_evidence": visual_evidence,
            "agent": {
                "task_type": task_type,
                "policy": "bounded_read_only",
                "tool_calls": len(trace),
                "trace": trace,
            },
        }

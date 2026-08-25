from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from maintenance_intelligence.retrieval import SearchResult


class Retriever(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...


class Answerer(Protocol):
    def answer(self, question: str) -> dict[str, object]: ...


@dataclass(frozen=True)
class GoldenQuestion:
    question_id: str
    question: str
    source_document: str
    pdf_page: int | None
    answerability: str


def load_golden_questions(path: Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            raw_page = row.get("pdf_page", "").strip()
            questions.append(
                GoldenQuestion(
                    question_id=row["id"],
                    question=row["question"],
                    source_document=row.get("source_document", "").strip(),
                    pdf_page=int(raw_page) if raw_page else None,
                    answerability=row["answerability"].strip(),
                )
            )
    return questions


def evaluate_retriever(
    retriever: Retriever,
    questions: list[GoldenQuestion],
    *,
    limit: int = 10,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    reciprocal_rank_sum = 0.0
    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    evaluated = 0

    for question in questions:
        if question.answerability != "answerable" or not question.source_document or question.pdf_page is None:
            rows.append({
                "id": question.question_id,
                "status": "skipped_insufficient_evidence",
                "question": question.question,
            })
            continue

        evaluated += 1
        results = retriever.search(question.question, limit=limit)
        expected = (question.source_document, question.pdf_page)
        rank = next(
            (
                result.rank
                for result in results
                if (result.chunk.source_filename, result.chunk.pdf_page) == expected
            ),
            None,
        )
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
            for cutoff in hit_counts:
                if rank <= cutoff:
                    hit_counts[cutoff] += 1
        rows.append({
            "id": question.question_id,
            "status": "hit" if rank is not None else "miss",
            "question": question.question,
            "expected": {"source_document": expected[0], "pdf_page": expected[1]},
            "rank": rank,
            "top_results": [
                {
                    "rank": result.rank,
                    "score": round(result.score, 6),
                    "source_document": result.chunk.source_filename,
                    "pdf_page": result.chunk.pdf_page,
                    "chunk_id": result.chunk.chunk_id,
                }
                for result in results
            ],
        })

    denominator = evaluated or 1
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluated_questions": evaluated,
        "skipped_questions": len(questions) - evaluated,
        "metrics": {
            "mrr": reciprocal_rank_sum / denominator,
            **{f"recall_at_{cutoff}": hit_counts[cutoff] / denominator for cutoff in hit_counts},
        },
        "questions": rows,
    }


def save_evaluation_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_answerer(
    answerer: Answerer,
    questions: list[GoldenQuestion],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    answerable = 0
    answered = 0
    citation_hits = 0
    unanswerable = 0
    correct_abstentions = 0

    for question in questions:
        response = answerer.answer(question.question)
        abstained = bool(response["abstained"])
        citations = response.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        citation_hit = False
        if question.answerability == "answerable":
            answerable += 1
            answered += int(not abstained)
            citation_hit = any(
                isinstance(citation, dict)
                and citation.get("source_filename") == question.source_document
                and citation.get("pdf_page") == question.pdf_page
                for citation in citations
            )
            citation_hits += int(not abstained and citation_hit)
            correct = not abstained and citation_hit
        else:
            unanswerable += 1
            correct_abstentions += int(abstained)
            correct = abstained
        rows.append(
            {
                "id": question.question_id,
                "answerability": question.answerability,
                "abstained": abstained,
                "citation_hit": citation_hit,
                "correct_decision": correct,
                "confidence": response.get("confidence"),
                "answer": response.get("answer"),
            }
        )

    answerable_denominator = answerable or 1
    unanswerable_denominator = unanswerable or 1
    total = len(questions) or 1
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions": rows,
        "metrics": {
            "answerable_coverage": answered / answerable_denominator,
            "answerable_citation_accuracy": citation_hits / answerable_denominator,
            "unanswerable_abstention_recall": correct_abstentions / unanswerable_denominator,
            "unanswerable_false_answer_rate": (unanswerable - correct_abstentions) / unanswerable_denominator,
            "overall_decision_accuracy": (citation_hits + correct_abstentions) / total,
        },
        "counts": {
            "answerable": answerable,
            "unanswerable": unanswerable,
            "answered": answered,
            "citation_hits": citation_hits,
            "correct_abstentions": correct_abstentions,
        },
    }

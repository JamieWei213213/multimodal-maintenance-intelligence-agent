from maintenance_intelligence.evaluation import GoldenQuestion, evaluate_answerer


class FakeAnswerer:
    def answer(self, question: str):
        if "unsupported" in question:
            return {"answer": "Insufficient", "abstained": True, "confidence": 0.0, "citations": []}
        return {
            "answer": "Supported",
            "abstained": False,
            "confidence": 0.9,
            "citations": [{"source_filename": "manual.pdf", "pdf_page": 7}],
        }


def test_answer_evaluation_measures_citations_and_abstention():
    questions = [
        GoldenQuestion("Q1", "supported question", "manual.pdf", 7, "answerable"),
        GoldenQuestion("Q2", "unsupported question", "", None, "insufficient_evidence"),
    ]
    report = evaluate_answerer(FakeAnswerer(), questions)
    assert report["metrics"] == {
        "answerable_coverage": 1.0,
        "answerable_citation_accuracy": 1.0,
        "unanswerable_abstention_recall": 1.0,
        "unanswerable_false_answer_rate": 0.0,
        "overall_decision_accuracy": 1.0,
    }

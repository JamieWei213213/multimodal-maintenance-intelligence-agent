from types import SimpleNamespace

from maintenance_intelligence.agent import MaintenanceAgent


def chunk(chunk_id, filename, *, safety=False):
    return SimpleNamespace(
        chunk_id=chunk_id,
        source_filename=filename,
        pdf_page=4,
        printed_page="4",
        text=f"Evidence {chunk_id}",
        contains_safety_warning=safety,
    )


def citation(item):
    return {
        "chunk_id": item.chunk_id,
        "source_filename": item.source_filename,
        "pdf_page": item.pdf_page,
        "printed_page": item.printed_page,
        "evidence": item.text,
        "bboxes": [],
    }


class FakeAnswerer:
    def __init__(self, chunks):
        self.chunks = chunks

    def answer(self, query):
        item = self.chunks["hardware"] if "hardware manual" in query else self.chunks["firmware"]
        return {
            "answer": f"Evidence from the manual: {item.text}",
            "abstained": False,
            "confidence": 0.8,
            "citations": [citation(item)],
        }


class FakeVisualIndex:
    def for_citations(self, citations):
        return [{"asset_id": "visual-1"}] if citations else []


def test_agent_compares_documents_with_bounded_validated_trace():
    chunks = {
        "hardware": chunk("hardware", "hardware.pdf", safety=True),
        "firmware": chunk("firmware", "firmware.pdf"),
    }
    result = MaintenanceAgent(FakeAnswerer(chunks), chunks, FakeVisualIndex()).run(
        "Compare the hardware and firmware guidance"
    )
    assert not result["abstained"]
    assert {item["source_filename"] for item in result["citations"]} == {"hardware.pdf", "firmware.pdf"}
    assert result["answer"].startswith("Safety-critical evidence applies.")
    assert result["agent"]["tool_calls"] <= 5
    assert result["agent"]["policy"] == "bounded_read_only"


def test_agent_abstains_when_citation_cannot_be_validated():
    item = chunk("firmware", "firmware.pdf")
    result = MaintenanceAgent(FakeAnswerer({"firmware": item, "hardware": item}), {}, FakeVisualIndex()).run(
        "What does A2B1 mean?"
    )
    assert result["abstained"]
    assert result["citations"] == []


class SequencedAnswerer:
    """Returns canned responses in call order so a test can script the tool trace."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.queries = []

    def answer(self, query):
        self.queries.append(query)
        return self.responses.pop(0)


def test_agent_drops_answer_text_whose_citations_fail_validation():
    real = chunk("hardware", "hardware.pdf")
    forged = chunk("ghost", "firmware.pdf")
    forged_citation = citation(forged)
    forged_citation["evidence"] = "text that is not in the indexed chunk"
    answerer = SequencedAnswerer(
        [
            {"answer": "Grounded hardware evidence.", "abstained": False, "confidence": 0.9, "citations": [citation(real)]},
            {"answer": "UNGROUNDED firmware text.", "abstained": False, "confidence": 0.95, "citations": [forged_citation]},
            {"answer": "Grounded hardware evidence.", "abstained": False, "confidence": 0.9, "citations": [citation(real)]},
        ]
    )
    result = MaintenanceAgent(answerer, {"hardware": real, "ghost": forged}, FakeVisualIndex()).run(
        "Compare the hardware and firmware guidance"
    )
    assert not result["abstained"]
    assert "UNGROUNDED" not in result["answer"]
    assert [item["chunk_id"] for item in result["citations"]] == ["hardware"]


def test_agent_fault_code_question_runs_single_bounded_search():
    item = chunk("firmware", "firmware.pdf")
    answerer = SequencedAnswerer(
        [{"answer": "A2B1 means overcurrent.", "abstained": False, "confidence": 0.7, "citations": [citation(item)]}]
    )
    result = MaintenanceAgent(answerer, {"firmware": item}, FakeVisualIndex()).run("What does warning A2B1 mean?")
    assert result["agent"]["task_type"] == "fault_code"
    assert answerer.queries == ["What does warning A2B1 mean?"]
    assert [step["tool"] for step in result["agent"]["trace"]] == [
        "search_manuals",
        "validate_citations",
        "check_safety_warnings",
    ]
    assert result["visual_evidence"] == []


def test_agent_attaches_visual_evidence_only_for_visual_requests():
    item = chunk("hardware", "hardware.pdf")
    answerer = SequencedAnswerer(
        [{"answer": "See the wiring overview.", "abstained": False, "confidence": 0.6, "citations": [citation(item)]}]
    )
    result = MaintenanceAgent(answerer, {"hardware": item}, FakeVisualIndex()).run("Show the wiring diagram for the drive")
    assert result["agent"]["task_type"] == "visual_evidence"
    assert result["visual_evidence"] == [{"asset_id": "visual-1"}]


def test_agent_abstains_when_every_search_abstains():
    answerer = SequencedAnswerer(
        [{"answer": "insufficient", "abstained": True, "confidence": 0.0, "citations": []}]
    )
    result = MaintenanceAgent(answerer, {}, FakeVisualIndex()).run("How do I bake bread?")
    assert result["abstained"]
    assert result["confidence"] == 0.0
    assert result["agent"]["trace"][0]["outcome"] == "abstained"

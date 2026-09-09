"""Pydantic response contracts for every answer-producing route.

The answerer and agent still return plain dicts internally (they predate this
module and are exercised by dozens of tests), so the API validates those dicts
against these models at the boundary. A response that fails validation is a
server bug and surfaces as a 500 rather than leaking a malformed payload.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=2, max_length=1000)


class Citation(BaseModel):
    chunk_id: str
    source_filename: str
    pdf_page: int = Field(ge=1)
    printed_page: str
    evidence: str = Field(min_length=1)
    bboxes: list[list[float]] = Field(default_factory=list)


class VisualEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_id: str
    source_filename: str
    pdf_page: int = Field(ge=1)


class GuardrailReport(BaseModel):
    input_flags: list[str] = Field(default_factory=list)
    control_intent: bool = False
    grounded_sentence_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_sentences: list[str] = Field(default_factory=list)


class AgentTraceStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool: str
    outcome: str


class AgentReport(BaseModel):
    task_type: str
    policy: Literal["bounded_read_only"]
    tool_calls: int = Field(ge=0)
    trace: list[AgentTraceStep]


class LLMUsage(BaseModel):
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)
    cached: bool = False


class AnswerResponse(BaseModel):
    """Contract shared by ``POST /ask`` and ``POST /agent/query``."""

    answer: str = Field(min_length=1)
    abstained: bool
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation]
    visual_evidence: list[VisualEvidence] = Field(default_factory=list)
    mode: Literal["extractive", "agent", "synthesis"] = "extractive"
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    agent: AgentReport | None = None
    llm: LLMUsage | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, mode: str) -> "AnswerResponse":
        data = dict(payload)
        data.setdefault("visual_evidence", [])
        data["mode"] = mode
        return cls.model_validate(data)


class GuardrailRejection(BaseModel):
    detail: str
    flags: list[str]

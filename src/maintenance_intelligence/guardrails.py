"""Input and output guardrails.

Three layers, all deterministic and unit-tested:

1. **Input guard** – rejects prompt-injection attempts, control characters, and
   payloads that are not a maintenance question; flags (but does not reject)
   requests that ask the system to *operate* equipment, because the system has no
   such capability and the answer must say so rather than pretend.
2. **Output grounding guard** – every sentence in an answer must be a substring of
   the text of a validated citation. The extractive answerer satisfies this by
   construction; the check exists so an LLM synthesis mode (or a regression in the
   extractive path) cannot ship an unsupported sentence.
3. **Abstention** – already implemented in :mod:`answering`; the report surfaces
   it as a first-class safety outcome instead of an error.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable


INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all)\b.{0,30}\b(instructions?|rules?|prompts?)\b", re.I | re.S)),
    ("role_override", re.compile(r"\b(you are now|act as|pretend (to be|you are)|from now on you)\b", re.I)),
    ("system_prompt_probe", re.compile(r"\b(system prompt|developer message|hidden instructions?)\b", re.I)),
    ("chat_template_tokens", re.compile(r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[INST\]|<<SYS>>|<\|endoftext\|>)", re.I)),
    ("role_tag", re.compile(r"^\s*(system|assistant)\s*:", re.I | re.M)),
    ("exfiltration", re.compile(r"\b(reveal|print|dump|show)\b.{0,30}\b(api key|secret|password|credentials?|environment variables?)\b", re.I)),
)

CONTROL_INTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(set|change|write|modify|override|update)\b.{0,40}\bparameter\b", re.I),
    re.compile(r"\b(start|stop|run|reset|reboot|restart|enable|disable)\b.{0,20}\bthe\s+(drive|motor|inverter|unit)\b", re.I),
    re.compile(r"\b(bypass|defeat|disable|override)\b.{0,30}\b(safety|sto|safe torque off|interlock|lockout)\b", re.I),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")
CONTROL_INTENT_NOTICE = (
    "This system cannot operate equipment or change drive parameters; it only "
    "retrieves manufacturer documentation. "
)

# Framing text added by the answerer/agent/guard, never a claim about the manuals.
_ANSWER_PREFIXES = (
    CONTROL_INTENT_NOTICE.strip(),
    "Safety-critical evidence applies.",
    "Evidence from the manual:",
)


@dataclass(frozen=True)
class InputDecision:
    allowed: bool
    flags: list[str] = field(default_factory=list)
    control_intent: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class GroundingReport:
    grounded: bool
    grounded_sentence_ratio: float
    unsupported_sentences: list[str]
    checked_sentences: int


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def check_input(question: str, *, max_length: int = 1000) -> InputDecision:
    """Deterministic pre-flight on a raw question."""
    flags: list[str] = []
    if len(question) > max_length:
        return InputDecision(False, ["too_long"], reason="Question exceeds the maximum length")
    if any(unicodedata.category(char) == "Cc" and char not in "\n\t\r" for char in question):
        flags.append("control_characters")
    stripped = _normalize(question)
    if len(stripped) < 2:
        return InputDecision(False, ["empty"], reason="Question is empty")
    letters = sum(char.isalpha() for char in stripped)
    if letters < max(2, int(len(stripped) * 0.2)):
        flags.append("low_text_ratio")
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(question):
            flags.append(f"injection:{name}")
    control_intent = any(pattern.search(question) for pattern in CONTROL_INTENT_PATTERNS)
    if control_intent:
        flags.append("control_intent")
    blocked = [flag for flag in flags if flag.startswith("injection:") or flag == "control_characters"]
    if blocked:
        return InputDecision(
            False,
            flags,
            control_intent=control_intent,
            reason="The question contains prompt-injection or unsafe control content and was not processed",
        )
    return InputDecision(True, flags, control_intent=control_intent)


def _cited_text(citations: Iterable[dict[str, Any]], chunks_by_id: dict[str, Any]) -> str:
    parts: list[str] = []
    for citation in citations:
        chunk = chunks_by_id.get(citation.get("chunk_id"))
        if chunk is not None:
            parts.append(_normalize(chunk.text))
        elif citation.get("evidence"):
            parts.append(_normalize(str(citation["evidence"])))
    # Joined with a single space, matching how the answerer concatenates adjacent
    # chunks, so a sentence that straddles two overlapping chunks still matches.
    return " ".join(parts)


def answer_sentences(answer: str) -> list[str]:
    text = answer
    for prefix in _ANSWER_PREFIXES:
        text = text.replace(prefix, " ")
    text = text.replace("…", " ").replace("...", " ")
    normalized = _normalize(text)
    return [part.strip() for part in _SENTENCE_SPLIT.split(normalized) if len(part.strip()) >= 12]


def check_grounding(
    answer: str,
    citations: Iterable[dict[str, Any]],
    chunks_by_id: dict[str, Any],
    *,
    min_ratio: float = 1.0,
) -> GroundingReport:
    """Every answer sentence must appear verbatim in a cited chunk.

    Sentences shorter than 12 characters are ignored (fragments such as
    "See 3." carry no claim). Comparison is whitespace- and Unicode-normalized
    and case-insensitive, nothing looser: paraphrase is exactly what this guard
    exists to catch.
    """
    haystack = _cited_text(citations, chunks_by_id).lower()
    sentences = answer_sentences(answer)
    if not sentences:
        return GroundingReport(True, 1.0, [], 0)
    unsupported = [sentence for sentence in sentences if sentence.lower() not in haystack]
    ratio = (len(sentences) - len(unsupported)) / len(sentences)
    return GroundingReport(ratio >= min_ratio, round(ratio, 3), unsupported, len(sentences))


def apply_output_guard(
    response: dict[str, Any],
    chunks_by_id: dict[str, Any],
    *,
    input_decision: InputDecision,
    insufficient_text: str,
) -> dict[str, Any]:
    """Attach a guardrail report and force abstention on ungrounded answers."""
    report = {
        "input_flags": list(input_decision.flags),
        "control_intent": input_decision.control_intent,
        "grounded_sentence_ratio": 1.0,
        "unsupported_sentences": [],
    }
    if not response.get("abstained"):
        grounding = check_grounding(str(response.get("answer", "")), response.get("citations", []), chunks_by_id)
        report["grounded_sentence_ratio"] = grounding.grounded_sentence_ratio
        report["unsupported_sentences"] = grounding.unsupported_sentences[:5]
        if not grounding.grounded:
            response = {
                **response,
                "answer": insufficient_text,
                "abstained": True,
                "confidence": 0.0,
                "citations": [],
                "visual_evidence": [],
            }
    if input_decision.control_intent and not response.get("abstained"):
        response = {**response, "answer": CONTROL_INTENT_NOTICE + str(response["answer"])}
    response = {**response, "guardrails": report}
    return response

"""Schema-enforced LLM calls with bounded retries and a deterministic fallback.

``complete_structured`` asks the provider for JSON matching a Pydantic model,
validates the reply, feeds validation errors back for up to ``max_attempts``
tries, and returns ``fallback()`` when the model still cannot comply. The caller
always gets a valid instance of the requested model or the fallback, never an
exception from a malformed reply; provider outages surface as
:class:`StructuredOutputFailure` only when no fallback is supplied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from maintenance_intelligence.llm import LLMClient, LLMError, LLMResult, LLMUnavailable

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


class StructuredOutputFailure(RuntimeError):
    pass


@dataclass
class StructuredAttempt:
    attempt: int
    error: str | None
    result: LLMResult | None = None


@dataclass
class StructuredOutcome:
    value: BaseModel
    attempts: list[StructuredAttempt] = field(default_factory=list)
    used_fallback: bool = False

    @property
    def prompt_tokens(self) -> int:
        return sum(item.result.prompt_tokens for item in self.attempts if item.result)

    @property
    def completion_tokens(self) -> int:
        return sum(item.result.completion_tokens for item in self.attempts if item.result)

    @property
    def cost_usd(self) -> float:
        return round(sum(item.result.cost_usd for item in self.attempts if item.result), 8)

    @property
    def latency_seconds(self) -> float:
        return sum(item.result.latency_seconds for item in self.attempts if item.result)


def extract_json(text: str) -> Any:
    """Parse JSON from a reply, tolerating code fences and leading prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.S)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(stripped)
        if not match:
            raise
        return json.loads(match.group(0))


def complete_structured(
    client: LLMClient,
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    fallback: Callable[[], T] | None = None,
    max_attempts: int = 3,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> StructuredOutcome:
    attempts: list[StructuredAttempt] = []
    conversation = list(messages)
    json_schema = schema.model_json_schema()
    for attempt in range(1, max_attempts + 1):
        try:
            result = client.complete(
                conversation,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )
        except (LLMUnavailable, LLMError) as exc:
            attempts.append(StructuredAttempt(attempt, f"provider_error: {exc}"))
            break  # a provider failure is not fixed by re-prompting
        try:
            value = schema.model_validate(extract_json(result.text))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
            attempts.append(StructuredAttempt(attempt, error, result))
            conversation = conversation + [
                {"role": "assistant", "content": result.text},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid for the required JSON schema. "
                        f"Error: {error}. Reply with only a JSON object matching this schema: "
                        + json.dumps(json_schema)
                    ),
                },
            ]
            continue
        attempts.append(StructuredAttempt(attempt, None, result))
        return StructuredOutcome(value=value, attempts=attempts)
    if fallback is None:
        raise StructuredOutputFailure(
            f"No valid {schema.__name__} after {len(attempts)} attempt(s): {attempts[-1].error if attempts else 'no attempts'}"
        )
    return StructuredOutcome(value=fallback(), attempts=attempts, used_fallback=True)

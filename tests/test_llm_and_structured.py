from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from maintenance_intelligence.llm import (
    FakeLLM,
    LLMError,
    LLMUnavailable,
    OpenAIClient,
    UnavailableLLM,
    build_llm_client,
    estimate_cost_usd,
)
from maintenance_intelligence.structured import (
    StructuredOutputFailure,
    complete_structured,
    extract_json,
)


class Verdict(BaseModel):
    supported: bool
    unsupported_claim_count: int = Field(ge=0)


def test_cost_estimate_uses_price_table_and_prefix_matching():
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost_usd("gpt-4o-mini-2024-07-18", 0, 1_000_000) == pytest.approx(0.60)
    assert estimate_cost_usd("unknown-model", 1000, 1000) == 0.0


def test_unavailable_llm_raises_actionable_error():
    with pytest.raises(LLMUnavailable, match="LLM_PROVIDER"):
        UnavailableLLM().complete([{"role": "user", "content": "hi"}])


def test_build_llm_client_defaults_to_unavailable_and_validates_provider():
    assert isinstance(build_llm_client(SimpleNamespace(llm_provider="none")), UnavailableLLM)
    with pytest.raises(LLMUnavailable):
        build_llm_client(SimpleNamespace(llm_provider="openai", openai_api_key=None, llm_model="gpt-4o-mini", llm_timeout_seconds=5))
    with pytest.raises(LLMUnavailable, match="Unknown"):
        build_llm_client(SimpleNamespace(llm_provider="banana"))


def test_openai_client_normalizes_sdk_response_and_errors():
    class FakeCompletions:
        def __init__(self, fail=False):
            self.fail = fail
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            if self.fail:
                raise RuntimeError("rate limited")
            return SimpleNamespace(
                model="gpt-4o-mini-2024-07-18",
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    completions = FakeCompletions()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAIClient(client=sdk, default_model="gpt-4o-mini")
    result = client.complete([{"role": "user", "content": "x"}], json_schema={"title": "Verdict", "type": "object"})
    assert result.text == '{"ok": true}'
    assert result.prompt_tokens == 10 and result.completion_tokens == 5
    assert result.cost_usd == pytest.approx(estimate_cost_usd("gpt-4o-mini", 10, 5))
    assert completions.kwargs["response_format"]["type"] == "json_schema"

    failing = OpenAIClient(client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(fail=True))))
    with pytest.raises(LLMError, match="rate limited"):
        failing.complete([{"role": "user", "content": "x"}])


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_structured_output_retries_with_validation_feedback_then_succeeds():
    llm = FakeLLM(["not json", '{"supported": "maybe"}', '{"supported": true, "unsupported_claim_count": 0}'])
    outcome = complete_structured(llm, [{"role": "user", "content": "grade"}], Verdict, max_attempts=3)
    assert outcome.value == Verdict(supported=True, unsupported_claim_count=0)
    assert not outcome.used_fallback
    assert [a.error is None for a in outcome.attempts] == [False, False, True]
    # The retry prompt carries the validation error back to the model.
    assert "JSON schema" in llm.calls[1]["messages"][-1]["content"]
    assert llm.calls[2]["json_schema"]["title"] == "Verdict"
    assert outcome.prompt_tokens > 0 and outcome.cost_usd >= 0


def test_structured_output_falls_back_after_exhausting_attempts():
    llm = FakeLLM(["nope", "still nope"])
    outcome = complete_structured(
        llm,
        [{"role": "user", "content": "grade"}],
        Verdict,
        max_attempts=2,
        fallback=lambda: Verdict(supported=False, unsupported_claim_count=99),
    )
    assert outcome.used_fallback
    assert outcome.value.unsupported_claim_count == 99


def test_structured_output_does_not_retry_provider_errors():
    outcome = complete_structured(
        UnavailableLLM(),
        [{"role": "user", "content": "grade"}],
        Verdict,
        fallback=lambda: Verdict(supported=False, unsupported_claim_count=0),
    )
    assert outcome.used_fallback
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].error.startswith("provider_error")
    with pytest.raises(StructuredOutputFailure):
        complete_structured(UnavailableLLM(), [{"role": "user", "content": "grade"}], Verdict)

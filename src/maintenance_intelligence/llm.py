"""Provider-agnostic chat-completion adapter.

The rest of the codebase talks to :class:`LLMClient` only. Concrete providers:

* :class:`OpenAIClient` – OpenAI or Azure OpenAI (same SDK, different constructor).
* :class:`FakeLLM` – deterministic scripted responses for tests and offline evals.
* :class:`UnavailableLLM` – raised-on-use placeholder when no provider is configured,
  so every LLM-backed feature degrades to the extractive path instead of crashing.

Token usage and an estimated USD cost ride along on every :class:`LLMResult`, which is
what the tracing layer and the cost/latency benchmark read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMUnavailable(RuntimeError):
    """No LLM provider is configured for this deployment."""


class LLMError(RuntimeError):
    """The provider returned an error or an unusable response."""


# USD per 1M tokens (input, output), standard tier, from the OpenAI pricing page
# as of 2026-09-08. Used only for the per-call cost estimate; verify against
# https://developers.openai.com/api/docs/pricing before quoting numbers. Older
# model names are kept so historical runs still get a cost estimate.
DEFAULT_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.00, 50.00),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.3-codex": (1.75, 14.00),
    "chat-latest": (5.00, 30.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prices: dict[str, tuple[float, float]] | None = None,
) -> float:
    table = prices or DEFAULT_PRICES_PER_MTOK
    key = model
    if key not in table:
        # Dated snapshots and Azure deployments usually carry the base name as a
        # prefix; prefer the longest matching base name (gpt-5.6-luna over gpt-5).
        candidates = [name for name in table if model.startswith(name)]
        key = max(candidates, key=len) if candidates else ""
    if not key:
        return 0.0
    input_price, output_price = table[key]
    return round((prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000, 8)


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    cost_usd: float
    cached: bool = False
    raw: Any = field(default=None, repr=False, compare=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    provider: str
    default_model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 600,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult: ...


class UnavailableLLM:
    provider = "none"
    default_model = "none"

    def complete(self, messages, *, model=None, temperature=0.0, max_tokens=600, json_schema=None) -> LLMResult:
        raise LLMUnavailable(
            "No LLM provider configured. Set LLM_PROVIDER=openai and OPENAI_API_KEY "
            "(or LLM_PROVIDER=azure with AZURE_OPENAI_* variables)."
        )


class FakeLLM:
    """Scripted responses for tests. Each call pops the next response in order."""

    provider = "fake"
    default_model = "fake-model"

    def __init__(self, responses: list[str] | None = None, *, model: str = "fake-model") -> None:
        self.responses = list(responses or [])
        self.default_model = model
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, *, model=None, temperature=0.0, max_tokens=600, json_schema=None) -> LLMResult:
        self.calls.append(
            {
                "messages": messages,
                "model": model or self.default_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_schema": json_schema,
            }
        )
        if not self.responses:
            raise LLMError("FakeLLM has no scripted response left")
        text = self.responses.pop(0)
        prompt_tokens = sum(len(str(item.get("content", ""))) // 4 for item in messages)
        completion_tokens = max(1, len(text) // 4)
        return LLMResult(
            text=text,
            model=model or self.default_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=0.0,
            cost_usd=estimate_cost_usd("gpt-4o-mini", prompt_tokens, completion_tokens),
        )


class OpenAIClient:
    """OpenAI and Azure OpenAI through the official SDK.

    ``provider="azure"`` uses ``AzureOpenAI`` and treats ``default_model`` as the
    deployment name, which is what Azure expects in the ``model`` field.
    """

    def __init__(
        self,
        *,
        provider: str = "openai",
        api_key: str | None = None,
        default_model: str = "gpt-5.6-luna",
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-10-21",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        prices: dict[str, tuple[float, float]] | None = None,
        client: Any = None,
    ) -> None:
        self.provider = provider
        self.default_model = default_model
        self.prices = prices
        if client is not None:
            self._client = client
            return
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - dependency is declared, but be explicit
            raise LLMUnavailable("The 'openai' package is not installed") from exc
        if provider == "azure":
            if not (azure_endpoint and api_key):
                raise LLMUnavailable("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required")
            self._client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )
        else:
            if not api_key:
                raise LLMUnavailable("OPENAI_API_KEY is required")
            self._client = openai.OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries)

    def complete(self, messages, *, model=None, temperature=0.0, max_tokens=600, json_schema=None) -> LLMResult:
        chosen = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": chosen,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema.get("title", "response"), "schema": json_schema, "strict": False},
            }
        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # the SDK raises many subclasses; normalize for callers
            raise LLMError(f"{self.provider} completion failed: {exc}") from exc
        latency = time.perf_counter() - started
        choice = completion.choices[0] if completion.choices else None
        text = (choice.message.content or "") if choice and choice.message else ""
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return LLMResult(
            text=text,
            model=getattr(completion, "model", chosen) or chosen,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            cost_usd=estimate_cost_usd(chosen, prompt_tokens, completion_tokens, self.prices),
            raw=completion,
        )


def build_llm_client(settings: Any) -> LLMClient:
    """Construct the configured provider, or an :class:`UnavailableLLM` placeholder."""
    provider = (getattr(settings, "llm_provider", "none") or "none").lower()
    if provider == "none":
        return UnavailableLLM()
    if provider == "openai":
        return OpenAIClient(
            provider="openai",
            api_key=settings.openai_api_key,
            default_model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if provider == "azure":
        return OpenAIClient(
            provider="azure",
            api_key=settings.azure_openai_api_key,
            default_model=settings.azure_openai_deployment or settings.llm_model,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_api_version=settings.azure_openai_api_version,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    raise LLMUnavailable(f"Unknown LLM_PROVIDER '{provider}'")

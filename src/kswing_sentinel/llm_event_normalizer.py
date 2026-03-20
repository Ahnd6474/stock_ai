from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .schemas import LLMNormalizedEvent


NEUTRAL_EVENT = {
    "canonical_summary": "No reliable new event. Semantic branch neutralized.",
    "event_score": 0.0,
    "event_half_life": "3d",
    "regime_hint": "chop",
    "red_flag": False,
    "flow_vs_tech_resolution": "NO_TRADE",
    "source_quality_score": 0.0,
    "freshness_score": 0.0,
    "semantic_confidence": 0.0,
    "evidence_spans": [],
    "entity_tags": [],
    "reasons": ["DEGRADED_MODE"],
}

SYSTEM_PROMPT = """You normalize Korean equity event text into strict JSON.
Return JSON only. Do not add markdown.
Required keys:
- canonical_summary
- event_score
- event_half_life
- regime_hint
- red_flag
- flow_vs_tech_resolution
- source_quality_score
- freshness_score
- semantic_confidence
- evidence_spans
- entity_tags
- reasons
Constraints:
- event_score in [-1, 1]
- source_quality_score, freshness_score, semantic_confidence in [0, 1]
- event_half_life in {intraday, 3d, 1w, 2w, 1m}
- regime_hint in {trend, event, chop, risk_off}
- flow_vs_tech_resolution in {FLOW_DOMINANT_CONTINUATION, TECH_DOMINANT_EXHAUSTION, WAIT_FOR_PULLBACK, NO_TRADE}
- evidence_spans must be a list
- reasons must be short machine-readable tokens
"""


class LLMProviderError(RuntimeError):
    pass


class BaseLLMProvider:
    provider_name = "base"

    def generate_structured_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class StaticLLMProvider(BaseLLMProvider):
    provider_name = "static"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = dict(payload)

    def generate_structured_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return dict(self.payload)


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str = "LLM_API_KEY",
        timeout_s: float = 15.0,
        provider_name: str = "openai_compatible",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.provider_name = provider_name

    def generate_structured_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            import httpx
        except Exception as exc:  # pragma: no cover - optional dependency
            raise LLMProviderError("httpx is required for OpenAI-compatible providers") from exc

        token = self.api_key or os.getenv(self.api_key_env)
        if not token:
            raise LLMProviderError(f"missing API key env: {self.api_key_env}")

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMProviderError("unexpected provider response shape") from exc
        return _coerce_json_payload(content)


def _coerce_json_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        return json.loads(payload)
    raise LLMProviderError("provider response must be dict or JSON string")


def _build_user_prompt(payload: dict[str, Any]) -> str:
    symbol = payload.get("symbol", "")
    source_type = payload.get("source_type", "UNKNOWN")
    headline = payload.get("headline", "")
    body = payload.get("body", payload.get("text", ""))
    published_at = payload.get("published_at", "")
    return (
        f"symbol={symbol}\n"
        f"source_type={source_type}\n"
        f"published_at={published_at}\n"
        f"headline={headline}\n"
        f"body={body}\n"
    )


def _looks_structured(payload: dict[str, Any]) -> bool:
    required = {
        "canonical_summary",
        "event_score",
        "event_half_life",
        "regime_hint",
        "red_flag",
        "flow_vs_tech_resolution",
        "source_quality_score",
        "freshness_score",
        "semantic_confidence",
        "evidence_spans",
        "entity_tags",
        "reasons",
    }
    return required.issubset(payload.keys())


@dataclass
class LLMEventNormalizer:
    prompt_version: str = "v1"
    provider_name: str = "fallback"
    provider: BaseLLMProvider | None = None
    schema_violations: int = 0

    def __post_init__(self) -> None:
        if self.provider is not None and self.provider_name == "fallback":
            self.provider_name = self.provider.provider_name

    def normalize(self, payload: dict, retry_once: bool = True) -> LLMNormalizedEvent:
        try:
            return self._normalize_once(payload)
        except Exception:
            self.schema_violations += 1
            if retry_once:
                try:
                    return self._normalize_once(payload)
                except Exception:
                    self.schema_violations += 1
            fallback = dict(NEUTRAL_EVENT)
            fallback["provider_name"] = self.provider_name
            fallback["prompt_version"] = self.prompt_version
            return LLMNormalizedEvent(**fallback)

    def _normalize_once(self, payload: dict) -> LLMNormalizedEvent:
        structured = dict(payload) if _looks_structured(payload) else self._generate_structured_payload(payload)
        structured.setdefault("provider_name", self.provider_name)
        structured.setdefault("prompt_version", self.prompt_version)
        return LLMNormalizedEvent(**structured)

    def _generate_structured_payload(self, payload: dict) -> dict[str, Any]:
        if self.provider is None:
            raise LLMProviderError("no live provider configured")
        user_prompt = _build_user_prompt(payload)
        structured = self.provider.generate_structured_json(SYSTEM_PROMPT, user_prompt)
        if not isinstance(structured.get("evidence_spans", []), list):
            structured["evidence_spans"] = []
        if not isinstance(structured.get("entity_tags", []), list):
            structured["entity_tags"] = []
        if not isinstance(structured.get("reasons", []), list):
            structured["reasons"] = []
        return structured

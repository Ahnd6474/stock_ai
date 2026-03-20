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
- evidence_spans must be a list of {doc_id,start_char,end_char,text_snippet}
- reasons must be short machine-readable tokens
Use search evidence only as supporting context. Do not invent facts.
"""


class LLMProviderError(RuntimeError):
    pass


class SearchProviderError(RuntimeError):
    pass


class BaseSearchClient:
    provider_name = "base_search"

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError


class StaticSearchClient(BaseSearchClient):
    provider_name = "static_search"

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = [dict(x) for x in results]

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return self.results[:max_results]


class InternalContextSearchClient(BaseSearchClient):
    provider_name = "internal_context_only"

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = [dict(x) for x in (results or [])]

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return self.results[:max_results]


class PerplexitySearchClient(BaseSearchClient):
    provider_name = "perplexity_search"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "PERPLEXITY_API_KEY",
        base_url: str = "https://api.perplexity.ai",
        timeout_s: float = 15.0,
        search_domain_filter: list[str] | None = None,
        search_recency_filter: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.search_domain_filter = search_domain_filter or []
        self.search_recency_filter = search_recency_filter

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        try:
            import httpx
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SearchProviderError("httpx is required for Perplexity search") from exc

        token = self.api_key or os.getenv(self.api_key_env)
        if not token:
            raise SearchProviderError(f"missing API key env: {self.api_key_env}")

        response = httpx.post(
            f"{self.base_url}/search",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": max_results,
                "search_domain_filter": self.search_domain_filter or None,
                "search_recency_filter": self.search_recency_filter,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or payload.get("data") or []
        if not isinstance(results, list):
            raise SearchProviderError("unexpected search response shape")
        normalized: list[dict[str, Any]] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", item.get("text", "")),
                    "published_at": item.get("published_at", ""),
                }
            )
        return normalized


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
        default_headers: dict[str, str] | None = None,
        default_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.provider_name = provider_name
        self.default_headers = dict(default_headers or {})
        self.default_body = dict(default_body or {})

    def generate_structured_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            import httpx
        except Exception as exc:  # pragma: no cover - optional dependency
            raise LLMProviderError("httpx is required for OpenAI-compatible providers") from exc

        token = self.api_key or os.getenv(self.api_key_env)
        if not token:
            raise LLMProviderError(f"missing API key env: {self.api_key_env}")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **self.default_headers,
        }
        body = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **self.default_body,
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMProviderError("unexpected provider response shape") from exc
        return _coerce_json_payload(content)


class EnsembleLLMProvider(BaseLLMProvider):
    provider_name = "ensemble"

    def __init__(self, providers: list[BaseLLMProvider]) -> None:
        self.providers = providers

    def generate_structured_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        outputs: list[dict[str, Any]] = []
        for provider in self.providers:
            try:
                outputs.append(provider.generate_structured_json(system_prompt, user_prompt))
            except Exception:
                continue
        if not outputs:
            raise LLMProviderError("all providers failed")
        if len(outputs) == 1:
            return outputs[0]
        return _merge_structured_outputs(outputs)


def _coerce_json_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        return json.loads(payload)
    raise LLMProviderError("provider response must be dict or JSON string")


def _build_user_prompt(payload: dict[str, Any], search_results: list[dict[str, Any]] | None = None) -> str:
    symbol = payload.get("symbol", "")
    source_type = payload.get("source_type", "UNKNOWN")
    headline = payload.get("headline", "")
    body = payload.get("body", payload.get("text", ""))
    published_at = payload.get("published_at", "")
    search_blob = ""
    if search_results:
        lines = []
        for idx, item in enumerate(search_results, start=1):
            lines.append(
                f"[{idx}] title={item.get('title', '')}\n"
                f"url={item.get('url', '')}\n"
                f"published_at={item.get('published_at', '')}\n"
                f"snippet={item.get('snippet', '')}"
            )
        search_blob = "search_results:\n" + "\n".join(lines) + "\n"
    return (
        f"symbol={symbol}\n"
        f"source_type={source_type}\n"
        f"published_at={published_at}\n"
        f"headline={headline}\n"
        f"body={body}\n"
        f"{search_blob}"
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


def _average_float(outputs: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [float(o.get(key, default)) for o in outputs]
    return sum(values) / max(len(values), 1)


def _majority_vote(outputs: list[dict[str, Any]], key: str, default: str) -> str:
    counts: dict[str, int] = {}
    for out in outputs:
        value = str(out.get(key, default))
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0] if counts else default


def _merge_list(outputs: list[dict[str, Any]], key: str) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for out in outputs:
        value = out.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
    return merged


def _merge_structured_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    primary = dict(outputs[0])
    merged = {
        "canonical_summary": primary.get("canonical_summary", ""),
        "event_score": max(-1.0, min(1.0, _average_float(outputs, "event_score", 0.0))),
        "event_half_life": _majority_vote(outputs, "event_half_life", "3d"),
        "regime_hint": _majority_vote(outputs, "regime_hint", "chop"),
        "red_flag": any(bool(o.get("red_flag", False)) for o in outputs),
        "flow_vs_tech_resolution": _majority_vote(outputs, "flow_vs_tech_resolution", "NO_TRADE"),
        "source_quality_score": max(0.0, min(1.0, _average_float(outputs, "source_quality_score", 0.0))),
        "freshness_score": max(0.0, min(1.0, _average_float(outputs, "freshness_score", 0.0))),
        "semantic_confidence": max(0.0, min(1.0, _average_float(outputs, "semantic_confidence", 0.0))),
        "evidence_spans": _merge_list(outputs, "evidence_spans"),
        "entity_tags": _merge_list(outputs, "entity_tags"),
        "reasons": _merge_list(outputs, "reasons"),
    }
    return merged


def _default_search_query(payload: dict[str, Any]) -> str:
    symbol = str(payload.get("symbol", "")).strip()
    headline = str(payload.get("headline", "")).strip()
    body = str(payload.get("body", payload.get("text", ""))).strip()
    source_type = str(payload.get("source_type", "")).strip()
    query_parts = [x for x in [symbol, headline, source_type] if x]
    if not query_parts and body:
        query_parts.append(body[:160])
    return " ".join(query_parts)


def build_openrouter_reasoning_provider(
    *,
    model: str,
    app_name: str = "K-Swing Sentinel",
    referer: str = "https://local.kswing-sentinel",
    api_key_env: str = "OPENROUTER_API_KEY",
) -> OpenAICompatibleLLMProvider:
    return OpenAICompatibleLLMProvider(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key_env=api_key_env,
        provider_name=f"openrouter:{model}",
        default_headers={
            "HTTP-Referer": referer,
            "X-Title": app_name,
        },
    )


def build_planned_market_llm_provider() -> EnsembleLLMProvider:
    return EnsembleLLMProvider(
        [
            build_openrouter_reasoning_provider(model="x-ai/grok-4.1-fast"),
            build_openrouter_reasoning_provider(model="qwen/qwen3.5-35b-a3b"),
            build_openrouter_reasoning_provider(model="deepseek/deepseek-v3.2"),
        ]
    )


def build_grok_only_market_llm_provider() -> OpenAICompatibleLLMProvider:
    return build_openrouter_reasoning_provider(model="x-ai/grok-4.1-fast")


@dataclass
class LLMEventNormalizer:
    prompt_version: str = "v1"
    provider_name: str = "fallback"
    provider: BaseLLMProvider | None = None
    search_client: BaseSearchClient | None = None
    search_max_results: int = 5
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

        search_results: list[dict[str, Any]] = []
        internal_results = payload.get("internal_search_results") or payload.get("search_results") or []
        if isinstance(internal_results, list):
            search_results = [dict(x) for x in internal_results if isinstance(x, dict)]
        if self.search_client is not None:
            try:
                query = str(payload.get("search_query") or _default_search_query(payload)).strip()
                if query and not search_results:
                    search_results = self.search_client.search(query, max_results=self.search_max_results)
            except Exception:
                if not search_results:
                    search_results = []

        user_prompt = _build_user_prompt(payload, search_results=search_results)
        structured = self.provider.generate_structured_json(SYSTEM_PROMPT, user_prompt)
        if not isinstance(structured.get("evidence_spans", []), list):
            structured["evidence_spans"] = []
        if not isinstance(structured.get("entity_tags", []), list):
            structured["entity_tags"] = []
        if not isinstance(structured.get("reasons", []), list):
            structured["reasons"] = []
        return structured

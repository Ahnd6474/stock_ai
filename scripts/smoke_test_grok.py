from __future__ import annotations

import json
import os
import sys

from kswing_sentinel.llm_event_normalizer import LLMEventNormalizer, build_grok_only_market_llm_provider


def main() -> int:
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is missing", file=sys.stderr)
        return 2

    normalizer = LLMEventNormalizer(
        prompt_version="normalizer_prompt_v2",
        provider=build_grok_only_market_llm_provider(),
    )
    payload = {
        "symbol": "005930",
        "source_type": "NEWS",
        "headline": "삼성전자 HBM 수요 확대 전망",
        "body": "메모리 가격 개선과 고객사 수요 확대 기대가 반영됐다는 보도가 나왔다.",
        "internal_search_results": [
            {
                "title": "내부 요약",
                "url": "memory://1",
                "snippet": "기관 수급 개선과 메모리 업황 회복 기대",
            }
        ],
    }
    out = normalizer.normalize(payload)
    print(json.dumps(out.model_dump(), ensure_ascii=False, indent=2))
    if out.canonical_summary == "No reliable new event. Semantic branch neutralized.":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

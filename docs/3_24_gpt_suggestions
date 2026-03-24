최신 논문 기준으로 보면, 이 프로젝트의 문제는 **“아이디어가 너무 낡았다”**가 아니라 **“좋은 방향인데 최신 퀀트 AI가 중요하게 보는 핵심 연결부가 아직 비어 있다”**에 가깝습니다. README도 스스로 이 레포를 **production-oriented scaffold**로 설명하고, live inference에 필요한 `state_sequence`, 외부 payload, venue eligibility, last prices 등을 아직 호출자가 공급해야 하며, full portfolio engine과 브로커 통합도 아직 없다고 적고 있습니다. 즉, 지금 상태는 **모델/런타임 조각은 있음 → 최신 실전형 연구가 요구하는 end-to-end 정렬은 아직 약함**입니다. ([GitHub][1])

참고로 아래에서 인용하는 최신 자료들 중 상당수는 **2025~2026 arXiv preprint**입니다. 그래서 “확정된 정설”이라기보다, 최근 연구 흐름을 반영하는 근거로 보는 게 맞습니다. 그래도 방향성은 꽤 일관됩니다. ([arXiv][2])

## 1) 예측 목표와 실제 투자 목표가 아직 분리돼 있습니다

지금 README상 모델은 `return`, `drawdown`, `probability-up`, `uncertainty`, `flow persistence`, `regime`를 예측하는 구조이고, 백테스터도 “cost-aware”라고 되어 있습니다. 하지만 README가 동시에 **full portfolio-level execution, reconciliation, lifecycle coverage가 아직 없다**고 밝히고 있습니다. 즉, 지금은 여전히 **좋은 예측기를 만든 뒤 나중에 실행/포트폴리오를 붙이는 구조**에 더 가깝습니다. ([GitHub][1])

최근 논문들은 이 분리를 점점 줄이는 쪽입니다. 2026년 decision-focused portfolio 논문은 **predict-then-optimize보다, 포트폴리오 최적화 제약과 목적을 학습 안으로 넣는 구조가 risk-adjusted performance를 더 잘 낸다**고 보고합니다. 또 2025/2026의 finance-grounded optimization 논문은 **Sharpe, PnL, MDD, turnover**를 직접 반영한 학습 목표가 deployability를 높인다고 주장합니다. ([arXiv][3])

**수정 방안**

* 예측기 출력값을 바로 액션으로 넘기지 말고, **differentiable portfolio / position-sizing layer**를 추가하세요.
* 학습 목표를 MSE 중심이 아니라 **Sharpe-aware loss + drawdown penalty + turnover penalty + exposure limit penalty**로 바꾸는 게 맞습니다.
* 단일 종목/단일 이벤트 판단에서도 최종 loss를 **“예측 정확도”보다 “비용 차감 후 포지션 품질”**에 더 가깝게 옮겨야 합니다.
* uncertainty를 그냥 보조 출력으로 두지 말고 **position cap**과 연결하세요. 불확실성이 높으면 노출을 자동 축소하는 방식입니다.

## 2) 텍스트는 읽지만, 최신 논문 수준의 “정렬(alignment)”은 약합니다

이 레포는 raw text를 직접 벡터화하고, sentence-level RoBERTa 뒤에 hierarchical transformer를 붙이는 구조를 채택했다고 밝힙니다. 이 자체는 나쁘지 않습니다. 오히려 단순 감성점수보다 훨씬 낫습니다. 하지만 현재 설명만 보면 텍스트는 **좋은 임베딩을 만들기 위한 부가 모달리티**에 가깝고, **가격 시계열과 텍스트가 언제, 어떤 사건으로 연결되는지**를 명시적으로 학습하는 축은 약합니다. ([GitHub][1])

최근 멀티모달 금융 예측 논문들은 한 단계 더 갑니다. 2025년 interleaved text-time-series 논문은 **modality-specific experts + cross-modal alignment**를 강조하고, 2026년 VoT 논문은 **event-driven reasoning**과 **multi-level alignment**가 필요하다고 주장합니다. 핵심은 “텍스트도 같이 넣었다”가 아니라, **텍스트가 가격 변화를 설명하는 사건 축으로 정렬되어야 한다**는 겁니다. ([arXiv][4])

**수정 방안**

* `z_event / z_social / z_macro`를 그냥 concat하지 말고, **timestamp-aware alignment layer**를 넣으세요.
* 뉴스/공시/소셜 각각에 대해 **freshness decay**와 **event impact window**를 별도로 두세요. 예를 들면 공시는 1~5일, 소셜은 수시간~1일 식으로요.
* **modality-specific expert + gated fusion** 구조를 고려하는 게 좋습니다. 종목/국면별로 어떤 텍스트 모달이 더 유효한지 자동 선택하게 해야 합니다.
* 단순 sentence embedding 말고, **event extractor**를 추가해서 “실적, 규제, 수주, 리콜, 경영진, 거시정책”처럼 시장 반응 단위로 추상화하는 편이 더 최신 흐름에 가깝습니다.

## 3) 평가 설계가 아직 최신 논문들이 가장 경계하는 편향들을 다 막지는 못합니다

README는 no-lookahead validation과 cost-aware backtest를 강조합니다. 이건 좋은 출발입니다. 하지만 최신 논문들은 이 정도로는 부족하다고 봅니다. 2026년 “Evaluating LLMs in Finance…”는 **look-ahead, survivorship, narrative, objective, cost bias**를 금융 AI의 대표적 문제로 지적했고, 2025/2026의 LLM investing 평가 논문도 **survivorship bias, look-ahead bias, data-snooping bias**를 장기·광범위 universe 기준으로 막아야 한다고 말합니다. ([arXiv][2])

이 레포는 sample training datasets가 있고, live inference용 `state_sequence`도 아직 외부가 공급해야 한다고 적고 있습니다. 이 상태에서는 **point-in-time 정확성**과 **historically correct universe**를 사용했는지 레포만으로 확신하기 어렵습니다. 특히 한국 시장에서는 상장폐지/관리종목/거래정지/인적분할/지주사 재편 같은 이벤트를 잘못 다루면 백테스트가 예쁘게 왜곡됩니다. ([GitHub][1])

**수정 방안**

* 데이터 레이어에 **point-in-time universe builder**를 넣으세요. 상장폐지 종목과 당시 시점 기준 종목 universe를 반드시 포함해야 합니다.
* 텍스트는 수집 시각, 게시 시각, 공급 시각을 분리해 저장하세요. 뉴스의 실제 이용 가능 시점이 중요합니다.
* validation은 일반 walk-forward만 하지 말고 **purged / embargoed time splits**를 도입하세요.
* 여러 아이디어를 동시에 시험한다면 **multiple testing correction**과 **seed robustness report**를 남겨야 합니다.
* 보고서에는 반드시 **gross return / net return / breakeven cost / turnover / hit ratio / tail metrics**를 같이 내세요. 최신 대규모 benchmark도 평균수익 하나보다 이 조합을 봅니다. ([arXiv][5])

## 4) temporal model은 있지만, 더 강한 시계열 표현학습 기준에서는 아직 baseline 쪽입니다

README도 training pipeline을 **baseline scaffolds**라고 스스로 적고 있습니다. 또 predictor는 temporal attention model을 지원하지만, research/serving stack으로는 아직 아니라고 합니다. ([GitHub][1])

최근 연구에서는 두 갈래가 보입니다. 하나는 **financial TSFM / foundation model** 계열이고, 다른 하나는 **현실적인 대규모 benchmark에서 어떤 backbone이 진짜 강한지 비교**하는 흐름입니다. 2025년 TSFM 논문은 금융 시계열에 대한 대규모 pretraining/fine-tuning 가능성을 제시했고, 2026 대규모 benchmark는 **rich temporal representation을 학습하는 모델이 단순 generic deep models보다 낫다**고 보고합니다. 다만 “커진 모델이면 무조건 좋다”는 뜻은 아닙니다. 오히려 강한 baseline 비교가 중요해졌다는 뜻에 가깝습니다. ([arXiv][6])

**수정 방안**

* 지금 temporal transformer를 유지하되, 반드시 **강한 baseline 세트**를 붙이세요.
  예: linear/ridge, LightGBM, TCN, SSM/Mamba류, TFT류, simple RNN.
* 가능하면 한국 시장용 **self-supervised pretraining**을 하세요.
  목표는 next-step regression보다 **masked span, temporal contrastive, regime discrimination**이 낫습니다.
* 출력도 점예측 하나보다 **distributional forecast**로 바꾸는 게 좋습니다.
  quantile / expected shortfall / calibrated uncertainty가 필요합니다.
* multi-horizon을 진짜로 쓰려면 1일, 5일, 20일을 분리하고, horizon별 loss weight를 둬야 합니다.

## 5) 포트폴리오와 크로스섹션 구조가 약합니다

지금 README 문구만 보면 이 시스템은 여전히 **KRX/NXT swing workflow** 중심이고, portfolio-level lifecycle은 아직 비포함입니다. 이건 단순히 “아직 주문 기능이 없다”가 아니라, **종목 간 상대가치와 자본배분**이 아직 핵심 모델 안에 들어오지 않았다는 뜻이기도 합니다. ([GitHub][1])

최신 퀀트 논문들은 단일 시계열 예측보다 **cross-sectional dependency**, **multi-period allocation**, **path-dependent risk**를 더 많이 다룹니다. 특히 portfolio optimization 계열은 예측값 하나보다 **어떤 자산에 얼마를 배분할지**를 직접 다룹니다. ([arXiv][3])

**수정 방안**

* 종목별 독립 예측기 구조에서 벗어나 **cross-sectional ranker**를 추가하세요.
* 섹터/스타일/유동성/시총/테마 관계를 넣은 **graph 또는 relation-aware layer**가 유효할 수 있습니다.
* 최종 의사결정은 `buy/hold/sell`보다 **weight / target exposure / max loss budget** 형태로 바꾸는 편이 낫습니다.
* 한국 시장 특성상 단순 종목 pick보다 **현금비중 + 업종비중 + 종목비중** 3단 구조가 더 실전적입니다.

## 6) 거래비용을 “반영”하는 수준에서 “학습이 두려워하는 수준”으로 올려야 합니다

README는 cost-aware backtesting을 말하지만, 최신 흐름은 한 걸음 더 나갑니다. 비용은 사후 차감이 아니라 **모델이 처음부터 무서워해야 하는 제약**이어야 한다는 쪽입니다. 2025/2026 finance-grounded optimization은 turnover regularization을 명시적으로 넣어 deployability를 높이는 방향을 제안합니다. ([GitHub][1])

**수정 방안**

* loss에 **turnover penalty**를 직접 넣으세요.
* 주문 시그널에는 **hysteresis / no-trade band**를 넣어 잦은 뒤집기를 줄이세요.
* KRX/NXT 세션별로 다른 비용 모형을 두세요.
  장중, 종가부근, 유동성 얕은 시간대의 슬리피지가 다릅니다.
* 텍스트 이벤트 기반 매매는 체결 경쟁이 심하므로 **latency bucket별 성능분석**도 해야 합니다.

## 7) 운영 안전성은 좋은데, 최신 기준의 “재현가능한 연구 스택”은 아직 약합니다

이 레포의 장점은 readiness gate, conservative execution mapping, audit logging 같은 운영 안전장치가 이미 있다는 점입니다. 이건 많은 논문 코드보다 낫습니다. 다만 README도 인정하듯 production broker integration, deployment automation, shadow-trading runbook, post-trade lifecycle은 아직 없습니다. ([GitHub][1])

최신 논문들이 공통으로 강조하는 건 단순 SOTA 수치보다 **robustness to seeds, cost break-even, multiple regimes, out-of-sample significance**입니다. 2026 대규모 benchmark도 random seed robustness와 computational efficiency까지 본다고 명시합니다. ([arXiv][5])

**수정 방안**

* 실험마다 **5~10 seed 반복**을 기본으로 하세요.
* 성과표에는 반드시 **평균 ± 표준편차**를 넣으세요.
* market regime별 결과를 분리하세요.
  상승장, 약세장, 고변동성, 저변동성, 이벤트 집중기.
* 최소 1~3개월 **shadow trading** 로그를 구조적으로 남기고, backtest/live drift를 추적하세요.

---

## 우선순위로 정리하면

가장 먼저 고쳐야 할 건 이 순서입니다.

**1. 평가 편향 통제 강화**
point-in-time universe, delisting, timestamp hygiene, purged split. ([arXiv][2])

**2. 예측기에서 포트폴리오/포지션 최적화로 목적함수 이동**
Sharpe/MDD/turnover-aware 학습. ([arXiv][3])

**3. 텍스트-시계열 정렬 강화**
event alignment, modality-specific experts, freshness decay. ([arXiv][4])

**4. 강한 baseline과 foundation-style pretraining 비교**
지금 모델이 진짜 이기는지 검증. ([arXiv][6])

**5. 포트폴리오 엔진 완성**
현재 README가 인정하는 가장 큰 실전 공백. ([GitHub][1])

---

## 한 줄 결론

이 프로젝트의 가장 큰 문제는 **모델이 너무 단순해서가 아니라, 최신 퀀트 AI가 중시하는 “bias-aware evaluation + decision-focused training + aligned multimodal fusion + portfolio-level optimization”이 아직 끝까지 연결되지 않았다는 점**입니다. 반대로 말하면, 지금 구조는 폐기할 정도로 나쁜 게 아니라 **좋은 뼈대 위에 최신 논문식 마지막 계층을 올리면 훨씬 강해질 타입**입니다. ([GitHub][1])

원하면 다음엔 이걸 바로 **“네 레포 기준 수정 로드맵 12주짜리”**로 바꿔드리겠습니다.

[1]: https://github.com/Ahnd6474/stock_ai "GitHub - Ahnd6474/stock_ai · GitHub"
[2]: https://arxiv.org/html/2602.14233v1 "Evaluating LLMs in Finance Requires Explicit Bias Consideration"
[3]: https://arxiv.org/abs/2601.04062 "[2601.04062] Smart Predict--then--Optimize Paradigm for Portfolio Optimization in Real Markets"
[4]: https://arxiv.org/abs/2509.19628?utm_source=chatgpt.com "Multimodal Language Models with Modality-Specific Experts for Financial Forecasting from Interleaved Sequences of Text and Time Series"
[5]: https://arxiv.org/abs/2603.01820 "[2603.01820] Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance"
[6]: https://arxiv.org/html/2511.18578v1 "Re(Visiting) Time Series Foundation Models in Finance"

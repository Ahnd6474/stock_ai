from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.feature_store import FeatureStore
from kswing_sentinel.yahoo_finance import YahooFinanceMarketData, resolve_yahoo_symbol


class FakeSeries:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)

    def __iter__(self):
        return iter(self._values)


class FakeHistory:
    def __init__(self, columns):
        self._columns = {key: list(values) for key, values in columns.items()}
        self.empty = not any(self._columns.values())

    def __getitem__(self, item):
        return FakeSeries(self._columns[item])


class FakeTicker:
    def __init__(self, *, fast_info=None, info=None, news=None, intraday_history=None, daily_history=None):
        self.fast_info = fast_info or {}
        self.info = info or {}
        self.news = news or []
        self._intraday_history = intraday_history
        self._daily_history = daily_history

    def history(self, *, period, interval, auto_adjust, prepost):
        if interval == "1d":
            return self._daily_history
        return self._intraday_history


def test_resolve_yahoo_symbol_defaults_krx_suffix():
    assert resolve_yahoo_symbol("005930") == "005930.KS"
    assert resolve_yahoo_symbol("AAPL") == "AAPL"
    assert resolve_yahoo_symbol("035720", symbol_overrides={"035720": "035720.KQ"}) == "035720.KQ"


def test_ingest_session_bars_builds_predictor_features():
    intraday = FakeHistory(
        {
            "Close": [100.0 + idx for idx in range(30)],
        }
    )
    daily = FakeHistory(
        {
            "Close": [90.0 + idx for idx in range(90)],
            "High": [91.0 + idx for idx in range(90)],
            "Low": [89.0 + idx for idx in range(90)],
            "Volume": [1_000_000 + idx * 10_000 for idx in range(90)],
        }
    )
    requested_symbols: list[str] = []

    def ticker_factory(symbol: str):
        requested_symbols.append(symbol)
        return FakeTicker(
            fast_info={
                "lastPrice": 129.0,
                "previousClose": 128.0,
                "open": 127.0,
                "dayHigh": 130.0,
                "dayLow": 126.0,
                "volume": 1_300_000,
                "marketCap": 500_000_000_000,
                "currency": "KRW",
                "exchange": "KSC",
            },
            info={"trailingPE": 12.5, "shortName": "Samsung Electronics"},
            intraday_history=intraday,
            daily_history=daily,
        )

    store = FeatureStore()
    service = YahooFinanceMarketData(
        symbols=["005930"],
        feature_store=store,
        ticker_factory=ticker_factory,
    )
    as_of_time = datetime(2026, 3, 20, 9, 35, tzinfo=ZoneInfo("Asia/Seoul"))
    service.ingest_session_bars(as_of_time)

    features = store.get_latest("005930", as_of_time)
    assert requested_symbols[0] == "005930.KS"
    assert features["feature_source"] == "yahoo_finance"
    assert features["last_price"] == 129.0
    assert features["currency"] == "KRW"
    assert features["flow_strength"] == 0.0
    assert features["trend_120m"] > 0
    assert features["extension_60m"] > 0
    assert features["volume_ratio_20d"] > 0
    assert features["sma_20"] is not None
    assert features["sma_60"] is not None
    assert features["ema_12"] is not None
    assert features["ema_26"] is not None
    assert features["macd"] is not None
    assert features["macd_signal"] is not None
    assert features["macd_hist"] is not None
    assert features["rsi_14"] is not None
    assert features["bb_width_20"] is not None
    assert features["atr_14"] is not None


def test_ingest_quote_proxies_and_build_live_inputs_use_cached_quote():
    def ticker_factory(symbol: str):
        return FakeTicker(
            fast_info={
                "lastPrice": 101.5,
                "previousClose": 100.0,
                "open": 100.5,
                "dayHigh": 102.0,
                "dayLow": 99.8,
                "volume": 555_000,
                "currency": "USD",
                "exchange": "NMS",
            },
            info={"trailingPE": 20.0, "shortName": "Apple"},
        )

    service = YahooFinanceMarketData(symbols=["AAPL"], ticker_factory=ticker_factory)
    as_of_time = datetime(2026, 3, 20, 9, 35, tzinfo=ZoneInfo("Asia/Seoul"))
    service.ingest_quote_proxies(as_of_time)

    features_by_symbol, last_price_by_symbol = service.build_live_inputs(as_of_time)
    features = features_by_symbol["AAPL"]
    assert last_price_by_symbol["AAPL"] == 101.5
    assert features["last_price"] == 101.5
    assert features["previous_close"] == 100.0
    assert features["flow_strength"] == 0.0
    assert features["trend_120m"] == 0.0
    assert features["extension_60m"] == 0.0


def test_fetch_company_profile_and_news_normalize_payloads():
    news = [
        {
            "uuid": "news-1",
            "content": {
                "id": "story-1",
                "title": "Apple supplier checks improved",
                "summary": "Demand commentary improved.",
                "canonicalUrl": {"url": "https://example.com/apple"},
                "provider": {"displayName": "Example News"},
                "pubDate": "2026-03-20T09:00:00Z",
            },
        }
    ]

    def ticker_factory(symbol: str):
        return FakeTicker(
            info={
                "shortName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "website": "https://apple.com",
                "country": "United States",
                "currency": "USD",
                "marketCap": 1_000,
                "fullTimeEmployees": 10,
                "trailingPE": 30.0,
                "forwardPE": 25.0,
                "longBusinessSummary": "Makes devices.",
            },
            news=news,
        )

    service = YahooFinanceMarketData(symbols=["AAPL"], ticker_factory=ticker_factory)
    profile = service.fetch_company_profile("AAPL")
    items = service.fetch_news("AAPL")

    assert profile["short_name"] == "Apple Inc."
    assert profile["sector"] == "Technology"
    assert items[0]["title"] == "Apple supplier checks improved"
    assert items[0]["url"] == "https://example.com/apple"

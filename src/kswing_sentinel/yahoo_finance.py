from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from .feature_store import FeatureStore, NumericFeatureRow
from .session_rules import classify_session


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _coerce_int(value: object) -> int | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    try:
        return dict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}


def _read_field(source: object, *keys: str) -> object:
    mapping = _as_mapping(source)
    for key in keys:
        if key in mapping:
            return mapping[key]
    for key in keys:
        if hasattr(source, key):
            return getattr(source, key)
    return None


def _series_values(history: object, column: str) -> list[float]:
    if history is None or bool(getattr(history, "empty", False)):
        return []
    try:
        series = history[column]
    except Exception:
        return []
    if hasattr(series, "tolist"):
        raw_values = series.tolist()
    else:
        raw_values = list(series)
    values: list[float] = []
    for raw in raw_values:
        numeric = _coerce_float(raw)
        if numeric is not None:
            values.append(numeric)
    return values


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator in {None, 0.0}:
        return 0.0
    return numerator / denominator - 1.0


def _sma(values: Sequence[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return _mean(values[-window:])


def _ema_series(values: Sequence[float], window: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (window + 1.0)
    ema_values = [float(values[0])]
    for value in values[1:]:
        ema_values.append(alpha * float(value) + (1.0 - alpha) * ema_values[-1])
    return ema_values


def _ema(values: Sequence[float], window: int) -> float | None:
    series = _ema_series(values, window)
    return series[-1] if len(series) >= window else None


def _stddev(values: Sequence[float], window: int) -> float | None:
    if len(values) < window:
        return None
    sample = list(values[-window:])
    avg = _mean(sample)
    if avg is None:
        return None
    variance = sum((value - avg) ** 2 for value in sample) / len(sample)
    return math.sqrt(variance)


def _macd(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if len(values) < 26:
        return None, None, None
    ema_12_series = _ema_series(values, 12)
    ema_26_series = _ema_series(values, 26)
    macd_series = [fast - slow for fast, slow in zip(ema_12_series, ema_26_series)]
    signal_series = _ema_series(macd_series, 9)
    if len(signal_series) < 9:
        return None, None, None
    macd_value = macd_series[-1]
    signal_value = signal_series[-1]
    return macd_value, signal_value, macd_value - signal_value


def _rsi(values: Sequence[float], window: int = 14) -> float | None:
    if len(values) <= window:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for prev, cur in zip(values[:-1], values[1:]):
        delta = cur - prev
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = _mean(gains[-window:])
    avg_loss = _mean(losses[-window:])
    if avg_gain is None or avg_loss is None:
        return None
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger_band_width(values: Sequence[float], window: int = 20, num_std: float = 2.0) -> float | None:
    center = _sma(values, window)
    std = _stddev(values, window)
    if center in {None, 0.0} or std is None:
        return None
    upper = center + num_std * std
    lower = center - num_std * std
    return (upper - lower) / center


def _atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], window: int = 14) -> float | None:
    if len(highs) < window or len(lows) < window or len(closes) < window + 1:
        return None
    true_ranges: list[float] = []
    for idx in range(1, len(closes)):
        high = float(highs[idx])
        low = float(lows[idx])
        prev_close = float(closes[idx - 1])
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return _mean(true_ranges[-window:])


def resolve_yahoo_symbol(
    symbol: str,
    *,
    symbol_overrides: Mapping[str, str] | None = None,
    default_suffix: str = ".KS",
) -> str:
    clean = symbol.strip()
    if symbol_overrides and clean in symbol_overrides:
        return symbol_overrides[clean]
    if "." in clean or "=" in clean or "^" in clean:
        return clean
    if clean.isdigit() and len(clean) == 6:
        return f"{clean}{default_suffix}"
    return clean


@dataclass(frozen=True)
class YahooFinanceQuote:
    symbol: str
    yahoo_symbol: str
    fetched_at: datetime
    currency: str | None
    exchange: str | None
    short_name: str | None
    last_price: float | None
    previous_close: float | None
    open_price: float | None
    day_high: float | None
    day_low: float | None
    volume: int | None
    market_cap: int | None
    trailing_pe: float | None


class YahooFinanceUnavailableError(RuntimeError):
    pass


class YahooFinanceMarketData:
    def __init__(
        self,
        *,
        symbols: Sequence[str],
        feature_store: FeatureStore | None = None,
        ticker_factory: Callable[[str], object] | None = None,
        symbol_overrides: Mapping[str, str] | None = None,
        default_suffix: str = ".KS",
        intraday_period: str = "5d",
        intraday_interval: str = "5m",
        daily_period: str = "3mo",
    ) -> None:
        self.symbols = list(symbols)
        self.feature_store = feature_store or FeatureStore()
        self.symbol_overrides = dict(symbol_overrides or {})
        self.default_suffix = default_suffix
        self.intraday_period = intraday_period
        self.intraday_interval = intraday_interval
        self.daily_period = daily_period
        self._ticker_factory = ticker_factory or self._default_ticker_factory
        self._custom_ticker_factory = ticker_factory is not None
        self.quote_cache: dict[str, YahooFinanceQuote] = {}

    @staticmethod
    def _default_ticker_factory(yahoo_symbol: str) -> object:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise YahooFinanceUnavailableError(
                "yfinance is not installed. Install it with `pip install -e .[marketdata]`."
            ) from exc
        return yf.Ticker(yahoo_symbol)

    def resolve_symbol(self, symbol: str) -> str:
        return resolve_yahoo_symbol(
            symbol,
            symbol_overrides=self.symbol_overrides,
            default_suffix=self.default_suffix,
        )

    def fetch_quote(self, symbol: str) -> YahooFinanceQuote:
        yahoo_symbol = self.resolve_symbol(symbol)
        if not self._custom_ticker_factory:
            quote = self._fetch_quote_from_chart(symbol, yahoo_symbol)
            self.quote_cache[symbol] = quote
            return quote
        ticker = self._ticker_factory(yahoo_symbol)
        quote = self._fetch_quote_from_ticker(symbol, yahoo_symbol, ticker)
        self.quote_cache[symbol] = quote
        return quote

    def _fetch_quote_from_chart(self, symbol: str, yahoo_symbol: str) -> YahooFinanceQuote:
        chart = self._request_chart(yahoo_symbol, range_value="5d", interval="1d")
        meta = _as_mapping(chart.get("meta"))
        history = self._history_from_chart(chart)
        closes = _series_values(history, "Close")
        opens = _series_values(history, "Open")
        highs = _series_values(history, "High")
        lows = _series_values(history, "Low")
        volumes = _series_values(history, "Volume")
        return YahooFinanceQuote(
            symbol=symbol,
            yahoo_symbol=yahoo_symbol,
            fetched_at=datetime.now(timezone.utc),
            currency=meta.get("currency"),
            exchange=meta.get("exchangeName") or meta.get("fullExchangeName"),
            short_name=None,
            last_price=_coerce_float(meta.get("regularMarketPrice")) or (closes[-1] if closes else None),
            previous_close=_coerce_float(meta.get("previousClose")),
            open_price=_coerce_float(meta.get("regularMarketOpen")) or (opens[-1] if opens else None),
            day_high=_coerce_float(meta.get("regularMarketDayHigh")) or (highs[-1] if highs else None),
            day_low=_coerce_float(meta.get("regularMarketDayLow")) or (lows[-1] if lows else None),
            volume=_coerce_int(meta.get("regularMarketVolume")) or (int(volumes[-1]) if volumes else None),
            market_cap=None,
            trailing_pe=None,
        )

    def _fetch_quote_from_ticker(self, symbol: str, yahoo_symbol: str, ticker: object) -> YahooFinanceQuote:
        fast_info = _read_field(ticker, "fast_info") or {}
        info = _read_field(ticker, "info") or {}
        return YahooFinanceQuote(
            symbol=symbol,
            yahoo_symbol=yahoo_symbol,
            fetched_at=datetime.now(timezone.utc),
            currency=_read_field(fast_info, "currency") or _read_field(info, "currency"),
            exchange=_read_field(fast_info, "exchange") or _read_field(info, "exchange"),
            short_name=_read_field(info, "shortName", "longName"),
            last_price=_coerce_float(
                _read_field(fast_info, "lastPrice", "regularMarketPrice")
                or _read_field(info, "regularMarketPrice", "currentPrice")
            ),
            previous_close=_coerce_float(
                _read_field(fast_info, "previousClose", "regularMarketPreviousClose")
                or _read_field(info, "previousClose", "regularMarketPreviousClose")
            ),
            open_price=_coerce_float(
                _read_field(fast_info, "open", "regularMarketOpen") or _read_field(info, "open", "regularMarketOpen")
            ),
            day_high=_coerce_float(
                _read_field(fast_info, "dayHigh", "regularMarketDayHigh")
                or _read_field(info, "dayHigh", "regularMarketDayHigh")
            ),
            day_low=_coerce_float(
                _read_field(fast_info, "dayLow", "regularMarketDayLow")
                or _read_field(info, "dayLow", "regularMarketDayLow")
            ),
            volume=_coerce_int(
                _read_field(fast_info, "lastVolume", "volume", "regularMarketVolume")
                or _read_field(info, "volume", "regularMarketVolume")
            ),
            market_cap=_coerce_int(_read_field(fast_info, "marketCap") or _read_field(info, "marketCap")),
            trailing_pe=_coerce_float(_read_field(info, "trailingPE")),
        )

    def fetch_company_profile(self, symbol: str) -> dict[str, object]:
        yahoo_symbol = self.resolve_symbol(symbol)
        ticker = self._ticker_factory(yahoo_symbol)
        info = _as_mapping(_read_field(ticker, "info"))
        return {
            "symbol": symbol,
            "yahoo_symbol": yahoo_symbol,
            "short_name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "market_cap": _coerce_int(info.get("marketCap")),
            "full_time_employees": _coerce_int(info.get("fullTimeEmployees")),
            "trailing_pe": _coerce_float(info.get("trailingPE")),
            "forward_pe": _coerce_float(info.get("forwardPE")),
            "business_summary": info.get("longBusinessSummary"),
        }

    def fetch_news(self, symbol: str, *, limit: int = 5) -> list[dict[str, object]]:
        yahoo_symbol = self.resolve_symbol(symbol)
        ticker = self._ticker_factory(yahoo_symbol)
        raw_items = list(_read_field(ticker, "news") or [])
        items: list[dict[str, object]] = []
        for raw in raw_items[: max(0, limit)]:
            mapping = _as_mapping(raw)
            content = _as_mapping(mapping.get("content"))
            provider = _as_mapping(content.get("provider"))
            items.append(
                {
                    "symbol": symbol,
                    "yahoo_symbol": yahoo_symbol,
                    "uuid": mapping.get("uuid") or content.get("id"),
                    "title": content.get("title") or mapping.get("title"),
                    "summary": content.get("summary"),
                    "url": content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else None,
                    "publisher": provider.get("displayName") or mapping.get("publisher"),
                    "published_at": content.get("pubDate") or mapping.get("providerPublishTime"),
                }
            )
        return items

    def ingest_session_bars(self, as_of_time: datetime) -> None:
        for symbol in self.symbols:
            yahoo_symbol = self.resolve_symbol(symbol)
            if self._custom_ticker_factory:
                ticker = self._ticker_factory(yahoo_symbol)
                intraday = ticker.history(
                    period=self.intraday_period,
                    interval=self.intraday_interval,
                    auto_adjust=False,
                    prepost=True,
                )
                daily = ticker.history(
                    period=self.daily_period,
                    interval="1d",
                    auto_adjust=False,
                    prepost=False,
                )
                quote = self._fetch_quote_from_ticker(symbol, yahoo_symbol, ticker)
                self.quote_cache[symbol] = quote
            else:
                intraday = self._history_from_chart(
                    self._request_chart(yahoo_symbol, range_value=self.intraday_period, interval=self.intraday_interval)
                )
                daily = self._history_from_chart(self._request_chart(yahoo_symbol, range_value=self.daily_period, interval="1d"))
                quote = self._fetch_quote_from_chart(symbol, yahoo_symbol)
                self.quote_cache[symbol] = quote
            features = self._build_features(symbol, yahoo_symbol, intraday, daily, quote)
            self.feature_store.put(
                NumericFeatureRow(
                    symbol=symbol,
                    timestamp=as_of_time,
                    features=features,
                    session_type=classify_session(as_of_time),
                )
            )

    def ingest_quote_proxies(self, as_of_time: datetime) -> None:
        for symbol in self.symbols:
            quote = self.fetch_quote(symbol)
            features = dict(self.feature_store.get_latest(symbol, as_of_time))
            features.update(
                {
                    "yahoo_symbol": quote.yahoo_symbol,
                    "last_price": quote.last_price,
                    "previous_close": quote.previous_close,
                    "open_price": quote.open_price,
                    "day_high": quote.day_high,
                    "day_low": quote.day_low,
                    "volume": quote.volume,
                    "market_cap": quote.market_cap,
                    "currency": quote.currency,
                    "exchange": quote.exchange,
                    "trailing_pe": quote.trailing_pe,
                    "quote_fetched_at": quote.fetched_at.isoformat(),
                }
            )
            features.setdefault("flow_strength", 0.0)
            features.setdefault("trend_120m", 0.0)
            features.setdefault("extension_60m", 0.0)
            features["session_liquidity_ok"] = bool(quote.volume and quote.volume > 0)
            self.feature_store.put(
                NumericFeatureRow(
                    symbol=symbol,
                    timestamp=as_of_time,
                    features=features,
                    session_type=classify_session(as_of_time),
                )
            )

    def build_live_inputs(self, as_of_time: datetime) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
        session_type = classify_session(as_of_time)
        features_by_symbol: dict[str, dict[str, object]] = {}
        last_price_by_symbol: dict[str, float] = {}
        for symbol in self.symbols:
            features = self.feature_store.build_online_features(symbol, as_of_time, session_type)
            quote = self.quote_cache.get(symbol)
            if quote is None or features.get("last_price") is None:
                quote = self.fetch_quote(symbol)
                features.setdefault("yahoo_symbol", quote.yahoo_symbol)
                features["last_price"] = quote.last_price
                features.setdefault("previous_close", quote.previous_close)
                features.setdefault("currency", quote.currency)
            features.setdefault("flow_strength", 0.0)
            features.setdefault("trend_120m", 0.0)
            features.setdefault("extension_60m", 0.0)
            last_price = _coerce_float(features.get("last_price"))
            if last_price is None and quote is not None:
                last_price = quote.previous_close
            if last_price is None:
                raise ValueError(f"no Yahoo Finance price available for {symbol}")
            features_by_symbol[symbol] = features
            last_price_by_symbol[symbol] = last_price
        return features_by_symbol, last_price_by_symbol

    def _build_features(
        self,
        symbol: str,
        yahoo_symbol: str,
        intraday_history: object,
        daily_history: object,
        quote: YahooFinanceQuote,
    ) -> dict[str, object]:
        intraday_closes = _series_values(intraday_history, "Close")
        daily_closes = _series_values(daily_history, "Close")
        daily_highs = _series_values(daily_history, "High")
        daily_lows = _series_values(daily_history, "Low")
        daily_volumes = _series_values(daily_history, "Volume")

        last_intraday = intraday_closes[-1] if intraday_closes else None
        last_daily = daily_closes[-1] if daily_closes else None
        last_price = quote.last_price or last_intraday or last_daily
        previous_close = quote.previous_close
        if previous_close is None and len(daily_closes) >= 2:
            previous_close = daily_closes[-2]
        trend_anchor = intraday_closes[-25] if len(intraday_closes) >= 25 else None
        extension_anchor = _mean(intraday_closes[-12:]) if len(intraday_closes) >= 12 else None
        volume_mean = _mean(daily_volumes[-20:]) if len(daily_volumes) >= 20 else _mean(daily_volumes)
        latest_volume = daily_volumes[-1] if daily_volumes else _coerce_float(quote.volume)
        sma_20 = _sma(daily_closes, 20)
        sma_60 = _sma(daily_closes, 60)
        ema_12 = _ema(daily_closes, 12)
        ema_26 = _ema(daily_closes, 26)
        macd_value, macd_signal, macd_hist = _macd(daily_closes)
        rsi_14 = _rsi(daily_closes, 14)
        bb_width_20 = _bollinger_band_width(daily_closes, 20)
        atr_14 = _atr(daily_highs, daily_lows, daily_closes, 14)

        stale_flags = {
            "intraday_history_short": len(intraday_closes) < 25,
            "daily_history_short": len(daily_closes) < 21,
        }
        missing_flags = {
            "price_missing": last_price is None,
            "volume_missing": latest_volume is None,
        }

        return {
            "symbol": symbol,
            "yahoo_symbol": yahoo_symbol,
            "last_price": last_price,
            "previous_close": previous_close,
            "open_price": quote.open_price,
            "day_high": quote.day_high,
            "day_low": quote.day_low,
            "volume": quote.volume if quote.volume is not None else _coerce_int(latest_volume),
            "market_cap": quote.market_cap,
            "currency": quote.currency,
            "exchange": quote.exchange,
            "trailing_pe": quote.trailing_pe,
            "flow_strength": 0.0,
            "trend_120m": _ratio(last_price, trend_anchor),
            "extension_60m": _ratio(last_price, extension_anchor),
            "gap_from_prev_close": _ratio(last_price, previous_close),
            "return_20d": _ratio(last_price, daily_closes[-21] if len(daily_closes) >= 21 else None),
            "volume_ratio_20d": 0.0 if latest_volume is None or volume_mean in {None, 0.0} else latest_volume / volume_mean,
            "sma_20": sma_20,
            "sma_60": sma_60,
            "price_vs_sma20": _ratio(last_price, sma_20),
            "price_vs_sma60": _ratio(last_price, sma_60),
            "ema_12": ema_12,
            "ema_26": ema_26,
            "macd": macd_value,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "rsi_14": rsi_14,
            "bb_width_20": bb_width_20,
            "atr_14": atr_14,
            "session_liquidity_ok": bool((quote.volume or 0) > 0 or (latest_volume or 0) > 0),
            "feature_source": "yahoo_finance",
            "quote_fetched_at": quote.fetched_at.isoformat(),
            "missing_flags": missing_flags,
            "stale_flags": stale_flags,
        }

    @staticmethod
    def _history_from_chart(chart_payload: Mapping[str, object]) -> dict[str, list[object]]:
        indicators = _as_mapping(chart_payload.get("indicators"))
        quote_list = indicators.get("quote") or []
        first_quote = quote_list[0] if isinstance(quote_list, list) and quote_list else {}
        quote = _as_mapping(first_quote)
        return {
            "Open": list(quote.get("open") or []),
            "High": list(quote.get("high") or []),
            "Low": list(quote.get("low") or []),
            "Close": list(quote.get("close") or []),
            "Volume": list(quote.get("volume") or []),
        }

    @staticmethod
    def _request_chart(yahoo_symbol: str, *, range_value: str, interval: str) -> Mapping[str, object]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise YahooFinanceUnavailableError(
                "requests is not installed. Install market data extras with `pip install -e .[marketdata]`."
            ) from exc

        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
            params={"range": range_value, "interval": interval, "includePrePost": "true"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        chart = _as_mapping(payload.get("chart"))
        error = chart.get("error")
        if error:
            raise ValueError(f"Yahoo chart API error for {yahoo_symbol}: {error}")
        result = chart.get("result") or []
        if not isinstance(result, list) or not result:
            raise ValueError(f"Yahoo chart API returned no result for {yahoo_symbol}")
        return _as_mapping(result[0])

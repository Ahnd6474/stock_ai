from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .fdr_dataset import _require_pandas, add_technical_features
from .yahoo_finance import YahooFinanceMarketData
from .yahoo_finance import resolve_yahoo_symbol

LABEL_COLUMNS = ["er_5b", "er_20b", "dd_20b", "p_up_20b"]


def _require_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("yfinance is required. Install market data extras first.") from exc
    return yf


def _ema(series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def _safe_div(numerator, denominator):
    return numerator / denominator.replace(0.0, float("nan"))


def _numeric(series):
    pd = _require_pandas()
    return pd.to_numeric(series, errors="coerce")


def _rolling_linear_stats(series, window: int):
    pd = _require_pandas()
    import numpy as np

    values = series.astype(float).to_numpy()
    slopes = np.full(len(values), np.nan)
    r2s = np.full(len(values), np.nan)
    x = np.arange(window, dtype=float)

    for idx in range(window - 1, len(values)):
        y = values[idx - window + 1 : idx + 1]
        if np.isnan(y).any():
            continue
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(((y - fitted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        slopes[idx] = slope
        r2s[idx] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def _consecutive_counts(condition_series, positive: bool = True):
    counts: list[int] = []
    current = 0
    for value in condition_series.fillna(False).astype(bool).tolist():
        matched = value if positive else not value
        if matched:
            current += 1
        else:
            current = 0
        counts.append(current)
    return counts


def add_intraday_labels(frame, horizon_bars_5: int = 5, horizon_bars_20: int = 20):
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    close = df["Close"].astype(float)
    future_close_5 = close.shift(-horizon_bars_5)
    future_close_20 = close.shift(-horizon_bars_20)
    future_low_20 = close.shift(-1).rolling(horizon_bars_20).min()
    df["er_5b"] = future_close_5 / close - 1.0
    df["er_20b"] = future_close_20 / close - 1.0
    df["dd_20b"] = (close - future_low_20) / close
    df["p_up_20b"] = (df["er_20b"] > 0).astype(float)
    return df


def add_intraday_market_features(frame):
    pd = _require_pandas()
    if frame.empty:
        return frame.copy()

    df = frame.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    volume = df["Volume"].astype(float)
    prev_close = close.shift(1)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = _numeric(_safe_div(close - low_14, high_14 - low_14) * 100.0)
    df["stoch_k_14"] = stoch_k
    df["stoch_d_3"] = _numeric(stoch_k.rolling(3).mean())

    df["roc_10"] = close.pct_change(10)
    df["momentum_10"] = close - close.shift(10)
    df["return_3b"] = close.pct_change(3)
    df["return_10b"] = close.pct_change(10)
    df["intraday_range"] = _safe_div(high - low, close)
    df["open_to_close_return"] = _safe_div(close - open_, open_)
    df["rolling_volatility_20"] = df["return_1d"].rolling(20).std(ddof=0)

    direction = close.diff().fillna(0.0)
    signed_volume = direction.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)) * volume
    df["obv"] = signed_volume.cumsum()
    df["obv_change_10"] = df["obv"].diff(10)

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0.0)
    negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0.0)
    positive_mf_14 = positive_flow.rolling(14).sum()
    negative_mf_14 = negative_flow.rolling(14).sum()
    money_ratio = _numeric(_safe_div(positive_mf_14, negative_mf_14))
    df["mfi_14"] = _numeric(100.0 - (100.0 / (1.0 + money_ratio)))
    df.loc[negative_mf_14 == 0.0, "mfi_14"] = 100.0

    cumulative_volume = volume.cumsum()
    df["vwap_session"] = _numeric(_safe_div((typical_price * volume).cumsum(), cumulative_volume))
    df["price_vs_vwap"] = _numeric(_safe_div(close, df["vwap_session"]) - 1.0)

    df["ema_5"] = _ema(close, 5)
    df["ema_10"] = _ema(close, 10)
    df["ema_20"] = _ema(close, 20)
    df["ema_40"] = _ema(close, 40)
    df["sma_5"] = close.rolling(5).mean()
    df["sma_10"] = close.rolling(10).mean()
    df["price_vs_ema5"] = _numeric(_safe_div(close, df["ema_5"]) - 1.0)
    df["price_vs_ema10"] = _numeric(_safe_div(close, df["ema_10"]) - 1.0)
    df["price_vs_ema20"] = _numeric(_safe_div(close, df["ema_20"]) - 1.0)
    df["price_vs_ema40"] = _numeric(_safe_div(close, df["ema_40"]) - 1.0)
    df["ema5_vs_ema20"] = _numeric(_safe_div(df["ema_5"], df["ema_20"]) - 1.0)
    df["ema20_slope"] = df["ema_20"].pct_change(3)
    df["ema40_slope"] = df["ema_40"].pct_change(3)
    df["ma_bull_stack"] = (
        (df["ema_5"] > df["ema_10"]) & (df["ema_10"] > df["ema_20"]) & (df["ema_20"] > df["ema_40"])
    ).astype(float)
    df["ma_bear_stack"] = (
        (df["ema_5"] < df["ema_10"]) & (df["ema_10"] < df["ema_20"]) & (df["ema_20"] < df["ema_40"])
    ).astype(float)

    slope_8, r2_8 = _rolling_linear_stats(close, 8)
    slope_20, r2_20 = _rolling_linear_stats(close, 20)
    df["lr_slope_8"] = slope_8
    df["lr_r2_8"] = r2_8
    df["lr_slope_20"] = slope_20
    df["lr_r2_20"] = r2_20
    df["up_bar_ratio_8"] = (close.diff() > 0).rolling(8).mean()
    df["up_bar_ratio_20"] = (close.diff() > 0).rolling(20).mean()

    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    atr_14 = tr.rolling(14).mean()
    plus_di = _numeric(100.0 * _safe_div(plus_dm.rolling(14).sum(), atr_14))
    minus_di = _numeric(100.0 * _safe_div(minus_dm.rolling(14).sum(), atr_14))
    dx = _numeric(100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di))
    df["adx_14"] = _numeric(dx.rolling(14).mean())

    tp = (high + low + close) / 3.0
    tp_sma_20 = tp.rolling(20).mean()
    mean_dev = (tp - tp_sma_20).abs().rolling(20).mean()
    df["cci_20"] = _numeric(_safe_div(tp - tp_sma_20, 0.015 * mean_dev))
    highest_14 = high.rolling(14).max()
    lowest_14 = low.rolling(14).min()
    df["williams_r_14"] = _numeric(-100.0 * _safe_div(highest_14 - close, highest_14 - lowest_14))
    df["rsi_6"] = _numeric(df["rsi_6"])

    df["atr_14_over_price"] = _numeric(_safe_div(df["atr_14"], close))
    df["avg_range_20"] = (high - low).rolling(20).mean()
    df["range_vs_avg_20"] = _numeric(_safe_div(high - low, df["avg_range_20"]))
    df["gap_size"] = _numeric(_safe_div(open_ - prev_close, prev_close))
    df["downside_vol_20"] = df["return_1d"].where(df["return_1d"] < 0, 0.0).rolling(20).std(ddof=0)

    candle_range = (high - low).replace(0.0, float("nan"))
    body = (close - open_).abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    df["body_ratio"] = _numeric(body / candle_range)
    df["upper_wick_ratio"] = _numeric(upper_wick / candle_range)
    df["lower_wick_ratio"] = _numeric(lower_wick / candle_range)
    df["bullish_candle"] = (close > open_).astype(float)
    df["bearish_candle"] = (close < open_).astype(float)
    df["consecutive_bull"] = _consecutive_counts(close > open_, positive=True)
    df["consecutive_bear"] = _consecutive_counts(close > open_, positive=False)
    avg_body_20 = body.rolling(20).mean()
    df["large_bull_candle"] = ((close > open_) & (body > avg_body_20 * 1.5)).astype(float)
    df["large_bear_candle"] = ((close < open_) & (body > avg_body_20 * 1.5)).astype(float)
    df["close_location_value"] = _numeric(_safe_div(close - low, high - low))

    recent_high_20 = high.rolling(20).max()
    recent_low_20 = low.rolling(20).min()
    recent_high_60 = high.rolling(60).max()
    recent_low_60 = low.rolling(60).min()
    df["pos_in_20bar_range"] = _numeric(_safe_div(close - recent_low_20, recent_high_20 - recent_low_20))
    df["pos_in_60bar_range"] = _numeric(_safe_div(close - recent_low_60, recent_high_60 - recent_low_60))
    df["recent_20_high_dist"] = _numeric(_safe_div(close, recent_high_20) - 1.0)
    df["recent_20_low_dist"] = _numeric(_safe_div(close, recent_low_20) - 1.0)
    df["recent_60_range_percentile"] = df["pos_in_60bar_range"]

    return df


def add_daily_bar_features(frame):
    pd = _require_pandas()
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    volume = df["Volume"].astype(float)
    prev_close = close.shift(1)

    typical_price = (high + low + close) / 3.0
    df["current_volume"] = volume
    df["current_turnover"] = close * volume
    df["cum_turnover_day"] = df["current_turnover"]
    df["vwap_session"] = typical_price
    df["intraday_return_from_open"] = _numeric(_safe_div(close - open_, open_))
    df["return_vs_prev_close"] = _numeric(_safe_div(close, prev_close) - 1.0)
    df["distance_from_day_high"] = _numeric(_safe_div(close, high) - 1.0)
    df["distance_from_day_low"] = _numeric(_safe_div(close, low) - 1.0)
    df["day_range_percentile"] = _numeric(_safe_div(close - low, high - low))
    df["drawdown_from_day_high"] = _numeric(_safe_div(high - close, high))
    df["gap_size_from_prev_close"] = _numeric(_safe_div(open_ - prev_close, prev_close))
    df["bar_index_in_day"] = 1.0
    df["is_morning_session"] = 0.0
    df["is_lunch_session"] = 0.0
    df["is_afternoon_session"] = 0.0
    df["is_opening_bar"] = 0.0
    df["is_near_close"] = 0.0
    df["opening_range_breakout_up"] = 0.0
    df["opening_range_breakout_down"] = 0.0
    df["break_prev_day_high"] = (close > high.shift(1)).astype(float)
    df["break_prev_day_low"] = (close < low.shift(1)).astype(float)
    df["relative_volume_by_slot"] = 1.0
    df["volume_ratio_4"] = _numeric(_safe_div(volume, volume.rolling(4).mean()))
    df["volume_ratio_8"] = _numeric(_safe_div(volume, volume.rolling(8).mean()))
    df["volume_ratio_20"] = _numeric(_safe_div(volume, volume.rolling(20).mean()))
    df["return_2b"] = close.pct_change(2)
    df["return_4b"] = close.pct_change(4)
    df["return_8b"] = close.pct_change(8)
    df["return_16b"] = close.pct_change(16)
    df["price_position_recent_8"] = _numeric(
        _safe_div(close - low.rolling(8).min(), high.rolling(8).max() - low.rolling(8).min())
    )
    return df


def add_session_features(frame):
    pd = _require_pandas()
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    dt_index = pd.to_datetime(df.index)
    grouped = df.groupby("date", sort=False)

    df["bar_index_in_day"] = grouped.cumcount() + 1
    max_bar_index = df.groupby("date", sort=False)["bar_index_in_day"].transform("max")
    df["is_morning_session"] = ((df["hour"] < 12)).astype(float)
    df["is_lunch_session"] = ((df["hour"] == 12)).astype(float)
    df["is_afternoon_session"] = ((df["hour"] >= 13)).astype(float)
    df["is_opening_bar"] = (df["bar_index_in_day"] <= 2).astype(float)
    df["is_near_close"] = (df["bar_index_in_day"] >= max_bar_index - 1).astype(float)
    df["weekday"] = dt_index.weekday
    df["is_monday"] = (df["weekday"] == 0).astype(float)
    df["is_friday"] = (df["weekday"] == 4).astype(float)

    df["intraday_return_from_open"] = grouped["Close"].transform(lambda s: s / s.iloc[0] - 1.0)
    prev_day_close = grouped["Close"].last().shift(1)
    prev_day_high = grouped["High"].max().shift(1)
    prev_day_low = grouped["Low"].min().shift(1)
    df["prev_day_close"] = df["date"].map(prev_day_close.to_dict())
    df["prev_day_high"] = df["date"].map(prev_day_high.to_dict())
    df["prev_day_low"] = df["date"].map(prev_day_low.to_dict())
    df["return_vs_prev_close"] = _safe_div(df["Close"], df["prev_day_close"]) - 1.0

    day_high = grouped["High"].cummax()
    day_low = grouped["Low"].cummin()
    day_open = grouped["Open"].transform("first")
    day_turnover = (df["Close"].astype(float) * df["Volume"].astype(float))
    df["current_volume"] = df["Volume"].astype(float)
    df["current_turnover"] = day_turnover
    df["cum_turnover_day"] = df.groupby("date", sort=False)["current_turnover"].cumsum()
    df["distance_from_day_high"] = _safe_div(df["Close"], day_high) - 1.0
    df["distance_from_day_low"] = _safe_div(df["Close"], day_low) - 1.0
    df["day_range_percentile"] = _safe_div(df["Close"] - day_low, day_high - day_low)
    df["drawdown_from_day_high"] = _safe_div(day_high - df["Close"], day_high)
    df["gap_size_from_prev_close"] = _safe_div(day_open - df["prev_day_close"], df["prev_day_close"])

    first_four = grouped.head(4).groupby("date")
    opening_range_high = first_four["High"].max()
    opening_range_low = first_four["Low"].min()
    df["opening_range_high"] = df["date"].map(opening_range_high.to_dict())
    df["opening_range_low"] = df["date"].map(opening_range_low.to_dict())
    df["opening_range_breakout_up"] = (df["Close"] > df["opening_range_high"]).astype(float)
    df["opening_range_breakout_down"] = (df["Close"] < df["opening_range_low"]).astype(float)
    df["break_prev_day_high"] = (df["Close"] > df["prev_day_high"]).astype(float)
    df["break_prev_day_low"] = (df["Close"] < df["prev_day_low"]).astype(float)

    slot_avg_volume = df.groupby(["hour", "minute"])["Volume"].transform("mean")
    df["relative_volume_by_slot"] = _safe_div(df["Volume"].astype(float), slot_avg_volume.astype(float))
    df["volume_ratio_4"] = _safe_div(df["Volume"].astype(float), df["Volume"].astype(float).rolling(4).mean())
    df["volume_ratio_8"] = _safe_div(df["Volume"].astype(float), df["Volume"].astype(float).rolling(8).mean())
    df["volume_ratio_20"] = _safe_div(df["Volume"].astype(float), df["Volume"].astype(float).rolling(20).mean())
    df["return_2b"] = df["Close"].astype(float).pct_change(2)
    df["return_4b"] = df["Close"].astype(float).pct_change(4)
    df["return_8b"] = df["Close"].astype(float).pct_change(8)
    df["return_16b"] = df["Close"].astype(float).pct_change(16)
    df["price_position_recent_8"] = _safe_div(
        df["Close"] - df["Low"].rolling(8).min(),
        df["High"].rolling(8).max() - df["Low"].rolling(8).min(),
    )
    return df


def add_market_relative_features(frame, benchmark_frame):
    pd = _require_pandas()
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    if benchmark_frame is None or benchmark_frame.empty:
        for col in [
            "kospi_return_1b",
            "kosdaq_return_1b",
            "excess_return_kospi_1b",
            "excess_return_kosdaq_1b",
            "excess_return_kospi_4b",
            "excess_return_kosdaq_4b",
            "kospi_defense_score",
            "kosdaq_defense_score",
        ]:
            df[col] = 0.0
        return df

    merged = df.join(benchmark_frame, how="left")
    merged["kospi_return_1b"] = merged["kospi_close"].pct_change(fill_method=None)
    merged["kosdaq_return_1b"] = merged["kosdaq_close"].pct_change(fill_method=None)
    merged["kospi_return_4b"] = merged["kospi_close"].pct_change(4, fill_method=None)
    merged["kosdaq_return_4b"] = merged["kosdaq_close"].pct_change(4, fill_method=None)
    merged["excess_return_kospi_1b"] = merged["return_1d"] - merged["kospi_return_1b"]
    merged["excess_return_kosdaq_1b"] = merged["return_1d"] - merged["kosdaq_return_1b"]
    merged["excess_return_kospi_4b"] = merged["return_4b"] - merged["kospi_return_4b"]
    merged["excess_return_kosdaq_4b"] = merged["return_4b"] - merged["kosdaq_return_4b"]
    merged["kospi_defense_score"] = merged["return_1d"] - merged["kospi_return_1b"].where(merged["kospi_return_1b"] < 0, 0.0)
    merged["kosdaq_defense_score"] = merged["return_1d"] - merged["kosdaq_return_1b"].where(merged["kosdaq_return_1b"] < 0, 0.0)
    return merged


def add_sector_relative_features(frames_by_symbol: Mapping[str, object], metadata_by_symbol: Mapping[str, Mapping[str, object]]):
    pd = _require_pandas()
    out: dict[str, object] = {}
    sector_to_symbols: dict[str, list[str]] = {}
    for symbol, metadata in metadata_by_symbol.items():
        sector = metadata.get("sector")
        if isinstance(sector, str) and sector.strip():
            sector_to_symbols.setdefault(sector, []).append(symbol)

    for symbol, frame in frames_by_symbol.items():
        df = frame.copy()
        metadata = metadata_by_symbol.get(symbol, {})
        sector = metadata.get("sector")
        peer_symbols = [peer for peer in sector_to_symbols.get(sector, []) if peer != symbol] if sector else []
        if not peer_symbols:
            for col in [
                "sector_return_1b",
                "sector_return_4b",
                "sector_excess_return_1b",
                "sector_excess_return_4b",
                "sector_defense_score",
            ]:
                df[col] = 0.0
            out[symbol] = df
            continue
        peer_returns_1b = []
        peer_returns_4b = []
        for peer in peer_symbols:
            peer_frame = frames_by_symbol.get(peer)
            if peer_frame is None or getattr(peer_frame, "empty", True):
                continue
            peer_returns_1b.append(peer_frame["return_1d"].rename(f"{peer}_return_1b"))
            peer_returns_4b.append(peer_frame["return_4b"].rename(f"{peer}_return_4b"))
        if not peer_returns_1b:
            for col in [
                "sector_return_1b",
                "sector_return_4b",
                "sector_excess_return_1b",
                "sector_excess_return_4b",
                "sector_defense_score",
            ]:
                df[col] = 0.0
            out[symbol] = df
            continue
        sector_return_1b = pd.concat(peer_returns_1b, axis=1).mean(axis=1)
        sector_return_4b = pd.concat(peer_returns_4b, axis=1).mean(axis=1) if peer_returns_4b else pd.Series(index=df.index, dtype=float)
        df = df.join(sector_return_1b.rename("sector_return_1b"), how="left")
        df = df.join(sector_return_4b.rename("sector_return_4b"), how="left")
        df["sector_excess_return_1b"] = df["return_1d"] - df["sector_return_1b"]
        df["sector_excess_return_4b"] = df["return_4b"] - df["sector_return_4b"]
        df["sector_defense_score"] = df["return_1d"] - df["sector_return_1b"].where(df["sector_return_1b"] < 0, 0.0)
        out[symbol] = df
    return out


def add_daily_context_features(frame, daily_frame):
    pd = _require_pandas()
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    if daily_frame is None or daily_frame.empty:
        for col in [
            "daily_ema20",
            "daily_ema60",
            "daily_ema20_slope",
            "daily_ema60_slope",
            "daily_rsi_14",
            "daily_return_5d",
            "daily_return_20d",
            "daily_box_breakout_20d",
            "daily_near_20d_high",
            "daily_volume_surge",
            "daily_volatility_20d",
        ]:
            df[col] = pd.NA
        return df

    ctx = daily_frame.copy()
    close = ctx["Close"].astype(float)
    volume = ctx["Volume"].astype(float)
    ctx["daily_ema20"] = _ema(close, 20)
    ctx["daily_ema60"] = _ema(close, 60)
    ctx["daily_ema20_slope"] = ctx["daily_ema20"].pct_change(3)
    ctx["daily_ema60_slope"] = ctx["daily_ema60"].pct_change(3)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rs = _safe_div(gain.rolling(14).mean(), loss.rolling(14).mean())
    ctx["daily_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    ctx.loc[(avg_loss == 0.0) & (avg_gain > 0.0), "daily_rsi_14"] = 100.0
    ctx.loc[(avg_loss == 0.0) & (avg_gain == 0.0), "daily_rsi_14"] = 50.0
    ctx["daily_return_5d"] = close.pct_change(5)
    ctx["daily_return_20d"] = close.pct_change(20)
    ctx["daily_box_breakout_20d"] = (close > ctx["High"].rolling(20).max().shift(1)).astype(float)
    ctx["daily_near_20d_high"] = _safe_div(close, ctx["High"].rolling(20).max()) - 1.0
    ctx["daily_volume_surge"] = _safe_div(volume, volume.rolling(20).mean())
    ctx["daily_volatility_20d"] = close.pct_change().rolling(20).std(ddof=0)
    ctx = ctx.reset_index()
    date_col = ctx.columns[0]
    ctx["date"] = pd.to_datetime(ctx[date_col]).dt.date
    merged = df.merge(
        ctx[
            [
                "date",
                "daily_ema20",
                "daily_ema60",
                "daily_ema20_slope",
                "daily_ema60_slope",
                "daily_rsi_14",
                "daily_return_5d",
                "daily_return_20d",
                "daily_box_breakout_20d",
                "daily_near_20d_high",
                "daily_volume_surge",
                "daily_volatility_20d",
            ]
        ],
        on="date",
        how="left",
    )
    merged.index = df.index
    return merged


def _normalize_intraday_frame(frame, symbol: str, yahoo_symbol: str):
    pd = _require_pandas()
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    if getattr(df.index, "tz", None) is not None:
        try:
            df.index = df.index.tz_convert("Asia/Seoul")
        except Exception:
            pass
    df.index.name = "DateTime"
    dt_index = pd.to_datetime(df.index)
    df["symbol"] = symbol
    df["yahoo_symbol"] = yahoo_symbol
    df["date"] = dt_index.date
    df["hour"] = dt_index.hour
    df["minute"] = dt_index.minute
    df["bar_index_in_day"] = 0
    return df


def _normalize_timeframe_frame(frame, symbol: str, yahoo_symbol: str, timeframe: str):
    pd = _require_pandas()
    df = _normalize_intraday_frame(frame, symbol, yahoo_symbol)
    if timeframe == "1d":
        dt_index = pd.to_datetime(df.index)
        df["hour"] = 0
        df["minute"] = 0
        df["date"] = dt_index.date
    return df


def _fetch_benchmark_intraday(period: str, interval: str):
    yf = _require_yfinance()
    pd = _require_pandas()
    benchmark_symbols = {"kospi_close": "^KS11", "kosdaq_close": "^KQ11"}
    merged = None
    for column, ticker_symbol in benchmark_symbols.items():
        try:
            frame = yf.Ticker(ticker_symbol).history(period=period, interval=interval, auto_adjust=False, prepost=False)
        except Exception:
            frame = None
        if frame is None or frame.empty:
            continue
        sample = frame[["Close"]].rename(columns={"Close": column})
        if getattr(sample.index, "tz", None) is not None:
            try:
                sample.index = sample.index.tz_convert("Asia/Seoul")
            except Exception:
                pass
        merged = sample if merged is None else merged.join(sample, how="outer")
    return merged if merged is not None else pd.DataFrame()


def _fetch_daily_context(yahoo_symbol: str, daily_period: str = "6mo"):
    yf = _require_yfinance()
    try:
        frame = yf.Ticker(yahoo_symbol).history(period=daily_period, interval="1d", auto_adjust=False, prepost=False)
    except Exception:
        return None
    return frame if frame is not None and not frame.empty else None


def _fetch_company_metadata(symbol: str, yahoo_symbol: str, ticker: object | None = None) -> dict[str, object]:
    profile = YahooFinanceMarketData(symbols=[symbol]).fetch_company_profile(symbol) if ticker is None else None
    if profile is None:
        profile = {"symbol": symbol, "yahoo_symbol": yahoo_symbol}
        try:
            info = getattr(ticker, "info", {}) or {}
        except Exception:
            info = {}
        profile.update(
            {
                "short_name": info.get("shortName") or info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "currency": info.get("currency"),
                "market_cap": info.get("marketCap"),
                "exchange": info.get("exchange"),
            }
        )
    return {
        "company_name": profile.get("short_name"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "currency": profile.get("currency"),
        "market_cap": profile.get("market_cap"),
        "exchange": profile.get("exchange"),
    }


def frame_to_intraday_training_rows(frame, symbol: str) -> list[dict]:
    pd = _require_pandas()
    rows: list[dict] = []
    if frame.empty:
        return rows
    normalized = frame.reset_index()
    base_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "vwap_session",
        "current_turnover",
        "cum_turnover_day",
    ]
    feature_columns = [
        "return_1d",
        "return_2b",
        "return_3b",
        "return_4b",
        "return_8b",
        "return_10b",
        "return_16b",
        "intraday_return_from_open",
        "return_vs_prev_close",
        "distance_from_day_high",
        "distance_from_day_low",
        "price_vs_vwap",
        "price_position_recent_8",
        "pos_in_20bar_range",
        "pos_in_60bar_range",
        "recent_20_high_dist",
        "recent_20_low_dist",
        "recent_60_range_percentile",
        "day_range_percentile",
        "volume_ratio_4",
        "volume_ratio_8",
        "volume_ratio_20",
        "volume_ratio_20d",
        "relative_volume_by_slot",
        "rsi_6",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_40",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_60",
        "price_vs_sma20",
        "price_vs_sma60",
        "price_vs_ema5",
        "price_vs_ema10",
        "price_vs_ema20",
        "price_vs_ema40",
        "ema5_vs_ema20",
        "ema20_slope",
        "ema40_slope",
        "ma_bull_stack",
        "ma_bear_stack",
        "lr_slope_8",
        "lr_r2_8",
        "lr_slope_20",
        "lr_r2_20",
        "adx_14",
        "up_bar_ratio_8",
        "up_bar_ratio_20",
        "consecutive_bull",
        "consecutive_bear",
        "rsi_14",
        "stoch_k_14",
        "stoch_d_3",
        "macd",
        "macd_signal",
        "macd_hist",
        "roc_10",
        "cci_20",
        "williams_r_14",
        "mfi_14",
        "atr_14",
        "atr_14_over_price",
        "rolling_volatility_20",
        "avg_range_20",
        "range_vs_avg_20",
        "gap_size",
        "gap_size_from_prev_close",
        "drawdown_from_day_high",
        "downside_vol_20",
        "intraday_range",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "bullish_candle",
        "bearish_candle",
        "large_bull_candle",
        "large_bear_candle",
        "close_location_value",
        "excess_return_kospi_1b",
        "excess_return_kosdaq_1b",
        "excess_return_kospi_4b",
        "excess_return_kosdaq_4b",
        "kospi_defense_score",
        "kosdaq_defense_score",
        "sector_return_1b",
        "sector_return_4b",
        "sector_excess_return_1b",
        "sector_excess_return_4b",
        "sector_defense_score",
        "bar_index_in_day",
        "is_morning_session",
        "is_lunch_session",
        "is_afternoon_session",
        "is_opening_bar",
        "is_near_close",
        "is_monday",
        "is_friday",
        "opening_range_breakout_up",
        "opening_range_breakout_down",
        "daily_ema20",
        "daily_ema60",
        "daily_ema20_slope",
        "daily_ema60_slope",
        "daily_rsi_14",
        "daily_return_5d",
        "daily_return_20d",
        "daily_box_breakout_20d",
        "daily_near_20d_high",
        "daily_volume_surge",
        "daily_volatility_20d",
    ]
    label_columns = LABEL_COLUMNS

    for _, row in normalized.iterrows():
        if any(pd.isna(row.get(col)) for col in base_columns + feature_columns + label_columns):
            continue
        rows.append(
            {
                "datetime": row["DateTime"].isoformat() if hasattr(row["DateTime"], "isoformat") else str(row["DateTime"]),
                "date": row["date"],
                "symbol": str(symbol),
                "yahoo_symbol": row["yahoo_symbol"],
                "hour": int(row["hour"]),
                "minute": int(row["minute"]),
                "open_price": float(row["Open"]),
                "high_price": float(row["High"]),
                "low_price": float(row["Low"]),
                "current_price": float(row["Close"]),
                "current_volume": float(row["Volume"]),
                "current_turnover": float(row["current_turnover"]),
                "cum_turnover_day": float(row["cum_turnover_day"]),
                "vwap_session": float(row["vwap_session"]),
                **{col: float(row[col]) for col in feature_columns},
                **{col: float(row[col]) for col in label_columns},
            }
        )
    return rows


def _frame_to_training_rows(frame, symbol: str, timeframe: str, metadata: Mapping[str, object] | None = None) -> list[dict]:
    rows = frame_to_intraday_training_rows(frame, symbol)
    metadata = metadata or {}
    for row in rows:
        row["timeframe"] = timeframe
        row["company_name"] = metadata.get("company_name")
        row["sector"] = metadata.get("sector")
        row["industry"] = metadata.get("industry")
        row["currency"] = metadata.get("currency")
        row["market_cap"] = metadata.get("market_cap")
        row["exchange"] = metadata.get("exchange")
    return rows


def collect_intraday_training_rows(
    *,
    symbols: Sequence[str],
    period: str = "30d",
    interval: str = "15m",
    symbol_overrides: dict[str, str] | None = None,
) -> list[dict]:
    yf = _require_yfinance()
    all_rows: list[dict] = []
    benchmark_frame = _fetch_benchmark_intraday(period, interval)

    metadata_by_symbol: dict[str, dict[str, object]] = {}
    frames_by_symbol: dict[str, object] = {}
    for symbol in symbols:
        yahoo_symbol = resolve_yahoo_symbol(symbol, symbol_overrides=symbol_overrides)
        ticker = yf.Ticker(yahoo_symbol)
        frame = ticker.history(period=period, interval=interval, auto_adjust=False, prepost=False)
        if frame is None or frame.empty:
            continue
        metadata_by_symbol[symbol] = _fetch_company_metadata(symbol, yahoo_symbol, ticker)
        normalized = _normalize_timeframe_frame(frame, symbol, yahoo_symbol, interval)
        enriched = add_technical_features(normalized)
        enriched = add_intraday_market_features(enriched)
        enriched = add_session_features(enriched)
        enriched = add_market_relative_features(enriched, benchmark_frame)
        enriched = add_daily_context_features(enriched, _fetch_daily_context(yahoo_symbol))
        enriched = add_intraday_labels(enriched)
        frames_by_symbol[symbol] = enriched
    frames_by_symbol = add_sector_relative_features(frames_by_symbol, metadata_by_symbol)
    for symbol, enriched in frames_by_symbol.items():
        all_rows.extend(_frame_to_training_rows(enriched, symbol, interval, metadata_by_symbol.get(symbol)))
    return all_rows


def collect_multi_timeframe_training_rows(
    *,
    symbols: Sequence[str],
    symbol_overrides: dict[str, str] | None = None,
    timeframe_specs: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, list[dict]]:
    specs = timeframe_specs or {
        "1d": {"period": "1y", "interval": "1d"},
        "60m": {"period": "120d", "interval": "60m"},
        "15m": {"period": "30d", "interval": "15m"},
    }
    results: dict[str, list[dict]] = {}
    for timeframe, spec in specs.items():
        interval = spec.get("interval", timeframe)
        period = spec.get("period", "30d")
        results[timeframe] = collect_intraday_training_rows(
            symbols=symbols,
            period=period,
            interval=interval,
            symbol_overrides=symbol_overrides,
        )
    return results


@dataclass(frozen=True)
class IntradayCollectionSummary:
    symbols: list[str]
    row_count: int
    output_path: str


def save_intraday_training_rows(rows: Sequence[dict], output_path: str | Path) -> IntradayCollectionSummary:
    pd = _require_pandas()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    symbols = sorted({str(row["symbol"]) for row in rows}) if rows else []
    return IntradayCollectionSummary(symbols=symbols, row_count=len(rows), output_path=str(path))


def split_feature_and_label_rows(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    feature_rows: list[dict] = []
    label_rows: list[dict] = []
    key_columns = ["datetime", "date", "symbol", "yahoo_symbol", "timeframe"]
    for row in rows:
        feature_row = {key: row.get(key) for key in row.keys() if key not in LABEL_COLUMNS}
        label_row = {key: row.get(key) for key in key_columns if key in row}
        label_row.update({col: row.get(col) for col in LABEL_COLUMNS})
        feature_rows.append(feature_row)
        label_rows.append(label_row)
    return feature_rows, label_rows


def save_multi_timeframe_training_rows(
    rows_by_timeframe: Mapping[str, Sequence[dict]],
    output_dir: str | Path,
    prefix: str = "training",
) -> dict[str, IntradayCollectionSummary]:
    output_root = Path(output_dir)
    summaries: dict[str, IntradayCollectionSummary] = {}
    for timeframe, rows in rows_by_timeframe.items():
        suffix = timeframe.replace("m", "min").replace("d", "day")
        feature_rows, label_rows = split_feature_and_label_rows(rows)
        feature_path = output_root / f"{prefix}_{suffix}_features.csv"
        label_path = output_root / f"{prefix}_{suffix}_labels.csv"
        save_intraday_training_rows(label_rows, label_path)
        summaries[timeframe] = save_intraday_training_rows(feature_rows, feature_path)
    return summaries

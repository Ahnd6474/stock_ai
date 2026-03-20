from __future__ import annotations

import pandas as pd

from kswing_sentinel.intraday_dataset import (
    LABEL_COLUMNS,
    add_daily_context_features,
    add_intraday_labels,
    add_intraday_market_features,
    add_market_relative_features,
    add_sector_relative_features,
    add_session_features,
    _frame_to_training_rows,
    frame_to_intraday_training_rows,
    split_feature_and_label_rows,
)
from kswing_sentinel.fdr_dataset import add_technical_features


def _sample_intraday_frame():
    idx = pd.date_range("2025-01-01 09:00:00", periods=120, freq="15min", tz="Asia/Seoul")
    frame = pd.DataFrame(
        {
            "Open": [100 + i * 0.1 for i in range(120)],
            "High": [100.3 + i * 0.1 for i in range(120)],
            "Low": [99.7 + i * 0.1 for i in range(120)],
            "Close": [100 + i * 0.1 for i in range(120)],
            "Volume": [1000 + i * 10 for i in range(120)],
            "symbol": ["005930"] * 120,
            "yahoo_symbol": ["005930.KS"] * 120,
            "date": pd.to_datetime(idx).date,
            "hour": pd.to_datetime(idx).hour,
            "minute": pd.to_datetime(idx).minute,
        },
        index=idx,
    )
    frame.index.name = "DateTime"
    return frame


def _sample_benchmark_frame(index):
    return pd.DataFrame(
        {
            "kospi_close": [3000 + i for i in range(len(index))],
            "kosdaq_close": [1000 + i * 0.5 for i in range(len(index))],
        },
        index=index,
    )


def _sample_daily_frame():
    idx = pd.date_range("2024-10-01", periods=120, freq="D")
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(120)],
            "High": [101 + i for i in range(120)],
            "Low": [99 + i for i in range(120)],
            "Close": [100 + i for i in range(120)],
            "Volume": [1_000_000 + 10_000 * i for i in range(120)],
        },
        index=idx,
    )


def test_intraday_dataset_builds_feature_and_label_rows():
    source = _sample_intraday_frame()
    frame = add_technical_features(source)
    frame = add_intraday_market_features(frame)
    frame = add_session_features(frame)
    frame = add_market_relative_features(frame, _sample_benchmark_frame(source.index))
    frame = add_sector_relative_features(
        {"005930": frame},
        {"005930": {"sector": "Technology"}},
    )["005930"]
    frame = add_daily_context_features(frame, _sample_daily_frame())
    frame = add_intraday_labels(frame)
    rows = frame_to_intraday_training_rows(frame, "005930")
    assert rows
    row = rows[0]
    assert row["symbol"] == "005930"
    assert row["yahoo_symbol"] == "005930.KS"
    assert row["current_price"] > 0
    assert row["open_price"] > 0
    assert row["current_volume"] > 0
    assert "datetime" in row
    assert "macd_hist" in row
    assert "price_vs_vwap" in row
    assert "stoch_k_14" in row
    assert "mfi_14" in row
    assert "rsi_6" in row
    assert "excess_return_kospi_1b" in row
    assert "sector_excess_return_1b" in row
    assert "daily_ema20" in row
    assert "bar_index_in_day" in row
    assert "er_20b" in row


def test_sector_relative_features_use_peer_average():
    source_a = _sample_intraday_frame()
    source_b = _sample_intraday_frame()
    source_b["Close"] = source_b["Close"] * 1.01
    frame_a = add_session_features(add_intraday_market_features(add_technical_features(source_a)))
    frame_b = add_session_features(add_intraday_market_features(add_technical_features(source_b)))
    enriched = add_sector_relative_features(
        {"005930": frame_a, "000660": frame_b},
        {
            "005930": {"sector": "Technology"},
            "000660": {"sector": "Technology"},
        },
    )
    row = enriched["005930"].iloc[-1]
    assert pd.notna(row["sector_return_1b"])
    assert pd.notna(row["sector_excess_return_1b"])


def test_frame_rows_include_metadata_and_timeframe():
    source = _sample_intraday_frame()
    frame = add_technical_features(source)
    frame = add_intraday_market_features(frame)
    frame = add_session_features(frame)
    frame = add_market_relative_features(frame, _sample_benchmark_frame(source.index))
    frame = add_sector_relative_features(
        {"005930": frame, "000660": frame.copy()},
        {
            "005930": {"sector": "Technology"},
            "000660": {"sector": "Technology"},
        },
    )["005930"]
    frame = add_daily_context_features(frame, _sample_daily_frame())
    frame = add_intraday_labels(frame)
    rows = _frame_to_training_rows(
        frame,
        "005930",
        "15m",
        {
            "company_name": "Samsung Electronics",
            "sector": "Technology",
            "industry": "Semiconductors",
            "currency": "KRW",
            "market_cap": 100,
            "exchange": "KOE",
        },
    )
    assert rows
    row = rows[0]
    assert row["timeframe"] == "15m"
    assert row["company_name"] == "Samsung Electronics"
    assert row["sector"] == "Technology"
    assert row["market_cap"] == 100


def test_split_feature_and_label_rows_removes_future_targets_from_features():
    rows = [
        {
            "datetime": "2025-01-01T09:00:00+09:00",
            "date": "2025-01-01",
            "symbol": "005930",
            "yahoo_symbol": "005930.KS",
            "timeframe": "15m",
            "current_price": 100.0,
            "er_5b": 0.1,
            "er_20b": 0.2,
            "dd_20b": 0.05,
            "p_up_20b": 1.0,
        }
    ]
    feature_rows, label_rows = split_feature_and_label_rows(rows)
    assert feature_rows
    assert label_rows
    for column in LABEL_COLUMNS:
        assert column not in feature_rows[0]
        assert column in label_rows[0]


def test_session_features_use_only_seen_intraday_extremes():
    source = _sample_intraday_frame().iloc[:8].copy()
    source.iloc[0, source.columns.get_loc("High")] = 101.0
    source.iloc[1, source.columns.get_loc("High")] = 102.0
    source.iloc[2, source.columns.get_loc("High")] = 103.0
    source.iloc[7, source.columns.get_loc("High")] = 120.0
    source.iloc[0, source.columns.get_loc("Low")] = 99.0
    source.iloc[1, source.columns.get_loc("Low")] = 98.0
    source.iloc[2, source.columns.get_loc("Low")] = 97.0
    source.iloc[7, source.columns.get_loc("Low")] = 80.0
    frame = add_session_features(source)
    third_bar = frame.iloc[2]
    assert abs(third_bar["distance_from_day_high"] - ((third_bar["Close"] / 103.0) - 1.0)) < 1e-9
    assert abs(third_bar["distance_from_day_low"] - ((third_bar["Close"] / 97.0) - 1.0)) < 1e-9
    assert third_bar["opening_range_breakout_up"] == 0.0

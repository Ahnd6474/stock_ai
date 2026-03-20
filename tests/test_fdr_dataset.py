from datetime import date

import pandas as pd

from kswing_sentinel.fdr_dataset import add_technical_features, add_training_labels, frame_to_training_rows


def _sample_frame():
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(90)],
            "High": [101 + i for i in range(90)],
            "Low": [99 + i for i in range(90)],
            "Close": [100 + i for i in range(90)],
            "Volume": [1_000_000 + 1_000 * i for i in range(90)],
        },
        index=dates,
    )


def test_add_technical_features_creates_first_batch_indicators():
    frame = add_technical_features(_sample_frame())
    last = frame.iloc[-1]
    assert pd.notna(last["sma_20"])
    assert pd.notna(last["sma_60"])
    assert pd.notna(last["price_vs_sma20"])
    assert pd.notna(last["ema_12"])
    assert pd.notna(last["macd"])
    assert pd.notna(last["rsi_14"])
    assert pd.notna(last["bb_width_20"])
    assert pd.notna(last["atr_14"])


def test_frame_to_training_rows_filters_incomplete_rows_and_emits_labels():
    frame = add_training_labels(add_technical_features(_sample_frame()))
    rows = frame_to_training_rows(frame, "005930")
    assert rows
    first = rows[0]
    assert first["symbol"] == "005930"
    assert isinstance(first["date"], date)
    assert "macd_hist" in first
    assert "er_20d" in first
    assert "p_up_20d" in first

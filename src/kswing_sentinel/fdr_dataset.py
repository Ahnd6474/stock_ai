from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("pandas is required. Install market data extras first.") from exc
    return pd


def _require_fdr():
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("FinanceDataReader is required. Install with `pip install finance-datareader`.") from exc
    return fdr


def _ema(series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def add_technical_features(frame):
    pd = _require_pandas()
    if frame.empty:
        return frame.copy()

    df = frame.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["return_1d"] = close.pct_change()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_60"] = close.rolling(60).mean()
    df["price_vs_sma20"] = close / df["sma_20"] - 1.0
    df["price_vs_sma60"] = close / df["sma_60"] - 1.0

    df["ema_12"] = _ema(close, 12)
    df["ema_26"] = _ema(close, 26)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain_14 = gain.rolling(14).mean()
    avg_loss_14 = loss.rolling(14).mean()
    rs = avg_gain_14 / avg_loss_14.replace(0.0, pd.NA)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    df.loc[avg_loss_14 == 0.0, "rsi_14"] = 100.0
    avg_gain_6 = gain.rolling(6).mean()
    avg_loss_6 = loss.rolling(6).mean()
    rs_6 = avg_gain_6 / avg_loss_6.replace(0.0, pd.NA)
    df["rsi_6"] = 100.0 - (100.0 / (1.0 + rs_6))
    df.loc[avg_loss_6 == 0.0, "rsi_6"] = 100.0

    rolling_std = close.rolling(20).std(ddof=0)
    bb_mid = df["sma_20"]
    df["bb_width_20"] = ((bb_mid + 2.0 * rolling_std) - (bb_mid - 2.0 * rolling_std)) / bb_mid

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["volume_ratio_20d"] = volume / volume.rolling(20).mean()
    return df


def add_training_labels(frame):
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    close = df["Close"].astype(float)
    future_close_5d = close.shift(-5)
    future_close_20d = close.shift(-20)
    future_low_20d = close.shift(-1).rolling(20).min()

    df["er_5d"] = future_close_5d / close - 1.0
    df["er_20d"] = future_close_20d / close - 1.0
    df["dd_20d"] = (close - future_low_20d) / close
    df["p_up_20d"] = (df["er_20d"] > 0).astype(float)
    return df


def frame_to_training_rows(frame, symbol: str) -> list[dict]:
    pd = _require_pandas()
    rows: list[dict] = []
    if frame.empty:
        return rows
    normalized = frame.copy()
    normalized = normalized.reset_index()
    if "Date" not in normalized.columns:
        normalized = normalized.rename(columns={normalized.columns[0]: "Date"})

    feature_columns = [
        "return_1d",
        "sma_20",
        "sma_60",
        "price_vs_sma20",
        "price_vs_sma60",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "bb_width_20",
        "atr_14",
        "volume_ratio_20d",
    ]
    label_columns = ["er_5d", "er_20d", "dd_20d", "p_up_20d"]

    for _, row in normalized.iterrows():
        if any(pd.isna(row.get(col)) for col in feature_columns + label_columns):
            continue
        rows.append(
            {
                "date": row["Date"].date() if hasattr(row["Date"], "date") else row["Date"],
                "symbol": symbol,
                **{col: float(row[col]) for col in feature_columns},
                **{col: float(row[col]) for col in label_columns},
            }
        )
    return rows


@dataclass(frozen=True)
class CollectionSummary:
    symbols: list[str]
    row_count: int
    output_path: str


def collect_training_rows(
    *,
    symbols: Sequence[str],
    start: str,
    end: str | None = None,
) -> list[dict]:
    fdr = _require_fdr()
    all_rows: list[dict] = []
    for symbol in symbols:
        frame = fdr.DataReader(symbol, start, end)
        if frame is None or frame.empty:
            continue
        enriched = add_training_labels(add_technical_features(frame))
        all_rows.extend(frame_to_training_rows(enriched, symbol))
    return all_rows


def save_training_rows(rows: Sequence[dict], output_path: str | Path) -> CollectionSummary:
    pd = _require_pandas()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    symbols = sorted({str(row["symbol"]) for row in rows}) if rows else []
    return CollectionSummary(symbols=symbols, row_count=len(rows), output_path=str(path))

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kswing_sentinel.intraday_dataset import collect_multi_timeframe_training_rows, save_multi_timeframe_training_rows


ESTIMATED_SECONDS_PER_SYMBOL_1D = 0.8
ESTIMATED_SECONDS_PER_SYMBOL_60M = 2.0
ESTIMATED_SECONDS_PER_SYMBOL_15M = 2.8


def _to_yahoo_ticker(row: pd.Series) -> str:
    if "YahooTicker" in row and pd.notna(row["YahooTicker"]):
        text = str(row["YahooTicker"]).strip()
        if text:
            return text

    if "Code" not in row:
        raise ValueError("symbols file must contain 'YahooTicker' or 'Code'")

    code = str(row["Code"]).strip().split(".", 1)[0]
    code = "".join(ch for ch in code if ch.isdigit()).zfill(6)
    market_bucket = row.get("market_bucket")
    market = row.get("Market")
    market_value = str(market_bucket if pd.notna(market_bucket) else market).strip().upper()

    if market_value == "KOSPI":
        return f"{code}.KS"
    if market_value == "KOSDAQ":
        return f"{code}.KQ"
    return f"{code}.KS"


def load_symbols(path: str | Path, limit: int | None = None) -> list[str]:
    df = pd.read_csv(path)
    symbols: list[str] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        symbol = _to_yahoo_ticker(row)
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if limit is not None and len(symbols) >= limit:
            break
    return symbols


def _format_seconds(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def estimate_duration(symbol_count: int) -> tuple[float, float, dict[str, float]]:
    per_timeframe = {
        "1d": max(symbol_count, 1) * ESTIMATED_SECONDS_PER_SYMBOL_1D,
        "60m": max(symbol_count, 1) * ESTIMATED_SECONDS_PER_SYMBOL_60M,
        "15m": max(symbol_count, 1) * ESTIMATED_SECONDS_PER_SYMBOL_15M,
    }
    center = sum(per_timeframe.values())
    low = center * 0.7
    high = center * 1.8
    return low, high, per_timeframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect 1d/60m/15m training rows from a universe CSV with ETA output.")
    parser.add_argument("--symbols-file", required=True, help="CSV containing YahooTicker or Code columns.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of symbols for quick tests.")
    parser.add_argument("--daily-period", default="10y", help="Yahoo period for daily download.")
    parser.add_argument("--hourly-period", default="60d", help="Yahoo period for 60m download.")
    parser.add_argument("--intraday-period", default="60d", help="Yahoo period for 15m download.")
    parser.add_argument("--output-dir", default="data/training", help="Output directory for timeframe files.")
    parser.add_argument("--prefix", default="krx500", help="Filename prefix for generated datasets.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    symbols = load_symbols(args.symbols_file, limit=args.limit)
    estimate_low, estimate_high, per_timeframe = estimate_duration(len(symbols))
    print(f"loaded_symbols={len(symbols)}")
    print(
        "estimated_duration="
        f"{_format_seconds(estimate_low)} ~ {_format_seconds(estimate_high)} "
        "(rough estimate; intraday sources vary a lot)"
    )
    print(
        "estimated_breakdown="
        f"1d:{_format_seconds(per_timeframe['1d'])}, "
        f"60m:{_format_seconds(per_timeframe['60m'])}, "
        f"15m:{_format_seconds(per_timeframe['15m'])}"
    )
    rows_by_timeframe = collect_multi_timeframe_training_rows(
        symbols=symbols,
        timeframe_specs={
            "1d": {"period": args.daily_period, "interval": "1d"},
            "60m": {"period": args.hourly_period, "interval": "60m"},
            "15m": {"period": args.intraday_period, "interval": "15m"},
        },
    )
    summaries = save_multi_timeframe_training_rows(rows_by_timeframe, args.output_dir, prefix=args.prefix)
    for timeframe, summary in summaries.items():
        print(f"timeframe={timeframe}")
        print(f"saved_rows={summary.row_count}")
        print(f"symbols={','.join(summary.symbols[:20])}")
        print(f"output={summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

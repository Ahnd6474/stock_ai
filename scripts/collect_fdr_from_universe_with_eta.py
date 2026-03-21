from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kswing_sentinel.fdr_dataset import collect_training_rows, save_training_rows


ESTIMATED_SECONDS_PER_SYMBOL = 0.6


def _extract_code(raw: object) -> str | None:
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    if "." in text:
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(6)


def load_symbols(path: str | Path, limit: int | None = None) -> list[str]:
    df = pd.read_csv(path)
    candidates = None
    if "Code" in df.columns:
        candidates = df["Code"].tolist()
    elif "YahooTicker" in df.columns:
        candidates = df["YahooTicker"].tolist()
    else:
        raise ValueError("symbols file must contain a 'Code' or 'YahooTicker' column")

    symbols: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        code = _extract_code(value)
        if code is None or code in seen:
            continue
        seen.add(code)
        symbols.append(code)
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


def estimate_duration(symbol_count: int) -> tuple[float, float]:
    center = max(symbol_count, 1) * ESTIMATED_SECONDS_PER_SYMBOL
    low = center * 0.7
    high = center * 1.6
    return low, high


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect daily FDR training rows from a universe CSV with ETA output.")
    parser.add_argument("--symbols-file", required=True, help="CSV containing Code or YahooTicker column.")
    parser.add_argument("--start", default="2016-01-01", help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional end date in YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of symbols for quick tests.")
    parser.add_argument(
        "--output",
        default="data/training/krx_500_daily.parquet",
        help="Output path. Supports .csv and .parquet.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    symbols = load_symbols(args.symbols_file, limit=args.limit)
    estimate_low, estimate_high = estimate_duration(len(symbols))
    print(f"loaded_symbols={len(symbols)}")
    print(
        "estimated_duration="
        f"{_format_seconds(estimate_low)} ~ {_format_seconds(estimate_high)} "
        "(rough estimate; depends on network and data source response)"
    )
    rows = collect_training_rows(symbols=symbols, start=args.start, end=args.end)
    summary = save_training_rows(rows, args.output)
    print(f"saved_rows={summary.row_count}")
    print(f"symbols={','.join(summary.symbols[:20])}")
    print(f"output={summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

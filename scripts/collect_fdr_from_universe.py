from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kswing_sentinel.fdr_dataset import collect_training_rows, save_training_rows


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect daily FDR training rows from a universe CSV.")
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
    rows = collect_training_rows(symbols=symbols, start=args.start, end=args.end)
    summary = save_training_rows(rows, args.output)
    print(f"loaded_symbols={len(symbols)}")
    print(f"saved_rows={summary.row_count}")
    print(f"symbols={','.join(summary.symbols[:20])}")
    print(f"output={summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

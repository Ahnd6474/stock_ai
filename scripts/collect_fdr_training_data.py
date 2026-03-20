from __future__ import annotations

import argparse
from datetime import date, timedelta

from kswing_sentinel.fdr_dataset import collect_training_rows, save_training_rows


def _default_start() -> str:
    return (date.today() - timedelta(days=240)).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a small FinanceDataReader training dataset.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["005930", "000660", "035420"],
        help="KRX symbols to download. Default is a small test set.",
    )
    parser.add_argument("--start", default=_default_start(), help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional end date in YYYY-MM-DD.")
    parser.add_argument(
        "--output",
        default="data/training/fdr_training_sample.csv",
        help="Output path. Supports .csv and .parquet.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rows = collect_training_rows(symbols=args.symbols, start=args.start, end=args.end)
    summary = save_training_rows(rows, args.output)
    print(f"saved_rows={summary.row_count}")
    print(f"symbols={','.join(summary.symbols)}")
    print(f"output={summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

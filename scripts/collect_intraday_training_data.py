from __future__ import annotations

import argparse

from kswing_sentinel.intraday_dataset import collect_multi_timeframe_training_rows, save_multi_timeframe_training_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect small daily, hourly, and 15-minute feature datasets.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["005930", "000660"],
        help="KRX symbols to download.",
    )
    parser.add_argument(
        "--daily-period",
        default="1y",
        help="Yahoo period for daily download. Example: 6mo, 1y, 2y.",
    )
    parser.add_argument(
        "--hourly-period",
        default="60d",
        help="Yahoo period for 60m download. Example: 30d, 60d, 120d.",
    )
    parser.add_argument(
        "--intraday-period",
        default="20d",
        help="Yahoo period for 15m download. Example: 5d, 30d, 60d.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training",
        help="Output directory for separate timeframe files.",
    )
    parser.add_argument(
        "--prefix",
        default="market_test",
        help="Filename prefix for generated datasets.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rows_by_timeframe = collect_multi_timeframe_training_rows(
        symbols=args.symbols,
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
        print(f"symbols={','.join(summary.symbols)}")
        print(f"output={summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

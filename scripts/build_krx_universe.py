from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _require_fdr():
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("FinanceDataReader is required. Install with `pip install -e .[marketdata]`.") from exc
    return fdr


BAD_NAME_KEYWORDS = ["스팩", "SPAC", "ETF", "ETN", "리츠", "REIT"]


def _load_krx_listing():
    fdr = _require_fdr()
    last_exc = None
    for market in ("KRX", "KRX-MARCAP"):
        try:
            frame = fdr.StockListing(market)
            if frame is not None and not frame.empty:
                return frame
        except Exception as exc:  # pragma: no cover - network/data source instability
            last_exc = exc
    raise RuntimeError("Unable to load KRX listing via FinanceDataReader.") from last_exc


def _normalize_listing(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    rename_map = {}
    if "Symbol" in df.columns and "Code" not in df.columns:
        rename_map["Symbol"] = "Code"
    if "MarketId" in df.columns and "Market" not in df.columns:
        rename_map["MarketId"] = "Market"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"Code", "Name", "Market", "Marcap"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Listing is missing required columns: {sorted(missing)}")

    df["Code"] = df["Code"].astype(str).str.zfill(6)
    df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
    mask_bad = df["Name"].astype(str).str.contains("|".join(BAD_NAME_KEYWORDS), case=False, na=False)
    df = df[~mask_bad].copy()
    df = df.drop_duplicates(subset=["Code"]).sort_values("Marcap", ascending=False).reset_index(drop=True)
    return df


def _pick_bucket(df: pd.DataFrame, market: str, n_large: int, n_mid: int, n_small: int) -> pd.DataFrame:
    subset = df[df["Market"] == market].copy().sort_values("Marcap", ascending=False).reset_index(drop=True)
    large = subset.iloc[: max(n_large, 0)]
    mid = subset.iloc[max(n_large, 0) : max(n_large, 0) + max(n_mid, 0)]
    small = subset.iloc[-max(n_small, 0) :] if n_small > 0 else subset.iloc[0:0]
    out = pd.concat([large, mid, small], axis=0).drop_duplicates(subset=["Code"]).reset_index(drop=True)
    out["market_bucket"] = market
    return out


def _to_yahoo_ticker(row: pd.Series) -> str:
    code = str(row["Code"]).zfill(6)
    market = str(row["market_bucket"])
    if market == "KOSPI":
        return f"{code}.KS"
    if market == "KOSDAQ":
        return f"{code}.KQ"
    raise ValueError(f"Unsupported market bucket: {market}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a mixed KOSPI/KOSDAQ universe CSV for Kaggle data collection.")
    parser.add_argument("--kospi-large", type=int, default=100, help="Number of KOSPI large-cap symbols.")
    parser.add_argument("--kospi-mid", type=int, default=75, help="Number of KOSPI mid-cap symbols.")
    parser.add_argument("--kospi-small", type=int, default=75, help="Number of KOSPI small-cap symbols.")
    parser.add_argument("--kosdaq-large", type=int, default=100, help="Number of KOSDAQ large-cap symbols.")
    parser.add_argument("--kosdaq-mid", type=int, default=75, help="Number of KOSDAQ mid-cap symbols.")
    parser.add_argument("--kosdaq-small", type=int, default=75, help="Number of KOSDAQ small-cap symbols.")
    parser.add_argument(
        "--output",
        default="data/training/krx_universe_500.csv",
        help="Output CSV path containing Code, Name, Market, Marcap, market_bucket, YahooTicker.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    listing = _normalize_listing(_load_krx_listing())
    kospi = _pick_bucket(listing, "KOSPI", args.kospi_large, args.kospi_mid, args.kospi_small)
    kosdaq = _pick_bucket(listing, "KOSDAQ", args.kosdaq_large, args.kosdaq_mid, args.kosdaq_small)
    universe = pd.concat([kospi, kosdaq], axis=0).drop_duplicates(subset=["Code"]).reset_index(drop=True)
    universe["YahooTicker"] = universe.apply(_to_yahoo_ticker, axis=1)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(path, index=False)

    print(f"saved_universe={path}")
    print(f"rows={len(universe)}")
    print(f"kospi={int((universe['market_bucket'] == 'KOSPI').sum())}")
    print(f"kosdaq={int((universe['market_bucket'] == 'KOSDAQ').sum())}")
    print(universe.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

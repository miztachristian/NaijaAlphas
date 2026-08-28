"""
tv_technicals.py — Pull technical indicators from TradingView's scanner API
and merge them into an existing snapshot parquet.

Adds the data shown on TradingView's per-stock Technicals tab:
  - Aggregated ratings (Summary / Moving Averages / Oscillators gauges)
  - Moving averages (EMA/SMA at 10/20/30/50/100/200 + Ichimoku/VWMA/HullMA)
  - Oscillators missing from the screener export (MACD, ADX, Williams %R,
    Bull Bear Power, Ultimate Oscillator)
  - Classic pivot points (S3..R3)

Usage (PowerShell):
    python -m ingest.tv_technicals                    # latest snapshot
    python -m ingest.tv_technicals --date 2026-05-19
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
SCANNER_URL = "https://scanner.tradingview.com/nigeria/scan"
EXCHANGE_PREFIX = "NSENG"

# TV scanner field -> snapshot column. Names match the convention already used
# by utils/tradingview_snapshot.py (snake_case, _1d suffix where it disambiguates).
FIELD_MAP: Dict[str, str] = {
    # Aggregated ratings (-1..+1; -0.5 strong sell, -0.1 sell, 0.1 buy, 0.5 strong buy)
    "Recommend.All": "technical_rating_1d",
    "Recommend.MA": "moving_averages_rating_1d",
    "Recommend.Other": "oscillators_rating_1d",
    # Moving averages
    "EMA10": "ema_10",
    "EMA20": "ema_20",
    "EMA30": "ema_30",
    "EMA50": "ema_50",
    "EMA100": "ema_100",
    "EMA200": "ema_200",
    "SMA10": "sma_10",
    "SMA20": "sma_20",
    "SMA30": "sma_30",
    "SMA50": "sma_50",
    "SMA100": "sma_100",
    "SMA200": "sma_200",
    "Ichimoku.BLine": "ichimoku_base_line",
    "VWMA": "vwma_20",
    "HullMA9": "hull_ma_9",
    # Oscillators not already in the screener export
    "MACD.macd": "macd",
    "MACD.signal": "macd_signal",
    "ADX": "adx_14",
    "W.R": "williams_r_14",
    "BBPower": "bull_bear_power",
    "UO": "ultimate_oscillator",
    # Classic pivots (monthly)
    "Pivot.M.Classic.S3": "pivot_classic_s3",
    "Pivot.M.Classic.S2": "pivot_classic_s2",
    "Pivot.M.Classic.S1": "pivot_classic_s1",
    "Pivot.M.Classic.Middle": "pivot_classic_middle",
    "Pivot.M.Classic.R1": "pivot_classic_r1",
    "Pivot.M.Classic.R2": "pivot_classic_r2",
    "Pivot.M.Classic.R3": "pivot_classic_r3",
}


def rating_label(value: float) -> str:
    """Map a -1..+1 rating to TradingView's text label."""
    if pd.isna(value):
        return ""
    if value >= 0.5:
        return "Strong Buy"
    if value >= 0.1:
        return "Buy"
    if value > -0.1:
        return "Neutral"
    if value > -0.5:
        return "Sell"
    return "Strong Sell"


def fetch_technicals(symbols: List[str]) -> pd.DataFrame:
    """One POST to TV scanner. Returns DataFrame indexed by symbol."""
    tickers = [f"{EXCHANGE_PREFIX}:{s}" for s in symbols]
    fields = list(FIELD_MAP.keys())

    payload = {
        "symbols": {"tickers": tickers},
        "columns": fields,
    }
    r = requests.post(SCANNER_URL, json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()

    rows = []
    for entry in body.get("data") or []:
        ticker = entry["s"].split(":")[-1]
        row = {"symbol": ticker}
        for field, value in zip(fields, entry["d"]):
            row[FIELD_MAP[field]] = value
        rows.append(row)

    if not rows:
        raise RuntimeError(f"Scanner returned 0 rows for {len(tickers)} tickers")

    return pd.DataFrame(rows).set_index("symbol")


def latest_snapshot_dir() -> Path:
    candidates = [p for p in SNAPSHOTS_DIR.iterdir() if p.is_dir() and (p / "snapshot.parquet").exists()]
    if not candidates:
        sys.exit(f"No snapshots with snapshot.parquet found in {SNAPSHOTS_DIR}")
    return max(candidates, key=lambda p: p.name)


def merge_into_snapshot(snapshot_path: Path, tech_df: pd.DataFrame) -> pd.DataFrame:
    snap = pd.read_parquet(snapshot_path)
    if "symbol" not in snap.columns:
        sys.exit(f"'symbol' column missing in {snapshot_path}")

    # Drop the empty placeholder columns the screener leaves behind so the
    # join brings in fresh values cleanly.
    overlap = [c for c in tech_df.columns if c in snap.columns]
    if overlap:
        snap = snap.drop(columns=overlap)

    merged = snap.merge(tech_df, left_on="symbol", right_index=True, how="left")

    # Convenience text labels next to the numeric ratings.
    for num_col, label_col in [
        ("technical_rating_1d", "technical_rating_1d_label"),
        ("moving_averages_rating_1d", "moving_averages_rating_1d_label"),
        ("oscillators_rating_1d", "oscillators_rating_1d_label"),
    ]:
        if num_col in merged.columns:
            merged[label_col] = merged[num_col].apply(rating_label)

    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Snapshot date (YYYY-MM-DD). Defaults to latest.")
    args = ap.parse_args()

    snap_dir = SNAPSHOTS_DIR / args.date if args.date else latest_snapshot_dir()
    snap_path = snap_dir / "snapshot.parquet"
    if not snap_path.exists():
        sys.exit(f"Snapshot not found: {snap_path}")

    print(f"Snapshot: {snap_path}")
    snap = pd.read_parquet(snap_path)
    symbols = sorted(s for s in snap["symbol"].dropna().unique() if s)
    print(f"Fetching technicals for {len(symbols)} symbols...")

    tech_df = fetch_technicals(symbols)
    print(f"Scanner returned {len(tech_df)} rows × {len(tech_df.columns)} fields")

    missing = [s for s in symbols if s not in tech_df.index]
    if missing:
        print(f"  No technicals for {len(missing)} symbols: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    merged = merge_into_snapshot(snap_path, tech_df)

    # Coverage report on the new columns
    print("\nCoverage of new columns:")
    for col in list(FIELD_MAP.values()) + [
        "technical_rating_1d_label",
        "moving_averages_rating_1d_label",
        "oscillators_rating_1d_label",
    ]:
        if col in merged.columns:
            n = merged[col].notna().sum() if not col.endswith("_label") else (merged[col] != "").sum()
            print(f"  {col:38s} {n:3d}/{len(merged)}")

    merged.to_parquet(snap_path, index=False)
    csv_path = snap_dir / "snapshot.csv"
    merged.to_csv(csv_path, index=False)
    print(f"\nWrote {snap_path}")
    print(f"Wrote {csv_path}")

    # Quick gauge summary
    if "technical_rating_1d_label" in merged.columns:
        counts = merged["technical_rating_1d_label"].value_counts()
        print("\nSummary rating distribution:")
        for label in ["Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"]:
            print(f"  {label:12s} {int(counts.get(label, 0)):3d}")


if __name__ == "__main__":
    main()

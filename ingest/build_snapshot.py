"""
build_snapshot.py — build a base snapshot directly from the TradingView
scanner API, removing the manual 12-CSV download for a brand-new trading day.

A whole-market scan of the Nigerian exchange returns identity, price, volume,
sector and performance columns for every listed name. The richer fundamental
and technical columns are then layered on by the existing, proven enrichers
(ingest/tv_financials.py and ingest/tv_technicals.py), which daily_ingest.py
runs after this.

Note: the manual TradingView screener export still gives the widest fundamental
coverage. This base + API enrichers is the automated path; it degrades
gracefully — thinner data simply lowers each stock's confidence.

Usage:
    python -m ingest.build_snapshot                 # today
    python -m ingest.build_snapshot --date 2026-05-22
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR  # noqa: E402

SCANNER_URL = "https://scanner.tradingview.com/nigeria/scan"

# TradingView scanner field -> snapshot column name. Kept to identity / price /
# volume / sector / performance — the columns a whole-market scan reliably
# returns. Fundamentals/technicals are added afterward by the enrichers.
BASE_COLUMNS = {
    "name": "symbol",
    "description": "description",
    "close": "price",
    "volume": "volume_1d",
    "sector": "sector",
    "market_cap_basic": "market_cap",
    "Perf.W": "perf_1w",
    "Perf.1M": "perf_1m",
    "Perf.3M": "perf_3m",
    "Perf.6M": "perf_6m",
    "Perf.Y": "perf_1y",
    "Perf.YTD": "perf_ytd",
    "RSI": "rsi_14",
}


def build_base_snapshot(date: str | None = None) -> Path:
    """Whole-market scan -> data/snapshots/<date>/snapshot.parquet (+ .csv)."""
    date = date or datetime.now().strftime("%Y-%m-%d")

    payload = {"columns": list(BASE_COLUMNS.keys()), "range": [0, 1000]}
    resp = requests.post(SCANNER_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data") or []
    if not data:
        raise RuntimeError("Scanner returned no rows for the Nigerian market.")

    fields = list(BASE_COLUMNS.keys())
    rows = []
    for entry in data:
        row = {}
        for field, value in zip(fields, entry["d"]):
            row[BASE_COLUMNS[field]] = value
        if not row.get("symbol"):
            row["symbol"] = entry["s"].split(":")[-1]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["symbol"]).drop_duplicates(subset="symbol", keep="first")
    df["snapshot_date"] = date

    snap_dir = DATA_DIR / "snapshots" / date
    snap_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = snap_dir / "snapshot.parquet"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(snap_dir / "snapshot.csv", index=False)
    return parquet_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a base snapshot from the TV scanner API.")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    path = build_base_snapshot(args.date)
    df = pd.read_parquet(path)
    print(f"Built base snapshot: {path}")
    print(f"  {len(df)} symbols x {len(df.columns)} base columns")
    print("  Run ingest.tv_technicals and ingest.tv_financials to enrich.")


if __name__ == "__main__":
    main()

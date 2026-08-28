"""
tv_financials.py — Pull every fundamental/financial field TV exposes that
isn't already in our snapshot, and merge it in.

Strategy:
  1. Fetch the TV scanner metainfo to enumerate all ~3500 available fields.
  2. Keep only number/percent/fundamental_price fields (drop text-only and
     currency-helper fields).
  3. Drop fields that are obviously technical indicators (already covered by
     ingest/tv_technicals.py).
  4. Drop fields whose name collides with a column already in the snapshot.
  5. Batch the survivors into chunks (TV caps per-request column count) and
     POST against the scanner for all NSENG tickers.
  6. Merge results into the snapshot, then drop columns where coverage is so
     low that the field clearly has no data for the Nigerian market.

Usage:
    python -m ingest.tv_financials                 # latest snapshot
    python -m ingest.tv_financials --date 2026-05-19
    python -m ingest.tv_financials --min-coverage 0.05   # keep cols with >=5% coverage
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
SCANNER_URL = "https://scanner.tradingview.com/nigeria/scan"
METAINFO_URL = "https://scanner.tradingview.com/nigeria/metainfo"
EXCHANGE_PREFIX = "NSENG"

# Keep numeric-ish fields. fundamental_price is TV's type for per-share fundamentals.
NUMERIC_TYPES = {"number", "percent", "fundamental_price", "price", "interface", "duration"}

# Skip these technical-indicator field name patterns — they're already in the
# snapshot via tv_technicals.py or the screener CSVs.
TECHNICAL_PREFIXES = (
    "RSI", "Stoch", "MACD", "ADX", "ATR", "AO", "Mom", "CCI", "BB", "ROC",
    "DMI", "Aroon", "PSAR", "BBPower", "UO", "W.R", "Recommend", "EMA", "SMA",
    "WMA", "HullMA", "VWMA", "Ichimoku", "Pivot", "Volatility", "Perf",
    "ChaikinMoneyFlow", "DonchianChannels", "KltChnl", "MoneyFlow",
)

# Fields we definitely don't want — price/volume snapshots and TV plumbing.
SKIP_EXACT = {
    "close", "open", "high", "low", "Volatility.D", "Volatility.W", "Volatility.M",
    "Volume.1", "Volume.5", "Volume.60", "Volume.W", "Volume.M",
    "change", "change_abs", "change.1M", "change.1W", "change.1Y", "change.3M",
    "change.5Y", "change.6M", "change.YTD", "change.60", "change.5",
    "premarket_change", "premarket_change_abs", "premarket_close", "premarket_gap",
    "premarket_high", "premarket_low", "premarket_open", "premarket_volume",
    "postmarket_change", "postmarket_change_abs", "postmarket_close",
    "postmarket_high", "postmarket_low", "postmarket_open", "postmarket_volume",
    "premarket_volume_change", "Value.Traded", "average_volume",
    "average_volume_10d_calc", "average_volume_30d_calc", "average_volume_60d_calc",
    "average_volume_90d_calc", "relative_volume_10d_calc", "relative_volume_intraday",
    "ask", "bid", "last_traded_time", "last_bar_update_time", "exchange",
    "type", "subtype", "country_code", "country", "industry",
    # We already have basic IDs
    "name", "description", "logoid",
}

# Skip patterns that aren't financials — TV plumbing or extreme detail.
SKIP_PATTERNS = (
    re.compile(r"^pricescale$|^minmov|^pointvalue$|^session"),
    re.compile(r"^source[12]?_id$|^source[12]?_currency_id$"),
    re.compile(r"^update_mode$|^market_$|^typespecs$"),
    re.compile(r"_currency$"),  # currency labels paired with monetary fields
    re.compile(r"^Volatility$"),
    re.compile(r"^figi"),
    re.compile(r"^isin$|^cusip$|^cik$|^lei$"),
    re.compile(r"^last_split"),
    re.compile(r"^fundamental_data_release"),
)

BATCH_SIZE = 250


def fetch_metainfo() -> List[dict]:
    r = requests.get(METAINFO_URL, timeout=30)
    r.raise_for_status()
    return r.json()["fields"]


def select_financial_fields(meta: List[dict], existing_cols: set) -> List[str]:
    keep: List[str] = []
    for f in meta:
        name = f["n"]
        ftype = f.get("t")

        if ftype not in NUMERIC_TYPES:
            continue
        if name in SKIP_EXACT:
            continue
        if name in existing_cols:
            continue
        if any(name.startswith(p) for p in TECHNICAL_PREFIXES):
            continue
        if any(p.search(name) for p in SKIP_PATTERNS):
            continue

        keep.append(name)
    return keep


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def fetch_batch(tickers: List[str], fields: List[str]) -> pd.DataFrame:
    payload = {"symbols": {"tickers": tickers}, "columns": fields}
    r = requests.post(SCANNER_URL, json=payload, timeout=60)
    r.raise_for_status()
    body = r.json()

    rows = []
    for entry in body.get("data") or []:
        ticker = entry["s"].split(":")[-1]
        row = {"symbol": ticker}
        for field, value in zip(fields, entry["d"]):
            row[field] = value
        rows.append(row)
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()


def latest_snapshot_dir() -> Path:
    candidates = [p for p in SNAPSHOTS_DIR.iterdir() if p.is_dir() and (p / "snapshot.parquet").exists()]
    if not candidates:
        sys.exit(f"No snapshots with snapshot.parquet found in {SNAPSHOTS_DIR}")
    return max(candidates, key=lambda p: p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Snapshot date (YYYY-MM-DD). Defaults to latest.")
    ap.add_argument("--min-coverage", type=float, default=0.05,
                    help="Drop pulled columns where fewer than this fraction of "
                         "stocks have data (default 0.05 = 5%).")
    args = ap.parse_args()

    snap_dir = SNAPSHOTS_DIR / args.date if args.date else latest_snapshot_dir()
    snap_path = snap_dir / "snapshot.parquet"
    if not snap_path.exists():
        sys.exit(f"Snapshot not found: {snap_path}")

    print(f"Snapshot: {snap_path}")
    snap = pd.read_parquet(snap_path)
    symbols = sorted(s for s in snap["symbol"].dropna().unique() if s)
    tickers = [f"{EXCHANGE_PREFIX}:{s}" for s in symbols]
    print(f"Stocks: {len(tickers)}")

    print("Fetching scanner metainfo...")
    meta = fetch_metainfo()
    existing_cols = set(snap.columns)
    fields = select_financial_fields(meta, existing_cols)
    print(f"  {len(meta)} total fields advertised by scanner")
    print(f"  {len(fields)} financial fields not already in snapshot")

    print(f"\nPulling in batches of {BATCH_SIZE} columns...")
    frames: List[pd.DataFrame] = []
    for i, batch in enumerate(chunked(fields, BATCH_SIZE), 1):
        try:
            df = fetch_batch(tickers, batch)
        except requests.HTTPError as e:
            print(f"  batch {i} failed ({len(batch)} cols): {e}; bisecting...")
            # Try halving — typically reveals a single bad field
            half = max(1, len(batch) // 2)
            try:
                df_a = fetch_batch(tickers, batch[:half])
                df_b = fetch_batch(tickers, batch[half:])
                df = pd.concat([df_a, df_b], axis=1)
            except requests.HTTPError as e2:
                print(f"    bisect also failed: {e2}; dropping batch")
                continue
        if not df.empty:
            frames.append(df)
        print(f"  batch {i:>3}: +{df.shape[1]} cols ({df.shape[0]} rows)")
        time.sleep(0.2)

    if not frames:
        sys.exit("No data fetched.")

    fin_df = pd.concat(frames, axis=1)
    # De-dup any columns that bisection may have re-added.
    fin_df = fin_df.loc[:, ~fin_df.columns.duplicated()]
    print(f"\nFetched {fin_df.shape[1]} columns × {fin_df.shape[0]} stocks")

    # Drop low-coverage columns.
    coverage = fin_df.notna().sum() / len(fin_df)
    keep_cols = coverage[coverage >= args.min_coverage].index.tolist()
    dropped = fin_df.shape[1] - len(keep_cols)
    print(f"Dropping {dropped} columns below {args.min_coverage:.0%} coverage; keeping {len(keep_cols)}.")
    fin_df = fin_df[keep_cols]

    # Merge into snapshot (left join on symbol). Any final collisions go right-side wins.
    overlap = [c for c in fin_df.columns if c in snap.columns]
    if overlap:
        snap = snap.drop(columns=overlap)
        print(f"Replaced {len(overlap)} overlapping columns with fresh values.")

    merged = snap.merge(fin_df, left_on="symbol", right_index=True, how="left")
    print(f"\nSnapshot grew from {len(snap.columns)} → {len(merged.columns)} columns.")

    merged.to_parquet(snap_path, index=False)
    csv_path = snap_dir / "snapshot.csv"
    merged.to_csv(csv_path, index=False)
    print(f"Wrote {snap_path}")
    print(f"Wrote {csv_path}")

    # Save the field catalog we pulled for future reference.
    catalog_path = snap_dir / "financials_fields.txt"
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write(f"# Financial fields pulled on {snap_dir.name}\n")
        f.write(f"# {len(keep_cols)} fields, min coverage {args.min_coverage:.0%}\n\n")
        for col in sorted(keep_cols):
            n = int(merged[col].notna().sum())
            f.write(f"{col:60s} {n:3d}/{len(merged)}\n")
    print(f"Wrote {catalog_path}")


if __name__ == "__main__":
    main()

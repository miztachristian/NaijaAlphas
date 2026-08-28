"""
macro.py — Nigeria macro / economic data ingestion.
====================================================
Pulls the five macro series that move the NGX, with graceful fallback:

  * CBN Monetary Policy Rate (MPR)   — manually maintained constant (changes
    only at MPC meetings; no clean CBN API). Override in config MacroConfig.
  * Headline inflation (NBS CPI YoY) — manually maintained constant.
  * USD/NGN spot                     — live, free FX API.
  * Brent crude (USD/bbl)            — live, stooq daily CSV.
  * 10-Year FGN bond yield, %        — manually maintained constant (FMDQ /
    DMO auction-driven; refresh after each new issue).

Output:
  data/macro/<YYYY-MM-DD>.json   one record per run, each field with provenance
  data/macro/history.parquet     append-only time series (for trend slopes)

A run where a *live* source (FX, Brent) cannot be reached is flagged
`degraded` — the MPR/inflation constants are expected, not a degradation.

Usage:
    python -m ingest.macro                 # fetch + store for today
    python -m ingest.macro --date 2026-05-22
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import MacroConfig, MACRO_DIR  # noqa: E402

HISTORY_PATH = MACRO_DIR / "history.parquet"
HTTP_TIMEOUT = 15


# --------------------------------------------------------------------------
# Individual source fetchers — each returns (value, source_label, is_live)
# --------------------------------------------------------------------------
def fetch_usdngn() -> tuple[float, str, bool]:
    """USD/NGN spot from a free FX API; falls back to a constant."""
    try:
        resp = requests.get(MacroConfig.USDNGN_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        rate = float(resp.json()["rates"]["NGN"])
        if rate > 0:
            return rate, "open.er-api.com", True
    except Exception as exc:  # noqa: BLE001 - network best-effort
        print(f"  ! USD/NGN live fetch failed: {exc}")
    return MacroConfig.FALLBACK_USDNGN, "fallback_config", False


def fetch_brent() -> tuple[float, str, bool]:
    """Brent crude (USD/bbl) from stooq's daily CSV; falls back to a constant."""
    try:
        resp = requests.get(MacroConfig.BRENT_STOOQ_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if not df.empty and "Close" in df.columns:
            close = float(df["Close"].dropna().iloc[-1])
            if close > 0:
                return close, "stooq.com", True
    except Exception as exc:  # noqa: BLE001 - network best-effort
        print(f"  ! Brent live fetch failed: {exc}")
    return MacroConfig.FALLBACK_BRENT, "fallback_config", False


def fetch_mpr() -> tuple[float, str, bool]:
    """CBN Monetary Policy Rate — manually maintained constant (MPC-driven)."""
    return MacroConfig.FALLBACK_MPR, "manual_config", True


def fetch_inflation() -> tuple[float, str, bool]:
    """Headline inflation (NBS CPI YoY) — manually maintained constant."""
    return MacroConfig.FALLBACK_INFLATION, "manual_config", True


def fetch_bond10y() -> tuple[float, str, bool]:
    """10-Year FGN bond yield — manually maintained constant (FMDQ/DMO-driven)."""
    return MacroConfig.FALLBACK_BOND10Y, "manual_config", True


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def fetch_macro(date: str | None = None) -> dict:
    """Fetch all four series and return a record with per-field provenance."""
    date = date or datetime.now().strftime("%Y-%m-%d")

    fields = {
        "mpr": fetch_mpr(),
        "inflation": fetch_inflation(),
        "usdngn": fetch_usdngn(),
        "brent": fetch_brent(),
        "bond_10y": fetch_bond10y(),
    }

    record: dict = {"date": date, "fetched_at": datetime.now().isoformat()}
    sources: dict = {}
    degraded = False
    for name, (value, source, is_live) in fields.items():
        record[name] = value
        sources[name] = {"value": value, "source": source, "asof": date}
        # A *live* source falling back to a constant => degraded run.
        if source == "fallback_config":
            degraded = True
    record["sources"] = sources
    record["degraded"] = degraded
    return record


def save_macro(record: dict) -> Path:
    """Write the per-run JSON and append to the rolling history parquet."""
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    date = record["date"]

    json_path = MACRO_DIR / f"{date}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    # Append to history.parquet, de-duplicating on date (keep latest run).
    row = {k: record[k] for k in ("date", "mpr", "inflation", "usdngn", "brent", "bond_10y")}
    new = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        hist = pd.read_parquet(HISTORY_PATH)
        hist = hist[hist["date"] != date]
        hist = pd.concat([hist, new], ignore_index=True)
    else:
        hist = new
    hist = hist.sort_values("date").reset_index(drop=True)
    hist.to_parquet(HISTORY_PATH, index=False)

    return json_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Nigeria macro data.")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    print("=" * 60)
    print("  Nigeria Macro Ingestion")
    print("=" * 60)

    record = fetch_macro(args.date)
    json_path = save_macro(record)

    print(f"\n  MPR        : {record['mpr']:.2f}%   ({record['sources']['mpr']['source']})")
    print(f"  Inflation  : {record['inflation']:.2f}%   ({record['sources']['inflation']['source']})")
    print(f"  USD/NGN    : {record['usdngn']:.2f}   ({record['sources']['usdngn']['source']})")
    print(f"  Brent      : ${record['brent']:.2f}   ({record['sources']['brent']['source']})")
    print(f"  10Y FGN    : {record['bond_10y']:.2f}%   ({record['sources']['bond_10y']['source']})")
    print(f"  Degraded   : {record['degraded']}")
    print(f"\n  Saved: {json_path}")
    print(f"  History: {HISTORY_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_report_scrape.py
──────────────────────
Standalone script to scrape annual-report data for all Nigerian stocks
from one or both sources and produce an analysis summary.

Sources:
    aff      — AfricanFinancials AI summaries (primary, richest data)
    proshare — ProShare listing + PDF URL (fills AFF coverage gaps)
    both     — AFF first, ProShare fallback for missing tickers (default)

Usage:
    python run_report_scrape.py                            # all tickers, both sources
    python run_report_scrape.py --source aff               # AFF only
    python run_report_scrape.py --source proshare          # ProShare only
    python run_report_scrape.py --tickers DANGCEM MTNN GTCO
    python run_report_scrape.py --year 2024
    python run_report_scrape.py --no-cache                 # force fresh scrape
    python run_report_scrape.py --no-playwright            # skip ProShare PDF enrichment
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 crash on box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.fetch_annual_reports import TICKER_TO_AFF_SLUG
from ingest.fetch_reports import fetch_all_reports
from analysis.report_analyzer import ReportAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape & analyse annual-report data from AFF and/or ProShare"
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Specific tickers to scrape (default: all tickers in latest snapshot)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Target fiscal year (default: most recent available)",
    )
    parser.add_argument(
        "--source",
        choices=["aff", "proshare", "both"],
        default="both",
        help="Which scraper(s) to run (default: both)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache, force fresh scrape",
    )
    parser.add_argument(
        "--no-playwright",
        action="store_true",
        help="Skip ProShare detail-page enrichment (faster, no PDF URLs)",
    )
    args = parser.parse_args()

    # Determine ticker list
    if args.tickers:
        tickers = args.tickers
    else:
        try:
            import pandas as pd
            data_dir = Path("data/snapshots")
            latest_snap = sorted(data_dir.glob("*/snapshot.parquet"))[-1]
            df = pd.read_parquet(latest_snap)
            tickers = sorted(df['symbol'].unique().tolist())
        except Exception as e:
            print(f"Could not load tickers from snapshot ({e}), falling back to mapped tickers.")
            tickers = sorted(TICKER_TO_AFF_SLUG.keys())

    print(f"\n[*] Scraping {len(tickers)} tickers (source={args.source}) ...")
    if args.year:
        print(f"    Target year: {args.year}")
    if args.source in ("proshare", "both") and args.no_playwright:
        print("    ProShare Playwright enrichment: DISABLED (metadata-only)")
    print()

    # ── Scrape via unified orchestrator ──
    reports = fetch_all_reports(
        tickers,
        year=args.year,
        source=args.source,
        use_cache=not args.no_cache,
        use_playwright=not args.no_playwright,
    )

    # ── Coverage stats ──
    found = len(reports)
    total = len(tickers)
    found_snapshot_tickers = {r.get("snapshot_ticker", k) for k, r in reports.items()}
    missing = sorted(set(tickers) - found_snapshot_tickers)
    pct = found / total * 100 if total else 0

    # Per-source breakdown
    by_source: dict[str, int] = {}
    for r in reports.values():
        by_source[r.get("source", "unknown")] = by_source.get(r.get("source", "unknown"), 0) + 1

    print(f"\n{'═' * 60}")
    print(f"  Coverage: {found}/{total} tickers ({pct:.0f}%)")
    for src, n in sorted(by_source.items()):
        print(f"    • {src}: {n}")
    print(f"{'═' * 60}")

    if missing:
        print(f"\n  Missing ({len(missing)}): {', '.join(missing)}")

    # ── Analyse ──
    if found > 0:
        analyzer = ReportAnalyzer()
        results = analyzer.analyze_batch(reports)
        summary = analyzer.summarize_batch(results)
        print(f"\n{summary}")

        # Save results to outputs/
        outdir = Path("outputs")
        outdir.mkdir(exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        outpath = outdir / f"ngx_report_signals_{today}.txt"
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(f"NGX Report Analysis — {today}\n")
            f.write(f"Coverage: {found}/{total} ({pct:.0f}%)\n")
            if missing:
                f.write(f"Missing: {', '.join(missing)}\n")
            f.write(f"\n{summary}\n")
        print(f"\n[+] Saved to {outpath}")
    else:
        print("\n  No reports found — nothing to analyse.")


if __name__ == "__main__":
    main()

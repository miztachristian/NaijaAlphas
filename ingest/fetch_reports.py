"""
Unified multi-source report fetcher
====================================
Orchestrates the two report scrapers we maintain:

  • AfricanFinancials (primary)  — AI-generated summaries with KPIs
  • ProShare           (fallback) — listing + PDF URL for tickers AFF misses

AFF wins on merge when both sources have the same ticker, because its
AI summary is richer than ProShare's metadata + PDF link.

Usage
-----
    from ingest.fetch_reports import fetch_all_reports
    reports = fetch_all_reports(tickers, source="both")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from ingest.fetch_annual_reports import ReportScraper
from ingest.fetch_proshare_reports import ProShareScraper

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "data" / "reports"

Source = Literal["aff", "proshare", "both"]


def _unified_cache_path(date_str: Optional[str] = None) -> Path:
    ds = date_str or datetime.now().strftime("%Y-%m-%d")
    return REPORTS_DIR / f"unified-{ds}.json"


def _save_unified_cache(data: dict, date_str: Optional[str] = None) -> None:
    p = _unified_cache_path(date_str)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Unified cache saved -> %s", p)


def fetch_all_reports(
    tickers: list[str],
    year: Optional[int] = None,
    source: Source = "both",
    use_cache: bool = True,
    use_playwright: bool = True,
) -> dict[str, dict]:
    """
    Fetch report data from one or both sources, merging with AFF priority.

    Parameters
    ----------
    tickers        : list of NGX ticker symbols
    year           : target fiscal year; defaults to most recent available
    source         : "aff", "proshare", or "both" (default)
    use_cache      : honour today's cached results
    use_playwright : enable ProShare detail-page enrichment (PDF URLs)

    Returns
    -------
    dict {ticker: report_dict}. Each record includes a "source" key so
    downstream code can tell AFF and ProShare entries apart.
    """
    tickers_upper = [t.upper() for t in tickers]
    aff_reports: dict[str, dict] = {}
    proshare_reports: dict[str, dict] = {}

    if source in ("aff", "both"):
        logger.info("[fetch_reports] AFF pass over %d tickers", len(tickers_upper))
        aff = ReportScraper(use_cache=use_cache)
        aff_reports = aff.fetch_all(tickers_upper, year=year)
        # Tag every AFF record explicitly (older caches may lack "source")
        for r in aff_reports.values():
            r.setdefault("source", "africanfinancials")

    if source in ("proshare", "both"):
        # When running "both", only hit ProShare for tickers AFF missed
        to_fetch = (
            [t for t in tickers_upper if not any(r.get("snapshot_ticker", k) == t for k, r in aff_reports.items())]
            if source == "both"
            else tickers_upper
        )
        if to_fetch:
            logger.info(
                "[fetch_reports] ProShare pass over %d tickers (%s)",
                len(to_fetch),
                "fallback" if source == "both" else "primary",
            )
            ps = ProShareScraper(use_cache=use_cache, use_playwright=use_playwright)
            proshare_reports = ps.fetch_all(to_fetch, year=year)
        else:
            logger.info("[fetch_reports] ProShare pass skipped — AFF covered everything")

    # Merge: AFF takes priority (richer summaries)
    merged = {**proshare_reports, **aff_reports}

    _save_unified_cache(merged)
    logger.info(
        "[fetch_reports] merged %d AFF + %d ProShare => %d unique tickers",
        len(aff_reports), len(proshare_reports), len(merged),
    )
    return merged


__all__ = ["fetch_all_reports"]

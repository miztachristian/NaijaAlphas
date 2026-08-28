"""
NGX Earnings-Forecast Parser
============================
Downloads the earnings-forecast PDFs filed on NGX and extracts the structured
forecast P&L — Revenue, Profit Before Tax, Profit After Tax — into a
per-ticker JSON cache.

Earnings forecasts are filed quarterly and carry the company's *own* forward
guidance. The analysis system otherwise has no forward earnings input
(only yfinance's `forward_pe`).

Coverage note: issuers file these in two formats — digital PDFs with a real
text layer (parsed here) and scanned-image PDFs with none. Scanned filings
are recorded with ``parse_status="unparseable"``; an OCR pass (see the
project plan follow-ups) could recover them later.

Artefacts:
  data/disclosures/forecast_pdfs/<file>.pdf   — cached source PDF
  data/disclosures/forecasts-<date>.json      — parsed forecasts by ticker

Usage:
    from ingest.parse_forecasts import parse_forecasts
    data = parse_forecasts()                      # latest forecast per ticker
    data = parse_forecasts(tickers=["VFDGROUP"])

Or from the command line:
    python -m ingest.parse_forecasts --ticker VFDGROUP --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

import requests
from pypdf import PdfReader

from ingest.fetch_disclosures import fetch_disclosures

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DISCLOSURES_DIR = BASE_DIR / "data" / "disclosures"
FORECAST_PDF_DIR = DISCLOSURES_DIR / "forecast_pdfs"
FORECAST_PDF_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
}
REQUEST_TIMEOUT = (10, 60)
POLITE_DELAY = 0.5

# ── P&L line-label patterns ──────────────────────────────────────────
# Checked in order; PAT before PBT so "profit before tax" can't be
# mis-captured as "profit ... after tax". Each pattern is anchored to the
# start of a (lower-cased, stripped) line.
LABEL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("forecast_pat", re.compile(
        r"profit\s*(?:/?\(?loss\)?)?\s*(?:for\s+the\s+(?:period|year)|"
        r"after\s+tax(?:ation)?)\b")),
    ("forecast_pbt", re.compile(
        r"profit\s*(?:/?\(?loss\)?)?\s*before\s+tax(?:ation)?\b")),
    ("forecast_revenue", re.compile(
        r"(?:gross\s+earnings|gross\s+premium\s+written|insurance\s+revenue|"
        r"turnover|(?:total\s+|net\s+)?revenue)\b")),
]

# A financial number is either comma-grouped in strict 3-digit blocks
# (e.g. "9,125,351") or plain ungrouped digits. Requiring strict grouping
# stops two merged columns ("9,125,351139") from parsing as one number —
# the grouped match ends cleanly at the malformed boundary.
_NUMBER = re.compile(r"\(?\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*\)?")
_PERIOD = re.compile(
    r"\b(Q\s?[1-4]|quarter\s+[1-4]|(?:first|second|third|fourth)\s+quarter)\b"
    r".{0,40}?(20\d\d)", re.IGNORECASE)
_QUARTER_WORD = {"FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4"}


def _to_number(token: str) -> Optional[float]:
    """Parse a financial number token; parentheses denote a negative."""
    negative = "(" in token
    digits = re.sub(r"[^\d.]", "", token)
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def _cache_name(pdf_url: str) -> str:
    """Local cache filename for a PDF URL (its url-decoded basename)."""
    return unquote(urlsplit(pdf_url).path.rsplit("/", 1)[-1]) or "forecast.pdf"


def _download(pdf_url: str) -> Optional[Path]:
    """Download a PDF to the cache (skip if already present)."""
    dest = FORECAST_PDF_DIR / _cache_name(pdf_url)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("[forecasts] download failed %s: %s", pdf_url, exc)
        return None
    dest.write_bytes(resp.content)
    time.sleep(POLITE_DELAY)
    return dest


def _extract_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as exc:  # pypdf raises a variety of errors on odd PDFs
        logger.warning("[forecasts] could not read %s: %s", pdf_path.name, exc)
        return ""


def _parse_pnl(text: str) -> dict:
    """Pull Revenue / PBT / PAT out of forecast PDF text."""
    found: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        for metric, pattern in LABEL_PATTERNS:
            if metric in found:
                continue
            m = pattern.match(low)
            if not m:
                continue
            num = _NUMBER.search(line, m.end())
            if num:
                value = _to_number(num.group())
                if value is not None:
                    found[metric] = value
    return found


def _validate(record: dict) -> None:
    """Flag a parsed record as 'suspect' when the figures fail basic sanity.

    Catches column-merge misparses and OCR noise: forecasts are filed in
    whole thousands, profit-after-tax cannot exceed profit-before-tax, and
    revenue must exceed PBT. A suspect record is not trusted downstream.
    """
    rev = record["forecast_revenue"]
    pbt = record["forecast_pbt"]
    pat = record["forecast_pat"]
    present = [v for v in (rev, pbt, pat) if v is not None]
    if not present:
        return

    # Magnitude ceilings (values are N'000): no NGX issuer forecasts a
    # profit above ~₦2tn or revenue above ~₦20tn. A value past these is a
    # column-merge misparse or a unit error.
    profit_ceiling = 2_000_000_000      # ₦2tn in N'000
    revenue_ceiling = 20_000_000_000    # ₦20tn in N'000

    issues: list[str] = []
    if any(v != int(v) for v in present):
        issues.append("fractional value (filings are whole thousands)")
    if pbt is not None and pat is not None and pbt > 0 and pat > pbt * 1.02:
        issues.append("PAT exceeds PBT")
    if rev is not None and pbt is not None and rev > 0 and pbt > rev:
        issues.append("PBT exceeds revenue")
    if rev is not None and rev <= 0:
        issues.append("non-positive revenue")
    if pbt is not None and abs(pbt) > profit_ceiling:
        issues.append("PBT magnitude implausible")
    if pat is not None and abs(pat) > profit_ceiling:
        issues.append("PAT magnitude implausible")
    if rev is not None and abs(rev) > revenue_ceiling:
        issues.append("revenue magnitude implausible")

    if issues:
        record["parse_status"] = "suspect"
        record["parse_issues"] = issues


def _extract_period(text: str, title: str) -> str:
    """Best-effort 'Q2 2026'-style period string from the PDF or its title."""
    for source in (text[:600], title):
        m = _PERIOD.search(source)
        if not m:
            continue
        token = m.group(1).upper()
        digit = next((d for word, d in _QUARTER_WORD.items() if word in token), None)
        if digit is None:
            hit = re.search(r"[1-4]", token)
            digit = hit.group() if hit else "?"
        return f"Q{digit} {m.group(2)}"
    return ""


def parse_forecasts(
    tickers: Optional[list[str]] = None,
    since: str = "2019-01-31",
    use_cache: bool = True,
    limit: Optional[int] = None,
) -> dict[str, dict]:
    """
    Parse the latest earnings forecast for each ticker.

    Returns {ticker: forecast dict}. Each forecast dict has: period,
    forecast_revenue, forecast_pbt, forecast_pat (raw N'000 as filed),
    unit, filed_date, parse_status ("ok" | "partial" | "unparseable"),
    source_pdf.
    """
    rows = fetch_disclosures(tickers=tickers, since=since,
                             categories=["earnings_forecast"], use_cache=use_cache)
    # rows are newest-first; keep only the most recent forecast per ticker
    latest: dict[str, dict] = {}
    for r in rows:
        latest.setdefault(r["ticker"], r)

    items = list(latest.items())
    if limit:
        items = items[:limit]

    results: dict[str, dict] = {}
    for ticker, row in items:
        pdf_path = _download(row["pdf_url"])
        record = {
            "ticker": ticker,
            "period": "",
            "forecast_revenue": None,
            "forecast_pbt": None,
            "forecast_pat": None,
            "unit": "NGN'000",
            "filed_date": row["created"][:10],
            "parse_status": "unparseable",
            "source_pdf": row["pdf_url"],
        }
        if pdf_path:
            text = _extract_text(pdf_path)
            pnl = _parse_pnl(text)
            record.update(pnl)
            record["period"] = _extract_period(text, row["title"])
            if len(pnl) >= 3:
                record["parse_status"] = "ok"
            elif pnl:
                record["parse_status"] = "partial"
            _validate(record)
        results[ticker] = record
        logger.info("[forecasts] %-12s %s  PBT=%s PAT=%s", ticker,
                    record["parse_status"], record["forecast_pbt"],
                    record["forecast_pat"])

    _save(results)
    return results


def _save(results: dict[str, dict]) -> None:
    from collections import Counter
    out = DISCLOSURES_DIR / f"forecasts-{datetime.now():%Y-%m-%d}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    counts = dict(Counter(r["parse_status"] for r in results.values()))
    logger.info("[forecasts] saved %d forecasts %s -> %s",
                len(results), counts, out)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse NGX earnings forecasts.")
    parser.add_argument("--ticker", action="append", dest="tickers",
                        help="restrict to ticker (repeatable)")
    parser.add_argument("--since", default="2019-01-31", help="earliest date")
    parser.add_argument("--limit", type=int, default=None, help="max tickers")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the disclosure cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = parse_forecasts(tickers=args.tickers, since=args.since,
                           use_cache=not args.no_cache, limit=args.limit)
    print(f"\n{len(data)} forecasts\n" + "-" * 72)
    for t, r in sorted(data.items()):
        print(f"{t:<13} {r['parse_status']:<12} {r['period']:<9} "
              f"rev={r['forecast_revenue']}  pbt={r['forecast_pbt']}  "
              f"pat={r['forecast_pat']}")


__all__ = ["parse_forecasts"]


if __name__ == "__main__":
    _main()

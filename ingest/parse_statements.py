"""
NGX Financial-Statement Parser
==============================
Downloads the financial-statement PDFs filed on NGX (audited annual and
quarterly results) and extracts the headline figures — Revenue, Profit
Before Tax, Profit After Tax, Total Assets, Total Equity — for the current
and prior reporting period.

These are *official* NGX-filed numbers. The analysis system otherwise relies
on Yahoo Finance fundamentals, which are sparse and unreliable for Nigerian
equities; the parsed figures let `FundamentalAnalyzer` validate or replace them.

Parsing approach — the PDFs are 50-200 pages with layouts that vary by sector:
- The Statement of Profit or Loss page yields Revenue / PBT / PAT.
- The Statement of Financial Position page yields Total Assets / Total Equity.
- Only *comma-grouped* numbers are read as values, which skips the small
  note-reference numbers printed between a label and its figures
  (e.g. "Revenue  7  330,639,543" -> 330,639,543, not 7).
Every record is sanity-checked; figures that fail are flagged "suspect".

Artefacts:
  data/disclosures/statement_text/<file>.txt   — cached extracted text
  data/disclosures/statement_pdfs/<file>.pdf   — source PDF (only with --keep-pdf)
  data/disclosures/statements-<date>.json      — parsed figures by ticker

Usage:
    from ingest.parse_statements import parse_statements
    data = parse_statements(tickers=["PRESCO"])

Or from the command line:
    python -m ingest.parse_statements --limit 5 --keep-pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

import requests
from pypdf import PdfReader

from ingest.fetch_disclosures import fetch_financial_statements

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DISCLOSURES_DIR = BASE_DIR / "data" / "disclosures"
STATEMENT_PDF_DIR = DISCLOSURES_DIR / "statement_pdfs"
STATEMENT_TEXT_DIR = DISCLOSURES_DIR / "statement_text"
for _d in (STATEMENT_PDF_DIR, STATEMENT_TEXT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
}
REQUEST_TIMEOUT = (10, 120)  # statements can be 10MB+
POLITE_DELAY = 0.5

# Only comma-grouped numbers count as values — this skips note-reference
# integers ("7", "11.3") that sit between a label and its figures.
_NUMBER = re.compile(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?")

# Profit-or-loss line labels (anchored to the start of a stripped line).
PNL_LABELS: list[tuple[str, re.Pattern]] = [
    ("revenue", re.compile(
        r"^(?:gross\s+earnings|gross\s+premium\s+(?:income|written)|"
        r"insurance\s+revenue|turnover|revenue)\b")),
    ("pbt", re.compile(
        r"^(?:\(?loss\)?\s*/?\s*)?profit\b.*\bbefore\s+(?:income\s+)?tax")),
    ("pat", re.compile(
        r"^(?:\(?loss\)?\s*/?\s*)?profit\b.*(?:for\s+the\s+(?:year|period)|"
        r"after\s+tax(?:ation)?)")),
]

# Statement-of-financial-position labels. "total equity" must not match
# "total equity and liabilities" (which is the asset-side total).
POS_LABELS: list[tuple[str, re.Pattern]] = [
    ("total_assets", re.compile(r"^total\s+assets\b")),
    ("total_equity", re.compile(r"^total\s+equity\b(?!\s+and\s+liab)")),
]


def _to_number(token: str) -> Optional[float]:
    negative = "(" in token
    digits = re.sub(r"[^\d.]", "", token)
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def _cache_stem(pdf_url: str) -> str:
    name = unquote(urlsplit(pdf_url).path.rsplit("/", 1)[-1]) or "statement.pdf"
    return name.rsplit(".", 1)[0]


def _get_text(pdf_url: str, keep_pdf: bool) -> list[str]:
    """Return the statement's text as a list of page strings.

    Uses a cached .txt (pages joined by form-feed) when available.
    """
    stem = _cache_stem(pdf_url)
    txt_path = STATEMENT_TEXT_DIR / f"{stem}.txt"
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path.read_text(encoding="utf-8", errors="ignore").split("\f")

    pdf_path = STATEMENT_PDF_DIR / f"{stem}.pdf"
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("[statements] download failed %s: %s", pdf_url, exc)
        return []
    pdf_path.write_bytes(resp.content)
    time.sleep(POLITE_DELAY)

    try:
        reader = PdfReader(str(pdf_path))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception as exc:
        logger.warning("[statements] could not read %s: %s", pdf_path.name, exc)
        pages = []
    finally:
        if not keep_pdf:
            pdf_path.unlink(missing_ok=True)

    if any(pages):
        txt_path.write_text("\f".join(pages), encoding="utf-8")
    return pages


def _grab(text: str, labels: list[tuple[str, re.Pattern]]) -> dict:
    """Pull the first/second comma-grouped number for each label found.

    Returns {metric: value, metric+'_prior': value} — the leading two
    numbers on a line are the current and prior reporting period (the
    consolidated/"Group" pair in a four-column statement).
    """
    found: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        for metric, pattern in labels:
            if metric in found:
                continue
            m = pattern.match(low)
            if not m:
                continue
            nums = [_to_number(t.group()) for t in _NUMBER.finditer(line, m.end())]
            nums = [n for n in nums if n is not None]
            if nums:
                found[metric] = nums[0]
                if len(nums) > 1:
                    found[f"{metric}_prior"] = nums[1]
    return found


def _pick_page(pages: list[str], labels: list[tuple[str, re.Pattern]],
               required: tuple[str, ...]) -> dict:
    """Parse the single page that best matches `labels`.

    Returns the first page that yields every metric in `required` (a real
    statement page); otherwise the page that yielded the most metrics.
    Parsing one focused page — instead of the whole document — stops stray
    label mentions elsewhere in the report from being picked up.
    """
    best: dict = {}
    for page in pages:
        grabbed = _grab(page, labels)
        if all(k in grabbed for k in required):
            return grabbed
        if len(grabbed) > len(best):
            best = grabbed
    return best


def _validate(record: dict) -> None:
    """Flag a record 'suspect' when figures fail basic financial sanity."""
    rev = record.get("revenue")
    pbt = record.get("pbt")
    pat = record.get("pat")
    assets = record.get("total_assets")
    equity = record.get("total_equity")
    profit_ceiling = 5_000_000_000     # ₦5tn in N'000
    balance_ceiling = 100_000_000_000  # ₦100tn in N'000

    issues: list[str] = []
    if pbt is not None and pat is not None and pbt > 0 and pat > pbt * 1.05:
        issues.append("PAT exceeds PBT")
    if rev is not None and pbt is not None and rev > 0 and pbt > rev:
        issues.append("PBT exceeds revenue")
    if rev is not None and rev <= 0:
        issues.append("non-positive revenue")
    if pbt is not None and abs(pbt) > profit_ceiling:
        issues.append("PBT magnitude implausible")
    if pat is not None and abs(pat) > profit_ceiling:
        issues.append("PAT magnitude implausible")
    if assets is not None and equity is not None and equity > assets * 1.02:
        issues.append("equity exceeds assets")
    if assets is not None and abs(assets) > balance_ceiling:
        issues.append("assets magnitude implausible")

    if issues:
        record["parse_status"] = "suspect"
        record["parse_issues"] = issues


def _pct_change(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None or prior is None or prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 1)


def parse_statements(
    tickers: Optional[list[str]] = None,
    since: str = "2019-01-31",
    use_cache: bool = True,
    limit: Optional[int] = None,
    keep_pdf: bool = False,
) -> dict[str, dict]:
    """
    Parse the latest financial statement for each ticker.

    Returns {ticker: statement dict} with current and prior-period revenue,
    pbt, pat, total_assets, total_equity, derived growth %, fiscal_year,
    filed_date, parse_status ("ok" | "partial" | "suspect" | "unparseable").
    """
    rows = fetch_financial_statements(tickers=tickers, since=since,
                                      include_forecasts=False, use_cache=use_cache)
    rows = [r for r in rows if "AUDITED" in r["title"].upper()
            or "FINANCIAL STATEMENT" in r["title"].upper()]
    latest: dict[str, dict] = {}
    for r in rows:  # rows are newest-first
        latest.setdefault(r["ticker"], r)

    items = list(latest.items())
    if limit:
        items = items[:limit]

    results: dict[str, dict] = {}
    for ticker, row in items:
        record = {
            "ticker": ticker,
            "fiscal_year": _fiscal_year(row["title"]),
            "filed_date": row["created"][:10],
            "revenue": None, "revenue_prior": None,
            "pbt": None, "pbt_prior": None,
            "pat": None, "pat_prior": None,
            "total_assets": None, "total_equity": None,
            "revenue_growth": None, "pat_growth": None,
            "unit": "NGN'000",
            "parse_status": "unparseable",
            "source_pdf": row["pdf_url"],
        }
        pages = _get_text(row["pdf_url"], keep_pdf)
        if pages:
            figures = _pick_page(pages, PNL_LABELS, ("revenue", "pbt", "pat"))
            figures.update(_pick_page(pages, POS_LABELS,
                                      ("total_assets", "total_equity")))
            for key, value in figures.items():
                record[key] = value
            # The in-PDF prior-period column is layout-dependent. Drop a prior
            # value that exactly duplicates the current figure (a repeated
            # Group/Company column) before deriving growth.
            for metric in ("revenue", "pbt", "pat"):
                if record.get(f"{metric}_prior") == record.get(metric):
                    record[f"{metric}_prior"] = None
            record["revenue_growth"] = _pct_change(record["revenue"],
                                                   record["revenue_prior"])
            record["pat_growth"] = _pct_change(record["pat"], record["pat_prior"])
            # An implausible YoY move means the prior column was mis-aligned;
            # discard it (the current-period figure is kept and validated).
            for metric, gkey in (("revenue", "revenue_growth"),
                                 ("pat", "pat_growth")):
                if record[gkey] is not None and abs(record[gkey]) > 250:
                    record[gkey] = None
                    record[f"{metric}_prior"] = None
            core = [record["revenue"], record["pbt"], record["pat"]]
            if all(v is not None for v in core):
                record["parse_status"] = "ok"
            elif any(v is not None for v in core):
                record["parse_status"] = "partial"
            _validate(record)
        results[ticker] = record
        logger.info("[statements] %-12s %-12s rev=%s pbt=%s pat=%s",
                    ticker, record["parse_status"],
                    record["revenue"], record["pbt"], record["pat"])

    _save(results)
    return results


def _fiscal_year(title: str) -> Optional[int]:
    m = re.search(r"\b(20\d\d)\b", title)
    return int(m.group(1)) if m else None


def _save(results: dict[str, dict]) -> None:
    out = DISCLOSURES_DIR / f"statements-{datetime.now():%Y-%m-%d}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    counts = dict(Counter(r["parse_status"] for r in results.values()))
    logger.info("[statements] saved %d statements %s -> %s",
                len(results), counts, out)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse NGX financial statements.")
    parser.add_argument("--ticker", action="append", dest="tickers",
                        help="restrict to ticker (repeatable)")
    parser.add_argument("--since", default="2019-01-31", help="earliest date")
    parser.add_argument("--limit", type=int, default=None, help="max tickers")
    parser.add_argument("--keep-pdf", action="store_true",
                        help="keep downloaded PDFs (default: text only)")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the disclosure cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = parse_statements(tickers=args.tickers, since=args.since,
                            use_cache=not args.no_cache, limit=args.limit,
                            keep_pdf=args.keep_pdf)
    print(f"\n{len(data)} statements\n" + "-" * 78)
    for t, r in sorted(data.items()):
        print(f"{t:<13} {r['parse_status']:<12} FY{r['fiscal_year']}  "
              f"rev={r['revenue']}  pbt={r['pbt']}  pat={r['pat']}  "
              f"rev_grw={r['revenue_growth']}  pat_grw={r['pat_growth']}")


__all__ = ["parse_statements"]


if __name__ == "__main__":
    _main()

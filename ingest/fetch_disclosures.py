"""
NGX Corporate Disclosures Scraper
==================================
Pulls corporate disclosures (results, dividend declarations, AGM notices,
board changes, etc.) for NGX-listed companies from the Nigerian Exchange
document library.

The disclosures listed on
    https://ngxgroup.com/exchange/data/corporate-disclosures/
are not embedded in that WordPress page — the page loads them client-side
from a public SharePoint REST list, 'XFinancial_News', hosted at
doclib.ngxgroup.com. We query that list directly (no browser needed),
paginating via the OData ``__next`` skiptoken.

Each disclosure record carries:
  ticker     — NGX symbol (SharePoint CompanySymbol)
  company    — full company name
  title      — disclosure description
  type       — raw Type_of_Submission, as filed (inconsistent casing/spelling)
  category   — normalised bucket derived from `type` (see CATEGORIES below)
  pdf_url    — link to the disclosure PDF
  created    — submission timestamp (UTC, ISO 8601)
  modified   — last-modified timestamp (UTC, ISO 8601)

The same list backs the "Financials Statements" tab on each company's NGX
page — those are simply rows whose category is "financial_statement". The
raw `type` field is filed inconsistently ("Financial Statements", "FINANCIAL
STATEMENTS", "Financial Statement", "EarningForcast", ...), so always filter
on `category`, never on `type`.

Artefacts:
  data/disclosures/<date>.json   — full list fetched on <date>

Usage:
    from ingest.fetch_disclosures import DisclosureScraper, fetch_financial_statements
    scraper = DisclosureScraper()
    rows = scraper.fetch_all()                          # everything since 2019
    rows = scraper.fetch_all(tickers=["DANGCEM"])       # filter to ticker(s)
    rows = scraper.fetch_all(since="2026-01-01")        # only recent disclosures
    fins = fetch_financial_statements()                 # all financial statements
    fins = fetch_financial_statements(tickers=["GTCO"]) # one company's filings

Or from the command line:
    python -m ingest.fetch_disclosures --since 2026-01-01 --ticker DANGCEM
    python -m ingest.fetch_disclosures --category financial_statement
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DISCLOSURES_DIR = BASE_DIR / "data" / "disclosures"
DISCLOSURES_DIR.mkdir(parents=True, exist_ok=True)

# ── NGX document library (SharePoint REST) configuration ─────────────

DOCLIB_ORIGIN = "https://doclib.ngxgroup.com"
LIST_TITLE = "XFinancial_News"
LIST_ENDPOINT = f"{DOCLIB_ORIGIN}/_api/Web/Lists/GetByTitle('{LIST_TITLE}')/items/"

# Fields the corporate-disclosures page itself selects.
SELECT_FIELDS = "URL,Modified,Created,CompanyName,CompanySymbol,Type_of_Submission"

# The live page only goes back to 2019; keep the same floor as a default.
DEFAULT_SINCE = "2019-01-31"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    # SharePoint REST requires this exact Accept to return JSON, not XML.
    "Accept": "application/json;odata=verbose",
    "Referer": "https://ngxgroup.com/exchange/data/corporate-disclosures/",
}

REQUEST_TIMEOUT = (10, 30)  # (connect, read) seconds
POLITE_DELAY = 0.5          # seconds between paged requests
PAGE_SIZE = 1000            # OData $top per page


# ── disclosure-type normalisation ────────────────────────────────────
# Issuers file Type_of_Submission free-hand, so the raw values vary wildly
# in casing, spelling and even combine multiple types. `_categorise` folds
# them into these stable buckets; downstream code should filter on those.

CATEGORIES = (
    "financial_statement",
    "earnings_forecast",
    "corporate_action",
    "directors_dealings",
    "agm",
    "egm",
    "board_meeting",
    "court_order_meeting",
    "other",
)


def _categorise(raw_type: str) -> str:
    """Map a free-text Type_of_Submission onto a stable CATEGORIES bucket."""
    t = raw_type.lower()
    # Order matters: a combined filing like "Corporate Actions Financial
    # Statements" should count as a financial statement.
    if "financial statement" in t:
        return "financial_statement"
    if "forcast" in t or "forecast" in t:          # "EarningForcast", "Earnings Forecast"
        return "earnings_forecast"
    if "director" in t and "dealing" in t:         # "DirectorsDealings", "Directors Dealings"
        return "directors_dealings"
    if "extra-ordinary" in t or "egm" in t:
        return "egm"
    if "court order" in t or "(com)" in t:
        return "court_order_meeting"
    if "annual general meeting" in t or "agm" in t:
        return "agm"
    if "board meeting" in t or "(bm)" in t or "(cbm)" in t:
        return "board_meeting"
    if "corporate action" in t or "corprorate action" in t:
        return "corporate_action"
    return "other"


def _iso_utc(date_str: str) -> str:
    """Normalise a YYYY-MM-DD (or full ISO) string to a UTC SharePoint literal."""
    date_str = date_str.strip()
    if "T" in date_str:
        return date_str if date_str.endswith("Z") else date_str + "Z"
    return f"{date_str}T00:00:00.000Z"


def _cache_path(date_str: Optional[str] = None) -> Path:
    ds = date_str or datetime.now().strftime("%Y-%m-%d")
    return DISCLOSURES_DIR / f"{ds}.json"


class DisclosureScraper:
    """Fetches NGX corporate disclosures from the doclib SharePoint list."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── public API ───────────────────────────────────────────────────

    def fetch_all(
        self,
        tickers: Optional[list[str]] = None,
        since: str = DEFAULT_SINCE,
        until: Optional[str] = None,
        categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Return disclosure records, newest first.

        Parameters
        ----------
        tickers    : restrict to these NGX symbols (case-insensitive); None = all
        since      : earliest disclosure date (YYYY-MM-DD)
        until      : latest disclosure date (YYYY-MM-DD)
        categories : restrict to these normalised buckets (see CATEGORIES);
                     None = all
        """
        rows = self._load_cache() if self.use_cache else None
        if rows is None:
            rows = self._fetch_from_api()
            self._save_cache(rows)
        else:
            logger.info("[disclosures] using cached %s (%d rows)", _cache_path(), len(rows))

        # `since` is always a post-filter: the daily cache holds the full
        # history (DEFAULT_SINCE), so it stays a complete superset no matter
        # what date range an individual call asks for.
        since_iso = _iso_utc(since)
        rows = [r for r in rows if r["created"] >= since_iso]

        if tickers:
            wanted = {t.upper() for t in tickers}
            rows = [r for r in rows if r["ticker"] in wanted]
        if until:
            until_iso = _iso_utc(until)
            rows = [r for r in rows if r["created"] <= until_iso]
        if categories:
            cats = set(categories)
            rows = [r for r in rows if r["category"] in cats]

        rows.sort(key=lambda r: r["created"], reverse=True)
        return rows

    # ── internals ────────────────────────────────────────────────────

    def _fetch_from_api(self) -> list[dict]:
        """Walk the SharePoint list page by page via the OData skiptoken.

        Always pulls the full history from DEFAULT_SINCE so the cached file
        is a complete superset; callers narrow the range via `fetch_all`.
        """
        params = {
            "$select": SELECT_FIELDS,
            "$orderby": "Created desc",
            "$filter": f"Created ge '{_iso_utc(DEFAULT_SINCE)}'",
            "$top": PAGE_SIZE,
        }
        url: Optional[str] = LIST_ENDPOINT
        rows: list[dict] = []
        page = 0

        while url:
            page += 1
            resp = self.session.get(
                url,
                params=params if page == 1 else None,  # __next carries its own query
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json().get("d", {})
            results = payload.get("results", [])
            rows.extend(self._parse(item) for item in results)
            logger.info("[disclosures] page %d: +%d rows (%d total)", page, len(results), len(rows))

            url = payload.get("__next")
            if url:
                time.sleep(POLITE_DELAY)

        return rows

    @staticmethod
    def _parse(item: dict) -> dict:
        """Flatten one raw SharePoint list item into a plain disclosure record."""
        url_field = item.get("URL") or {}
        symbol = (item.get("CompanySymbol") or "").strip().upper()
        raw_type = (item.get("Type_of_Submission") or "").strip()
        return {
            "ticker": symbol,
            "company": (item.get("CompanyName") or "").strip(),
            "title": (url_field.get("Description") or "").strip(),
            "type": raw_type,
            "category": _categorise(raw_type),
            "pdf_url": (url_field.get("Url") or "").strip(),
            "created": item.get("Created") or "",
            "modified": item.get("Modified") or "",
        }

    # ── cache ────────────────────────────────────────────────────────

    def _load_cache(self) -> Optional[list[dict]]:
        p = _cache_path()
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                rows = json.load(f)
            # Backfill `category` for caches written before that field existed.
            for r in rows:
                r.setdefault("category", _categorise(r.get("type", "")))
            return rows
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[disclosures] ignoring unreadable cache %s: %s", p, exc)
            return None

    def _save_cache(self, rows: list[dict]) -> None:
        p = _cache_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        logger.info("[disclosures] cache saved -> %s (%d rows)", p, len(rows))


def fetch_disclosures(
    tickers: Optional[list[str]] = None,
    since: str = DEFAULT_SINCE,
    until: Optional[str] = None,
    categories: Optional[list[str]] = None,
    use_cache: bool = True,
) -> list[dict]:
    """Convenience wrapper mirroring ingest.fetch_reports.fetch_all_reports."""
    return DisclosureScraper(use_cache=use_cache).fetch_all(
        tickers, since, until, categories
    )


def fetch_financial_statements(
    tickers: Optional[list[str]] = None,
    since: str = DEFAULT_SINCE,
    until: Optional[str] = None,
    include_forecasts: bool = True,
    use_cache: bool = True,
) -> list[dict]:
    """
    Return only financial-statement filings (audited, year-end and quarterly).

    This is the data behind the "Financials Statements" tab on each company's
    NGX page. With ``include_forecasts`` (default), issuer earnings forecasts
    are included too, since the NGX tab lists them alongside the statements.
    """
    cats = ["financial_statement"]
    if include_forecasts:
        cats.append("earnings_forecast")
    return fetch_disclosures(tickers, since, until, cats, use_cache)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NGX corporate disclosures.")
    parser.add_argument("--ticker", action="append", dest="tickers",
                        help="restrict to ticker (repeatable)")
    parser.add_argument("--since", default=DEFAULT_SINCE, help="earliest date YYYY-MM-DD")
    parser.add_argument("--until", default=None, help="latest date YYYY-MM-DD")
    parser.add_argument("--category", action="append", dest="categories",
                        choices=CATEGORIES, help="restrict to category (repeatable)")
    parser.add_argument("--financials", action="store_true",
                        help="shorthand for --category financial_statement "
                             "--category earnings_forecast")
    parser.add_argument("--no-cache", action="store_true", help="bypass today's cache")
    parser.add_argument("--limit", type=int, default=20, help="rows to print")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    categories = args.categories
    if args.financials:
        categories = (categories or []) + ["financial_statement", "earnings_forecast"]
    rows = fetch_disclosures(
        tickers=args.tickers,
        since=args.since,
        until=args.until,
        categories=categories,
        use_cache=not args.no_cache,
    )
    print(f"\n{len(rows)} disclosures\n" + "-" * 78)
    for r in rows[: args.limit]:
        print(f"{r['created'][:10]}  {r['ticker']:<12} {r['category']:<20} {r['title']}")


__all__ = [
    "DisclosureScraper",
    "fetch_disclosures",
    "fetch_financial_statements",
    "CATEGORIES",
]


if __name__ == "__main__":
    _main()

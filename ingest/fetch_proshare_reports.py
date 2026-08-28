"""
ProShare Report Scraper
========================
Scrapes the ProShare annual-report listing at
https://proshare.co/articles/list?menu=Reports&category=Annual%20Reports
and maps each matching article to an NGX ticker.

Two-stage strategy
------------------
1. **Listing pass (cloudscraper):** fast — pulls the index pages, parses
   article slugs into (company, year, report_type, title), matches to
   NGX tickers via snapshot descriptions.
2. **Detail pass (Playwright, optional):** slower — for each matched
   article, navigates the JS-rendered detail page and extracts:
       • published date
       • author / source
       • view count
       • **direct PDF download URL** (hosted on S3)

The article body itself is *not* rendered HTML on ProShare — annual-report
posts are metadata shells linking to an external PDF. Getting the actual
audited-statement numbers requires downloading that PDF and running text
extraction, which is out of scope here; we capture the URL so a later
pass can fetch it.

If Playwright isn't available or a detail fetch times out, we degrade
gracefully to the listing-only record (URL + title + slug-parsed year).

Usage
-----
    from ingest.fetch_proshare_reports import ProShareScraper
    scraper = ProShareScraper()
    reports = scraper.fetch_all(["ACCESSCORP", "WEMABANK"])
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cloudscraper
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── ProShare configuration ──────────────────────────────────────────

PROSHARE_BASE = "https://proshare.co"
PROSHARE_LIST_URL = (
    f"{PROSHARE_BASE}/articles/list"
    "?menu=Reports&category=Annual%20Reports&pageId={page}"
)

REQUEST_TIMEOUT = (10, 20)
POLITE_DELAY = 1.5
MAX_PAGES = 30  # safety bound; ~20 articles/page => 600 articles ceiling

# Playwright tuning — Cloudflare's interstitial can take ~5-8s to clear
PW_NAV_TIMEOUT_MS = 45_000
PW_RENDER_WAIT_MS = 8_000
PW_CHALLENGE_EXTRA_WAIT_MS = 8_000

# Stopwords removed from company names before slug matching
_STOPWORDS = {
    "plc", "nig", "nigeria", "nigerian", "ng",
    "holdings", "holding", "group", "company", "co", "corporation", "corp",
    "limited", "ltd", "inc", "incorporated",
    "the", "of", "and", "for",
}

# Report-type tokens found in ProShare slugs
_REPORT_TYPE_PATTERNS = [
    (re.compile(r"audited[- ]financial[- ]statement", re.I), "ar"),
    (re.compile(r"annual[- ]report", re.I),                  "ar"),
    (re.compile(r"unaudited", re.I),                         "ir"),
    (re.compile(r"quarterly", re.I),                         "ir"),
    (re.compile(r"half[- ]year|h1|interim", re.I),           "ir"),
]

DOC_TYPE_LABELS = {
    "ar": "Annual Report",
    "ir": "Interim Report",
    "ab": "Abridged Report",
    "pr": "Presentation",
}

# Regex for the ProShare meta-header line:
#   "Apr 07, 2026   •   by Wema Bank   •   Source: NGX Group   •    146 views"
_META_HEADER_RX = re.compile(
    r"(?P<date>[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})"
    r"\s*[•·]\s*"
    r"by\s+(?P<author>[^•·|]+?)"
    r"\s*[•·]\s*"
    r"Source:\s*(?P<source>[^•·|]+?)"
    r"(?:\s*[•·]\s*(?P<views>\d+)\s*views?)?",
    re.I,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    """Normalize free text into a comparable token set (lowercased, no stopwords)."""
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", text).lower()
    return {t for t in text.split() if t and t not in _STOPWORDS and len(t) > 1}


def _parse_article_slug(href: str, title: str) -> Optional[dict]:
    """
    Extract (company_tokens, year, doc_type, period_end, url, title) from a
    ProShare article href + title. Returns None if the slug doesn't look
    like a dated report article.
    """
    # href looks like: /articles/{slug}?menu=...
    path = href.split("?", 1)[0].lstrip("/")
    if not path.startswith("articles/"):
        return None
    slug = path[len("articles/"):]

    # Extract year — slug has `-fy-YYYY-` or `-YYYY-`; prefer `-fy-YYYY-`
    year = None
    m = re.search(r"-fy-(\d{4})-", slug)
    if m:
        year = int(m.group(1))
    else:
        m = re.search(r"-(\d{4})-", slug)
        if m:
            y = int(m.group(1))
            if 2000 <= y <= datetime.now().year:
                year = y
    if not year:
        return None

    # Doc type from slug
    doc_type = "ar"  # default — listing is filtered to Annual Reports
    for pat, dt in _REPORT_TYPE_PATTERNS:
        if pat.search(slug):
            doc_type = dt
            break

    # Company portion = slug before `-fy-` or before the year
    company_part = re.split(r"-fy-\d{4}|-\d{4}-", slug, maxsplit=1)[0]
    company_tokens = _tokens(company_part.replace("-", " "))
    if not company_tokens:
        return None

    # Period-end date from title (e.g. "31st December 2025")
    period_end = ""
    m = re.search(
        r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})",
        title,
        flags=re.I,
    )
    if m:
        period_end = m.group(1)

    url = f"{PROSHARE_BASE}/{path}"
    return {
        "slug": slug,
        "url": url,
        "title": title.strip(),
        "year": year,
        "doc_type": doc_type,
        "company_tokens": company_tokens,
        "period_end": period_end,
    }


def _build_ticker_tokens(descriptions: dict[str, str]) -> dict[str, set[str]]:
    """Map each ticker → canonical token set from its company description."""
    out: dict[str, set[str]] = {}
    for ticker, desc in descriptions.items():
        toks = _tokens(desc or "")
        if not toks:
            toks = {ticker.lower()}
        out[ticker.upper()] = toks
    return out


def _match_article_to_ticker(
    article_tokens: set[str],
    ticker_tokens: dict[str, set[str]],
) -> Optional[tuple[str, float]]:
    """
    Return the (ticker, score) with the strongest match, or None.
    Requires *all* of the ticker's identifying tokens to appear in the
    article — prevents "Dangote Cement" matching a "Dangote Sugar" post.
    Ties broken by token-set size (more specific ticker wins).
    """
    best: Optional[tuple[str, float]] = None
    for ticker, toks in ticker_tokens.items():
        if not toks:
            continue
        overlap = toks & article_tokens
        score = len(overlap) / len(toks)
        if score < 1.0:
            continue
        if best is None or len(toks) > len(ticker_tokens[best[0]]):
            best = (ticker, score)
    return best


def _cache_path(date_str: Optional[str] = None) -> Path:
    ds = date_str or datetime.now().strftime("%Y-%m-%d")
    return REPORTS_DIR / f"proshare-{ds}.json"


def _load_cache(date_str: Optional[str] = None) -> dict:
    p = _cache_path(date_str)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(data: dict, date_str: Optional[str] = None) -> None:
    p = _cache_path(date_str)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("ProShare cache saved -> %s", p)


def _load_snapshot_descriptions() -> dict[str, str]:
    """Load ticker → description from the latest parquet snapshot."""
    try:
        import pandas as pd
        data_dir = BASE_DIR / "data" / "snapshots"
        latest = sorted(data_dir.glob("*/snapshot.parquet"))[-1]
        df = pd.read_parquet(latest)
        if "description" not in df.columns:
            return {}
        return dict(zip(df["symbol"].astype(str), df["description"].astype(str)))
    except Exception as e:  # pragma: no cover
        logger.warning("Could not load snapshot descriptions: %s", e)
        return {}


# ── Scraper ─────────────────────────────────────────────────────────

class ProShareScraper:
    """
    Scrapes ProShare's Annual Reports listing for Nigerian-listed companies.

    Listing metadata (URL, fiscal year, report type, company) is parsed
    from the cloudscraper-fetched index pages. When Playwright is
    available, each matched article's detail page is opened to harvest
    the published date, author, source, view count, and the S3 PDF URL.
    """

    def __init__(
        self,
        use_cache: bool = True,
        use_playwright: bool = True,
    ):
        self.session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True},
        )
        self.use_cache = use_cache
        # Can be disabled per-call or globally; if the import fails we
        # degrade silently to metadata-only.
        self.use_playwright = use_playwright and self._playwright_available()

    @staticmethod
    def _playwright_available() -> bool:
        try:
            import playwright.sync_api  # noqa: F401
            return True
        except ImportError:
            logger.info("Playwright not installed; running metadata-only")
            return False

    # ── Public API ──────────────────────────────────────────────────

    def fetch_all(
        self,
        tickers: list[str],
        year: Optional[int] = None,
        max_pages: int = MAX_PAGES,
    ) -> dict[str, dict]:
        """
        Return {ticker: report_dict} for requested tickers with matching
        ProShare annual-report articles. Report dict keys:
           title, url, doc_type, doc_type_label, year, period, published,
           summary, source, pdf_url, author, views, body_available
        """
        today = datetime.now().strftime("%Y-%m-%d")
        requested = {t.upper() for t in tickers}

        cache = _load_cache(today) if self.use_cache else {}
        if cache and requested.issubset(cache.keys()):
            logger.info("ProShare: all %d tickers served from cache", len(requested))
            return {t: cache[t] for t in requested}

        descriptions = _load_snapshot_descriptions()
        descriptions = {t: d for t, d in descriptions.items() if t.upper() in requested}
        ticker_tokens = _build_ticker_tokens(descriptions)

        # Stage 1: paginate listing → match → collect article records
        matched: dict[str, dict] = {}
        seen_slugs: set[str] = set()

        for page in range(1, max_pages + 1):
            url = PROSHARE_LIST_URL.format(page=page)
            logger.info("ProShare: fetching listing page %d", page)
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except Exception as e:
                logger.warning("ProShare: page %d failed (%s) — stopping pagination", page, e)
                break

            articles = self._parse_listing(resp.text)
            new_articles = [a for a in articles if a["slug"] not in seen_slugs]
            if not new_articles:
                logger.info("ProShare: page %d had no new articles — stopping", page)
                break
            for a in new_articles:
                seen_slugs.add(a["slug"])

            for article in new_articles:
                if year and article["year"] != year:
                    continue
                m = _match_article_to_ticker(article["company_tokens"], ticker_tokens)
                if not m:
                    continue
                ticker, _score = m
                existing = matched.get(ticker)
                if existing and existing["year"] >= article["year"]:
                    continue
                matched[ticker] = article  # stash article for stage 2

            if requested.issubset(matched.keys()):
                logger.info("ProShare: all requested tickers matched by page %d", page)
                break

            time.sleep(POLITE_DELAY)

        # Stage 2: detail-page enrichment (optional, Playwright)
        enriched: dict[str, dict] = {}
        if self.use_playwright and matched:
            logger.info(
                "ProShare: opening %d detail pages with Playwright", len(matched),
            )
            details = self._enrich_with_playwright(matched)
        else:
            details = {t: {} for t in matched}

        for ticker, article in matched.items():
            enriched[ticker] = self._article_to_report(article, ticker, details.get(ticker, {}))

        cache.update(enriched)
        _save_cache(cache, today)

        return {t: enriched[t] for t in enriched if t in requested}

    # ── Listing parser ──────────────────────────────────────────────

    def _parse_listing(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        seen_slugs: set[str] = set()
        articles: list[dict] = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("/articles/") or href.startswith("/articles/list"):
                continue
            title = a.get_text(strip=True)
            parsed = _parse_article_slug(href, title)
            if not parsed or parsed["slug"] in seen_slugs:
                continue
            # Prefer occurrence with a real title
            if not title:
                continue
            seen_slugs.add(parsed["slug"])
            articles.append(parsed)
        return articles

    # ── Detail-page enrichment ──────────────────────────────────────

    def _enrich_with_playwright(
        self, matched: dict[str, dict]
    ) -> dict[str, dict]:
        """
        Visit each matched article's detail page with a headless Chromium
        to extract the PDF URL + metadata. Returns {ticker: details_dict};
        missing keys mean the enrichment failed.

        Cloudflare challenges every navigation after the first in the same
        context, and waiting/reloading does not clear it. Using a fresh
        context per article lets each hit land clean. Cost: ~8-10s per
        article, so this is the slow path — results are cached.
        """
        from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

        out: dict[str, dict] = {}
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )
            try:
                for i, (ticker, article) in enumerate(matched.items(), 1):
                    logger.info(
                        "  [%d/%d] ProShare detail: %s", i, len(matched), ticker,
                    )
                    ctx = browser.new_context(
                        user_agent=(
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/125.0.0.0 Safari/537.36"
                        ),
                        viewport={"width": 1280, "height": 800},
                    )
                    ctx.add_init_script(
                        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                    )
                    page = ctx.new_page()
                    try:
                        out[ticker] = self._fetch_detail(page, article["url"])
                    except PwTimeout:
                        logger.warning("  [timeout] %s — keeping listing metadata", ticker)
                        out[ticker] = {}
                    except Exception as e:
                        logger.warning("  [err] %s: %s — keeping listing metadata", ticker, e)
                        out[ticker] = {}
                    finally:
                        ctx.close()
            finally:
                browser.close()
        return out

    @staticmethod
    def _fetch_detail(page, url: str) -> dict:
        """Navigate `page` to `url` and extract PDF + metadata."""
        page.goto(url, timeout=PW_NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        page.wait_for_timeout(PW_RENDER_WAIT_MS)

        # If Cloudflare interstitial is still showing, wait longer
        body_sample = page.locator("body").inner_text()[:300].lower()
        if "security verification" in body_sample or "checking your browser" in body_sample:
            page.wait_for_timeout(PW_CHALLENGE_EXTRA_WAIT_MS)

        # Pull out the Download link (S3 PDF) and meta-header line in one pass
        data = page.evaluate(
            """() => {
              const pick = (sel) => Array.from(document.querySelectorAll(sel));
              // Find the Download anchor
              let pdf_url = '';
              for (const a of pick('a')) {
                const txt = (a.innerText || '').trim().toLowerCase();
                const href = a.href || '';
                if (txt === 'download' && href) { pdf_url = href; break; }
              }
              // Find the meta-header (date • by X • Source: Y • N views)
              let meta = '';
              for (const el of pick('*')) {
                const t = (el.innerText || '').trim();
                if (t.length < 300 && /by .+Source:/i.test(t)) { meta = t; break; }
              }
              return { pdf_url, meta };
            }"""
        )

        details: dict = {"pdf_url": data.get("pdf_url", "") or ""}
        meta = (data.get("meta") or "").replace("\n", " ")
        m = _META_HEADER_RX.search(meta)
        if m:
            details["published"] = m.group("date").strip()
            details["author"] = m.group("author").strip()
            details["source_attribution"] = m.group("source").strip()
            views = m.group("views")
            if views:
                details["views"] = int(views)
        return details

    # ── Record shaping ──────────────────────────────────────────────

    def _article_to_report(
        self, article: dict, ticker: str, details: dict
    ) -> dict:
        doc_type = article["doc_type"]
        pdf_url = details.get("pdf_url", "")
        published = details.get("published") or article.get("period_end", "")

        summary_parts = [article["title"]]
        if details.get("author"):
            src = details.get("source_attribution", "")
            attrib = f"by {details['author']}"
            if src:
                attrib += f" — Source: {src}"
            summary_parts.append(attrib)
        if pdf_url:
            summary_parts.append(f"Audited statement PDF: {pdf_url}")
        else:
            summary_parts.append(
                "(Listing-only record — detail-page enrichment skipped or failed. "
                "Install Playwright and re-run to capture the PDF URL.)"
            )
        summary = "\n\n".join(summary_parts)

        return {
            "ticker": ticker,
            "title": article["title"],
            "url": article["url"],
            "doc_type": doc_type,
            "doc_type_label": DOC_TYPE_LABELS.get(doc_type, doc_type),
            "year": article["year"],
            "period": "00",
            "published": published,
            "summary": summary,
            "source": "proshare",
            "pdf_url": pdf_url,
            "author": details.get("author", ""),
            "source_attribution": details.get("source_attribution", ""),
            "views": details.get("views"),
            "body_available": False,  # ProShare articles have no rendered body
            "pdf_available": bool(pdf_url),
        }


__all__ = ["ProShareScraper"]

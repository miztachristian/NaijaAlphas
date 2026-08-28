"""
NGX News Scraper  (v2 — Google News RSS)
==========================================
Fetches recent financial news headlines for Nigerian stocks from:
  1. Google News RSS — per-ticker targeted queries (PRIMARY source)
  2. Nairametrics / BusinessDay / ThisDay / Punch RSS (supplementary)

Stores results as JSON in data/news/<date>.json for caching.

Usage:
    from ingest.news_scraper import NewsScraper
    scraper = NewsScraper()
    headlines = scraper.fetch_all(["WEMABANK", "TIP", "CUSTODIAN"])
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
NEWS_DIR = BASE_DIR / "data" / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# Mapping of ticker symbols to company name search terms.
# First alias is the PRIMARY query used for Google News.
TICKER_ALIASES = {
    "TIP": ["initiates plc", "tip plc", "initiates"],
    "WEMABANK": ["wema bank", "wemabank", "wema"],
    "CUSTODIAN": ["custodian investment", "custodian plc", "custodian"],
    "STANBIC": ["stanbic ibtc", "stanbic"],
    "FCMB": ["fcmb", "first city monument"],
    "MBENEFIT": ["mutual benefits", "mutual benefit", "mbenefit"],
    "WAPCO": ["lafarge africa", "wapco", "lafarge"],
    "TRANSCORP": ["transcorp", "transnational corporation"],
    "NPFMCRFBK": ["npf microfinance", "npfmcrfbk", "npf micro"],
    "NASCON": ["nascon allied", "nascon"],
    "IKEJAHOTEL": ["ikeja hotel", "ikeja hotels"],
    "DANGCEM": ["dangote cement", "dangcem"],
    "GTCO": ["guaranty trust", "gtco"],
    "ZENITHBANK": ["zenith bank", "zenithbank"],
    "ACCESSCORP": ["access holdings", "access bank", "accesscorp"],
    "BUACEMENT": ["bua cement", "buacement"],
    "OKOMUOIL": ["okomu oil", "okomuoil"],
    "PRESCO": ["presco plc", "presco"],
    "INFINITY": ["infinity trust mortgage", "infinity mortgage"],
    "UBA": ["united bank for africa", "uba"],
    "FBNH": ["fbn holdings", "first bank", "fbnh"],
    "DANGSUGAR": ["dangote sugar", "dangsugar"],
    "SEPLAT": ["seplat energy", "seplat"],
    "MTNN": ["mtn nigeria", "mtnn"],
    "AIRTELAFRI": ["airtel africa", "airtelafri"],
    "GEREGU": ["geregu power", "geregu"],
    "BUAFOODS": ["bua foods", "buafoods"],
    "ETI": ["ecobank transnational", "ecobank", "eti"],
    "VITAFOAM": ["vitafoam nigeria", "vitafoam"],
    "STERLINGNG": ["sterling financial", "sterling bank", "sterlingng"],
    "CWG": ["cwg plc", "cwg"],
    "UNILEVER": ["unilever nigeria", "unilever"],
    "CAP": ["chemical and allied products", "cap plc"],
    "FLOURMILL": ["flour mills nigeria", "flourmill"],
    "INTBREW": ["international breweries", "intbrew"],
    "FIDELITYBK": ["fidelity bank", "fidelitybk"],
    "NAHCO": ["nahco aviance", "nahco"],
    "JAIZBANK": ["jaiz bank", "jaizbank"],
    "GUINNESS": ["guinness nigeria", "guinness"],
    "LIVESTOCK": ["livestock feeds", "livestock"],
    "LEARNAFRCA": ["learn africa", "learnafrca"],
    # ── Added batch — previously uncovered tickers ──
    "ABBEYBDS": ["abbey mortgage bank", "abbeybds"],
    "ABCTRANS": ["abc transport", "abctrans"],
    "AFRINSURE": ["african alliance insurance", "afrinsure"],
    "AFROMEDIA": ["afromedia plc", "afromedia"],
    "AVAIF": ["ava insurance", "ava nigeria", "avaif"],
    "CHELLARAM": ["chellarams plc", "chellaram"],
    "CILEASING": ["c&i leasing", "ci leasing", "cileasing"],
    "CONHALLPLC": ["consolidated hallmark", "conhallplc", "hallmark insurance"],
    "CORNERST": ["cornerstone insurance", "cornerst"],
    "EKOCORP": ["ekocorp plc", "eko corporation", "ekocorp"],
    "ENAMELWA": ["enamelware nigeria", "enamelwa"],
    "FTNCOCOA": ["ftn cocoa processors", "ftncocoa"],
    "GOLDBREW": ["goldenbrew nigeria", "goldbrew"],
    "GUINEAINS": ["guinea insurance", "guineains"],
    "HONYFLOUR": ["honeywell flour", "honeywell group", "honyflour"],
    "INTENEGINS": ["international energy insurance", "intenegins"],
    "JBERGER": ["julius berger nigeria", "julius berger", "jberger"],
    "JOHNHOLT": ["john holt plc", "johnholt"],
    "JULI": ["juli plc", "juli nigeria"],
    "LEGENDINT": ["legend internet", "legendint"],
    "LINKASSURE": ["linkage assurance", "linkassure"],
    "MAYBAKER": ["may and baker nigeria", "may & baker", "maybaker"],
    "MULTITREX": ["multitrex integrated foods", "multitrex"],
    "NIDF": ["nigeria infrastructure debt fund", "nidf"],
    "NNFM": ["northern nigeria flour mills", "nnfm"],
    "NREIT": ["updc real estate investment trust", "nreit"],
    "NSLTECH": ["nsl technology", "nsltech"],
    "PHARMDEKO": ["pharma-deko plc", "pharmdeko"],
    "PREMPAINTS": ["premium paints", "prempaints"],
    "REDSTAREX": ["red star express", "redstarex"],
    "REGALINS": ["regal insurance", "regalins"],
    "ROYALEX": ["royal exchange plc", "royal exchange nigeria", "royalex"],
    "SFSREIT": ["sfs real estate investment trust", "sfsreit"],
    "SKYAVN": ["skyway aviation handling", "skyavn", "sahco"],
    "SOVRENINS": ["sovereign trust insurance", "sovrenins"],
    "SUNUASSUR": ["sunu assurances", "sunuassur"],
    "THOMASWY": ["thomas wyatt nigeria", "thomaswy"],
    "TRANSCOHOT": ["transcorp hotels", "transcohot"],
    "TRANSEXPR": ["trans-nationwide express", "transexpr"],
    "TRIPPLEG": ["tripple gee", "trippleg"],
    "UNITYBNK": ["unity bank nigeria", "unitybnk", "unity bank"],
    "UPDCREIT": ["updc reit", "updcreit"],
    "VANLEER": ["vanleer plc", "vanleer"],
    "VFDGROUP": ["vfd group", "vfd microfinance", "vfdgroup"],
}

# Supplementary broad RSS feeds (Nigerian financial media)
RSS_FEEDS = [
    {
        "name": "Nairametrics",
        "url": "https://nairametrics.com/feed/",
    },
    {
        "name": "BusinessDay",
        "url": "https://businessday.ng/feed/",
    },
    {
        "name": "ThisDay",
        "url": "https://www.thisdaylive.com/index.php/category/business/feed/",
    },
    {
        "name": "Punch",
        "url": "https://punchng.com/topics/business/feed/",
    },
]

# Google News RSS template — per-ticker targeted queries
# hl=en-NG, gl=NG ensures Nigerian results; "when:30d" limits to last 30 days
GOOGLE_NEWS_URL = (
    "https://news.google.com/rss/search"
    "?q={query}+Nigeria+stock+when:30d"
    "&hl=en-NG&gl=NG&ceid=NG:en"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT = 15  # seconds
GOOGLE_DELAY = 2.5  # seconds between Google News requests (polite, avoids rate-limit)


class NewsScraper:
    """Fetches and caches Nigerian stock news headlines."""

    def __init__(self, cache_hours: int = 6, descriptions: dict = None):
        self.cache_hours = cache_hours
        # descriptions: { "TICKER": "Full Company Name Plc" } from snapshot
        self._descriptions = descriptions or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self, tickers: list[str], max_per_ticker: int = 5) -> dict:
        """
        Fetch news for a list of tickers.
        Returns: { "TICKER": [ { "title", "source", "date", "url" }, ... ] }

        Strategy:
          1. Pull broad RSS feeds (supplementary) and match to tickers.
          2. For EVERY ticker, also query Google News RSS with targeted
             company-name search — this is the primary hit source.
          3. Deduplicate by title, cache results.
        """
        cache = self._load_cache()
        if cache is not None:
            logger.info("Using cached news (< %dh old)", self.cache_hours)
            return {t: cache.get(t, [])[:max_per_ticker] for t in tickers}

        # Step 1: broad RSS feeds (fast, ~50 articles)
        broad_articles = self._scrape_rss_feeds()
        matched = self._match_to_tickers(broad_articles, tickers)

        # Step 2: Google News RSS per ticker (targeted, high coverage)
        google_hits = 0
        for i, ticker in enumerate(tickers):
            gnews = self._google_news_search(ticker)
            if gnews:
                google_hits += len(gnews)
                existing_titles = {a["title"].lower() for a in matched.get(ticker, [])}
                for article in gnews:
                    if article["title"].lower() not in existing_titles:
                        matched.setdefault(ticker, []).append(article)
                        existing_titles.add(article["title"].lower())

            # Progress log every 10 tickers
            if (i + 1) % 10 == 0:
                logger.info(
                    "Google News: %d/%d tickers queried (%d hits so far)",
                    i + 1, len(tickers), google_hits,
                )

        logger.info(
            "Google News total: %d articles for %d tickers",
            google_hits, len(tickers),
        )

        # Step 3: cache and return
        self._save_cache(matched)

        return {t: matched.get(t, [])[:max_per_ticker] for t in tickers}

    # ------------------------------------------------------------------
    # Google News RSS (PRIMARY source)
    # ------------------------------------------------------------------

    def _google_news_search(
        self, ticker: str, max_results: int = 5
    ) -> list[dict]:
        """
        Query Google News RSS for a specific ticker using its primary alias.
        Returns up to max_results articles.
        """
        aliases = TICKER_ALIASES.get(ticker)
        if aliases:
            query = aliases[0]  # Primary alias = best search term
        elif ticker in self._descriptions:
            # Auto-generate from snapshot description
            desc = self._descriptions[ticker]
            # Strip common suffixes for cleaner search
            import re as _re
            query = _re.sub(
                r'\s+(Plc\.?|PLC|Ltd\.?|Limited|Co\.?|Units\s+NGN)\s*$',
                '', desc, flags=_re.IGNORECASE
            ).strip()
            if not query:
                query = ticker
        else:
            query = ticker

        url = GOOGLE_NEWS_URL.format(query=quote_plus(f'"{query}"'))

        try:
            feed = feedparser.parse(url, request_headers=HEADERS)
            results = []
            for entry in feed.entries[:max_results]:
                pub_date = ""
                if hasattr(entry, "published"):
                    pub_date = entry.published
                elif hasattr(entry, "updated"):
                    pub_date = entry.updated

                # Google News wraps the real source in source tag
                source = "Google News"
                if hasattr(entry, "source") and hasattr(entry.source, "title"):
                    source = entry.source.title

                results.append(
                    {
                        "title": entry.get("title", "").strip(),
                        "source": source,
                        "date": pub_date,
                        "url": entry.get("link", ""),
                    }
                )

            if results:
                logger.debug(
                    "%s: %d articles from Google News", ticker, len(results)
                )

            time.sleep(GOOGLE_DELAY)  # Polite delay
            return results

        except Exception as exc:
            logger.warning("Google News failed for %s: %s", ticker, exc)
            time.sleep(GOOGLE_DELAY)
            return []

    # ------------------------------------------------------------------
    # Broad RSS Parsing (supplementary)
    # ------------------------------------------------------------------

    def _scrape_rss_feeds(self) -> list[dict]:
        """Parse all configured RSS feeds and return a flat list of articles."""
        articles = []
        for feed_cfg in RSS_FEEDS:
            try:
                logger.info("Fetching RSS: %s", feed_cfg["name"])
                feed = feedparser.parse(
                    feed_cfg["url"],
                    request_headers=HEADERS,
                )
                for entry in feed.entries:
                    pub_date = ""
                    if hasattr(entry, "published"):
                        pub_date = entry.published
                    elif hasattr(entry, "updated"):
                        pub_date = entry.updated

                    articles.append(
                        {
                            "title": entry.get("title", "").strip(),
                            "summary": entry.get("summary", "").strip(),
                            "url": entry.get("link", ""),
                            "date": pub_date,
                            "source": feed_cfg["name"],
                        }
                    )
            except Exception as exc:
                logger.warning("RSS fetch failed for %s: %s", feed_cfg["name"], exc)

        logger.info("Fetched %d articles from broad RSS feeds", len(articles))
        return articles

    # ------------------------------------------------------------------
    # Ticker matching (for broad RSS articles)
    # ------------------------------------------------------------------

    def _match_to_tickers(
        self, articles: list[dict], tickers: list[str]
    ) -> dict[str, list[dict]]:
        """Match articles to tickers using alias keywords."""
        matched: dict[str, list[dict]] = {t: [] for t in tickers}

        for article in articles:
            text = (article["title"] + " " + article.get("summary", "")).lower()
            for ticker in tickers:
                aliases = TICKER_ALIASES.get(ticker, [ticker.lower()])
                if any(alias.lower() in text for alias in aliases):
                    matched[ticker].append(
                        {
                            "title": article["title"],
                            "source": article["source"],
                            "date": article["date"],
                            "url": article["url"],
                        }
                    )
        return matched

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        return NEWS_DIR / f"{today}.json"

    def _load_cache(self) -> Optional[dict]:
        path = self._cache_path()
        if not path.exists():
            return None
        age = datetime.now().timestamp() - path.stat().st_mtime
        if age > self.cache_hours * 3600:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, data: dict) -> None:
        try:
            self._cache_path().write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save news cache: %s", exc)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scraper = NewsScraper()
    test_tickers = ["WEMABANK", "TIP", "CUSTODIAN", "WAPCO", "FCMB",
                     "ZENITHBANK", "GTCO", "DANGCEM", "SEPLAT", "MTNN"]
    results = scraper.fetch_all(test_tickers)

    total_with_news = 0
    for ticker, articles in results.items():
        print(f"\n{'='*60}")
        print(f"  {ticker} -- {len(articles)} headline(s)")
        print(f"{'='*60}")
        if not articles:
            print("  (no recent news found)")
        else:
            total_with_news += 1
        for a in articles:
            src = a.get("source", "")
            print(f"  [{src}] {a['title']}")
            if a.get("url"):
                print(f"    -> {a['url']}")

    print(f"\n{'='*60}")
    print(f"  COVERAGE: {total_with_news}/{len(test_tickers)} tickers have news")
    print(f"{'='*60}")


def fetch_news_for_tickers(tickers: list[str], max_per_ticker: int = 5):
    return NewsScraper().fetch_all(tickers, max_per_ticker=max_per_ticker)

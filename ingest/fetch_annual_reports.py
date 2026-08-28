"""
AfricanFinancials Report Scraper
=================================
Scrapes investor-relations documents (Annual Reports, Interim Reports,
Abridged Reports) from https://africanfinancials.com for Nigerian-listed
companies.

Discovery is driven by the documents index page:
    https://africanfinancials.com/nigeria-listed-company-documents/
Each entry on the index links to a detail page whose AI-generated summary
contains a structured KPI block, a liquidity section, and an embedded
Google-Drive PDF of the full report. We capture all three, plus extract
text from the PDF itself.

Per-document artefacts:
  data/reports/<date>.json        — one JSON record per ticker
  data/reports/pdfs/<slug>.pdf    — downloaded PDF (cached)
  data/reports/pdf_text/<slug>.txt — extracted plain text (cached)

Usage:
    from ingest.fetch_annual_reports import ReportScraper
    scraper = ReportScraper()
    reports = scraper.fetch_all(["DANGCEM", "MTNN", "GTCO"])
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cloudscraper
import requests
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "data" / "reports"
PDF_DIR = REPORTS_DIR / "pdfs"
PDF_TEXT_DIR = REPORTS_DIR / "pdf_text"
for _d in (REPORTS_DIR, PDF_DIR, PDF_TEXT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── AfricanFinancials configuration ──────────────────────────────────

AFF_BASE = "https://africanfinancials.com"
AFF_DOCS_INDEX = f"{AFF_BASE}/nigeria-listed-company-documents/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://africanfinancials.com/",
}

REQUEST_TIMEOUT = (10, 30)  # (connect, read) seconds — read bumped for PDF downloads
POLITE_DELAY = 2.0  # seconds between HTML requests
PDF_TIMEOUT = (10, 120)  # PDFs can be 50MB+; give them 2 minutes

# ── Ticker → AfricanFinancials slug mapping ──────────────────────────
# Used only as a fallback for old/rare reports that no longer appear in
# the docs index. Index-driven discovery is the primary path.

TICKER_TO_AFF_SLUG: dict[str, str] = {
    # ── Banks & Financial Services ──
    "ACCESSCORP":  "access",
    "ETI":         "eti",
    "FBNH":        "fbn",
    "FCMB":        "fcmb",
    "FIDELITYBK":  "fidelity",
    "FIRSTHOLDCO": "firsthold",
    "GTCO":        "gtco",
    "JAIZBANK":    "jaiz",
    "NPFMCRFBK":   "npfmcr",
    "STANBIC":     "stanbic",
    "STERLINGNG":  "sterling",
    "UBA":         "uba",
    "UNITYBNK":    "unity",
    "WEMABANK":    "wema",
    "ZENITHBANK":  "zenith",
    # ── Other Financial / Capital Markets ──
    "AFRIPRUD":    "afrip",
    "CILEASING":   "cilea",
    "DEAPCAP":     "deapcap",
    "NGXGROUP":    "ngxg",
    "UCAP":        "ubcap",     # verified from index
    "VERITASKAP":  "veritaskap",
    "VFDGROUP":    "vfd",
    # ── Insurance ──
    "AFRINSURE":   "afrin",
    "AIICO":       "aiico",
    "CORNERST":    "corner",
    "CUSTODIAN":   "custo",
    "FTGINSURE":   "ftginsure",
    "GUINEAINS":   "guinea",
    "INTENEGINS":  "inten",
    "LASACO":      "lasaco",
    "LINKASSURE":  "link",
    "MANSARD":     "mansard",
    "MBENEFIT":    "mutben",
    "NEM":         "nem",
    "REGALINS":    "regal",
    "ROYALEX":     "royalex",
    "SOVRENINS":   "sovren",    # verified from index
    "STACO":       "staco",
    "SUNUASSUR":   "sunu",
    "UNIVINSURE":  "univ",
    "WAPIC":       "wapic",
    # ── Industrial / Building / Materials ──
    "BERGER":      "berger",
    "BETAGLAS":    "beta",
    "BUACEMENT":   "buac",
    "CAP":         "cap",
    "CUTIX":       "cutix",
    "DANGCEM":     "dangce",
    "DUNLOP":      "dunlop",
    "ENAMELWA":    "enamel",
    "MEYER":       "meyer",
    "PREMPAINTS":  "premp",
    "WAPCO":       "wapco",
    # ── Consumer Goods / Food & Beverage ──
    "BUAFOODS":    "buaf",
    "CADBURY":     "cadbury",
    "CHAMPION":    "champ",
    "DANGSUGAR":   "dangsug",
    "FLOURMILL":   "fmn",
    "FTNCOCOA":    "ftnco",
    "GUINNESS":    "guinness",
    "HONYFLOUR":   "hony",
    "INTBREW":     "intbrew",
    "MCNICHOLS":   "mcnich",
    "NASCON":      "nascon",
    "NB":          "nb",
    "NESTLE":      "nestle",
    "NNFM":        "nnfm",
    "PZ":          "pz",
    "TANTALIZER":  "tanta",
    "UNILEVER":    "unilev",    # verified from index
    "UNIONDICON":  "udc",
    "UPL":         "upl",
    "VITAFOAM":    "vitafoam",
    # ── Agriculture ──
    "ELLAHLAKES":  "ellahl",
    "LIVESTOCK":   "livestock",
    "OKOMUOIL":    "okomu",
    "PRESCO":      "presco",
    # ── Oil & Gas / Energy ──
    "ARADEL":      "aradel",
    "CONOIL":      "conoil",
    "ETERNA":      "eterna",
    "EUNISELL":    "eunisell",
    "GEREGU":      "geregu",
    "JAPAULGOLD":  "japaul",
    "MRS":         "mrs",
    "OANDO":       "oando",
    "SEPLAT":      "seplat",
    "TOTAL":       "total",
    # ── Telecoms / Tech / ICT ──
    "AIRTELAFRI":  "airtel",
    "CHAMS":       "chams",
    "CWG":         "cwg",
    "ETRANZACT":   "etran",
    "MTNN":        "mtn",
    "NSLTECH":     "nsltec",    # verified from index
    "OMATEK":      "omatek",
    # ── Construction & Engineering ──
    "ABCTRANS":    "abctr",
    "AUSTINLAZ":   "austi",
    "CAVERTON":    "caver",
    "JBERGER":     "jber",
    "NAHCO":       "nahco",
    "RTBRISCOE":   "rtbriscoe",
    "SKYAVN":      "skyavn",
    "TRANSEXPR":   "transexpr",
    "TRANSPOWER":  "tpower",
    "REDSTAREX":   "redstar",
    # ── Real Estate / REITs / Mortgage ──
    "ABBEYBDS":    "abbey",
    "LIVINGTRUST": "living",
    "NREIT":       "nreit",
    "SFSREIT":     "sfsreit",
    "UHOMREIT":    "uhom",
    "UPDC":        "updc",
    "UPDCREIT":    "updcrei",
    # ── Healthcare / Pharma ──
    "EKOCORP":     "eko",
    "FIDSON":      "fidson",
    "GLAXOSMITH":  "gsk",
    "MAYBAKER":    "maybaker",
    "MECURE":      "mecure",
    "MORISON":     "morison",
    "NEIMETH":     "neimeth",
    "PHARMDEKO":   "pharm",
    # ── Conglomerates / Holding ──
    "CHELLARAM":   "chella",
    "JOHNHOLT":    "johnholt",
    "SCOA":        "scoa",
    "TRANSCORP":   "transcorp",
    "TRIPPLEG":    "trippl",
    "UACN":        "uacn",
    "VANLEER":     "vanleer",
    # ── Hospitality / Leisure / Services ──
    "ACADEMY":     "acad",
    "CONHALLPLC":  "conhallplc",
    "DAARCOMM":    "daar",
    "IKEJAHOTEL":  "ikejah",
    "IMG":         "img",
    "INFINITY":    "infini",
    "JULI":        "juli",
    "LEARNAFRCA":  "learn",
    "LEGENDINT":   "legend",
    "MULTIVERSE":  "multi",
    "NCR":         "ncr",
    "PRESTIGE":    "prest",
    "TIP":         "tip",
    "TRANSCOHOT":  "transc",
    "ZICHIS":      "zichi",
    # ── Media / Print ──
    "AFROMEDIA":   "afrom",
    "ALEX":        "alex",
    "HMCALL":      "hmcall",
    "THOMASWY":    "thom",
}

DOC_TYPE_PRIORITY = ["ar", "ab", "ir", "pr"]  # annual, abridged, interim, presentation
DOC_TYPE_LABELS = {
    "ar": "Annual Report",
    "ab": "Abridged Report",
    "ir": "Interim Report",
    "pr": "Presentation",
}

# Reverse lookup: AFF URL-slug → NGX ticker. URL slugs (e.g. `ucap`,
# `sovren`, `updcrei`) are canonical on AFF; the ticker strings printed
# in index titles are often truncated (UNILEVER → UNILEV) and unreliable.
AFF_SLUG_TO_TICKER: dict[str, str] = {v: k for k, v in TICKER_TO_AFF_SLUG.items()}

# Section headings that appear in the AI-generated summary block.
# The keys are normalised identifiers used in the output `sections` dict.
_SECTION_ALIASES = {
    "executive_summary":              {"executive summary"},
    "financial_performance":          {"financial performance", "financial performance highlights"},
    "operational_highlights":         {"operational highlights", "key operational highlights"},
    "strategic_initiatives":          {"strategic initiatives", "strategic initiatives and future outlook"},
    "risk_management":                {"risk management", "risk management and governance"},
    "sustainability_esg":             {"sustainability", "esg performance", "sustainability and esg performance"},
    "key_performance_indicators":     {"key performance indicators", "key performance indicators (kpis)", "kpis"},
    "future_outlook":                 {"future outlook", "investment and future outlook"},
    "governance":                     {"governance"},
    "dividends":                      {"dividends", "dividends and shareholder returns"},
    "management_commentary":          {"management commentary"},
    "market_insights":                {"market and industry insights", "market insights"},
    "risk_factors":                   {"risk factors", "risk factors and challenges"},
    "liquidity":                      {"indicative share trading liquidity"},
    "company_profile":                {"company profile"},
}

# ── Helper functions ─────────────────────────────────────────────────

def _slug_for_ticker(ticker: str) -> str:
    return TICKER_TO_AFF_SLUG.get(ticker.upper(), ticker.lower())


def _build_company_url(ticker: str) -> str:
    return f"{AFF_BASE}/company/ng-{_slug_for_ticker(ticker)}/"


def _cache_path(date_str: Optional[str] = None) -> Path:
    ds = date_str or datetime.now().strftime("%Y-%m-%d")
    return REPORTS_DIR / f"{ds}.json"


def _load_cache(date_str: Optional[str] = None) -> dict:
    p = _cache_path(date_str)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(data: dict, date_str: Optional[str] = None) -> None:
    p = _cache_path(date_str)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Cache saved -> %s", p)


def _normalise_heading(text: str) -> Optional[str]:
    """Map a raw heading string to a canonical section key."""
    t = text.strip().lower().rstrip(":")
    for key, aliases in _SECTION_ALIASES.items():
        if t in aliases:
            return key
    # Loose match — "Financial Performance: FY 2025" style
    for key, aliases in _SECTION_ALIASES.items():
        for a in aliases:
            if t.startswith(a):
                return key
    return None


def _parse_document_url(url: str) -> Optional[dict]:
    """
    Parse metadata from an AfricanFinancials document URL.
      /document/ng-dangce-2025-ar-00/   → year=2025, doc_type=ar, period=00, company_slug=dangce
      /document/ng-cutix-2026-ir-q3/    → year=2026, doc_type=ir, period=q3, company_slug=cutix
    """
    match = re.search(
        r"/document/(ng-([\w]+?)-(\d{4})-(ar|ab|ir|pr)-(\w+))/", url
    )
    if not match:
        return None
    return {
        "slug": match.group(1),
        "company_slug": match.group(2),
        "year": int(match.group(3)),
        "doc_type": match.group(4),
        "period": match.group(5),
    }


# ── Google Drive PDF handling ────────────────────────────────────────

_DRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)")


def _drive_id_from_src(src: str) -> Optional[str]:
    m = _DRIVE_FILE_RE.search(src or "")
    return m.group(1) if m else None


def _drive_download_url(drive_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={drive_id}"


def _download_pdf(session, drive_id: str, dest: Path) -> Optional[int]:
    """
    Download a Google-Drive-hosted PDF. Returns size in bytes on success,
    None on failure. Skips if the file already exists and looks valid.
    """
    if dest.exists() and dest.stat().st_size > 1024:
        with open(dest, "rb") as fh:
            head = fh.read(4)
        if head == b"%PDF":
            return dest.stat().st_size

    url = _drive_download_url(drive_id)
    try:
        resp = session.get(url, timeout=PDF_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("PDF download failed (%s): %s", drive_id, e)
        return None

    content = resp.content
    if content[:4] != b"%PDF":
        # Likely the Drive virus-scan interstitial for large files.
        m = re.search(rb"confirm=([0-9A-Za-z_]+)", content)
        if m:
            token = m.group(1).decode()
            confirm_url = f"{url}&confirm={token}"
            try:
                resp = session.get(confirm_url, timeout=PDF_TIMEOUT, allow_redirects=True)
                resp.raise_for_status()
                content = resp.content
            except requests.RequestException as e:
                logger.warning("PDF confirm fetch failed (%s): %s", drive_id, e)
                return None
        if content[:4] != b"%PDF":
            logger.warning("PDF download returned non-PDF content for %s", drive_id)
            return None

    with open(dest, "wb") as fh:
        fh.write(content)
    return len(content)


def _extract_pdf_text(pdf_path: Path, text_path: Path) -> tuple[Optional[str], Optional[int]]:
    """
    Extract plain text from a PDF using pypdf. Writes the full text to
    text_path. Returns (excerpt, page_count) where excerpt is the first
    ~2000 chars. Skips extraction if text_path already exists.
    """
    try:
        import pypdf
    except ImportError:
        logger.warning("pypdf not installed — skipping PDF text extraction")
        return None, None

    if text_path.exists() and text_path.stat().st_size > 0:
        with open(text_path, "r", encoding="utf-8") as fh:
            existing = fh.read(4096)
        # page count is in header we write below
        pages = None
        m = re.match(r"# pages=(\d+)\n", existing)
        if m:
            pages = int(m.group(1))
            existing = existing[m.end():]
        return existing[:2000], pages

    try:
        reader = pypdf.PdfReader(str(pdf_path))
        n_pages = len(reader.pages)
        parts = []
        for i, page in enumerate(reader.pages):
            try:
                parts.append(page.extract_text() or "")
            except Exception as e:
                logger.debug("PDF page %d extract failed: %s", i, e)
                parts.append("")
        full = "\n\n".join(parts)
    except Exception as e:
        logger.warning("PDF parse failed (%s): %s", pdf_path.name, e)
        return None, None

    with open(text_path, "w", encoding="utf-8") as fh:
        fh.write(f"# pages={n_pages}\n")
        fh.write(full)
    return full[:2000], n_pages


# ── Core scraping logic ─────────────────────────────────────────────

class ReportScraper:
    """
    Scrapes AfricanFinancials for investor-relations report data.

    Discovery strategy (in order):
      1. Walk the documents index until every requested ticker has a hit.
      2. For any ticker still missing, try its company page.
      3. As a last resort, probe direct document URLs using the slug map.
    """

    def __init__(
        self,
        use_cache: bool = True,
        download_pdf: bool = True,
        parse_pdf: bool = True,
        max_index_pages: int = 50,
    ):
        self.session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True},
        )
        self.use_cache = use_cache
        self.download_pdf = download_pdf
        self.parse_pdf = parse_pdf
        self.max_index_pages = max_index_pages

    # ── Public API ───────────────────────────────────────────────────

    def fetch_all(
        self,
        tickers: list[str],
        year: Optional[int] = None,
    ) -> dict[str, dict]:
        """
        Fetch report data for a list of tickers. Returns dict ticker → report.
        """
        target_year = year or datetime.now().year
        today = datetime.now().strftime("%Y-%m-%d")

        cached_hits: dict[str, dict] = {}
        cache: dict = {}
        if self.use_cache:
            cache = _load_cache(today)
            if cache:
                for t in tickers:
                    t_upper = t.upper()
                    for k, v in cache.items():
                        if v.get("snapshot_ticker", k) == t_upper:
                            cached_hits[k] = v
                            break
                missing = [t for t in tickers if not any(v.get("snapshot_ticker", k) == t.upper() for k, v in cache.items())]
                if not missing:
                    logger.info("All %d tickers served from cache", len(tickers))
                    return cached_hits
                logger.info("%d cached, %d to fetch", len(cached_hits), len(missing))
                tickers = missing

        # Step 1 — index-driven discovery for the whole batch
        logger.info("Walking documents index for %d tickers ...", len(tickers))
        index_docs = self._walk_index(tickers, target_year)

        results: dict[str, dict] = {}
        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            logger.info("[%d/%d] %s", i, total, ticker)
            try:
                doc = self._select_doc_for_ticker(ticker, target_year, index_docs)
                if doc is None:
                    logger.warning("  [X] %s — not found in index, falling back", ticker)
                    doc = self._fallback_discover(ticker, target_year)
                if doc is None:
                    logger.warning("  [X] %s — no report found", ticker)
                    continue
                report = self._fetch_detail(doc, ticker)
                if report:
                    aff_ticker = report["ticker"]
                    results[aff_ticker] = report
                    logger.info(
                        "  [OK] %s — %s (%s %s)",
                        aff_ticker,
                        report.get("title", "?")[:60],
                        report.get("doc_type", "?"),
                        report.get("year", "?"),
                    )
            except Exception as e:
                logger.exception("  [X] %s — error: %s", ticker, e)

            if i < total:
                time.sleep(POLITE_DELAY)

        cache.update(results)
        _save_cache(cache, today)
        return {**cached_hits, **results}

    # ── Index-driven discovery ──────────────────────────────────────

    def _walk_index(
        self,
        tickers_wanted: list[str],
        target_year: int,
    ) -> dict[str, list[dict]]:
        """
        Paginate the docs index and collect entries keyed by ticker.

        Stops early once every wanted ticker has at least one doc for
        target_year OR (target_year - 1), whichever is found first.
        """
        wanted = {t.upper() for t in tickers_wanted}
        found: dict[str, list[dict]] = {t: [] for t in wanted}
        fresh_years = {target_year, target_year - 1}

        for page in range(1, self.max_index_pages + 1):
            url = AFF_DOCS_INDEX if page == 1 else f"{AFF_DOCS_INDEX}page/{page}/"
            resp = None
            for attempt in range(2):
                try:
                    resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                except requests.RequestException as e:
                    logger.warning("Index page %d fetch error: %s", page, e)
                    resp = None
                    break
                if resp.status_code == 404:
                    resp = None
                    break
                if resp.status_code in (403, 429) and attempt == 0:
                    logger.info("Index page %d rate-limited (%d); backing off", page, resp.status_code)
                    time.sleep(10)
                    continue
                if not resp.ok:
                    logger.warning("Index page %d returned %d", page, resp.status_code)
                    resp = None
                break
            if resp is None:
                # 403/404 — stop crawling but don't abort the batch; fallback
                # per-ticker discovery will handle whoever we missed.
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            entries = self._parse_index_page(soup)
            if not entries:
                break

            for entry in entries:
                snapshot_ticker = entry.get("snapshot_ticker")
                matched = None
                if snapshot_ticker and snapshot_ticker in wanted:
                    matched = snapshot_ticker
                else:
                    # Fuzzy-match the AFF title ticker against the user's
                    # wanted NGX tickers (handles UNILEVER ↔ UNILEV etc.).
                    title_tkr = entry.get("ticker")
                    matched = (
                        self._matches_wanted(snapshot_ticker, wanted)
                        or self._matches_wanted(title_tkr, wanted)
                    )
                if matched:
                    entry["snapshot_ticker"] = matched
                    found[matched].append(entry)

            # Early stop: every wanted ticker has at least one fresh-year doc
            all_fresh = all(
                any(d.get("year") in fresh_years for d in found[t])
                for t in wanted
            )
            if all_fresh:
                logger.info(
                    "Index walk: every ticker has a %d/%d doc after %d pages",
                    target_year - 1, target_year, page,
                )
                break

            time.sleep(POLITE_DELAY)

        hit_count = sum(1 for t in wanted if found[t])
        logger.info("Index walk complete: %d/%d tickers matched", hit_count, len(wanted))
        return found

    def _parse_index_page(self, soup: BeautifulSoup) -> list[dict]:
        """Extract document entries from a docs-index page."""
        entries: list[dict] = []
        seen_urls: set[str] = set()

        # Each entry lives in a div.af20-news block
        for block in soup.find_all("div", class_="af20-news"):
            # The main title link is the first /document/ anchor
            title_link = None
            for a in block.find_all("a", href=True):
                if "/document/" in a["href"]:
                    title_link = a
                    break
            if title_link is None:
                continue

            href = title_link["href"]
            if not href.startswith("http"):
                href = AFF_BASE + href
            if href in seen_urls:
                continue
            seen_urls.add(href)

            title = title_link.get_text(strip=True)
            meta = _parse_document_url(href) or {}
            
            aff_ticker = self._ticker_from_title(title)
            if not aff_ticker:
                aff_ticker = (meta.get("company_slug") or "").upper()
                
            ngx_ticker = AFF_SLUG_TO_TICKER.get(meta.get("company_slug") or "")
            if not ngx_ticker:
                ngx_ticker = aff_ticker

            # Blurb: the paragraph text inside the block, stripped of title
            block_text = block.get_text(" ", strip=True)
            blurb = block_text
            if title in blurb:
                blurb = blurb.split(title, 1)[-1].strip()

            entries.append({
                "url": href,
                "title": title,
                "ticker": aff_ticker,
                "snapshot_ticker": ngx_ticker,
                "slug": meta.get("slug"),
                "year": meta.get("year"),
                "doc_type": meta.get("doc_type"),
                "period": meta.get("period"),
                "index_blurb": blurb[:800],
            })

        # Fallback: if the af20-news selector misses, scan raw /document/ anchors
        if not entries:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/document/" not in href:
                    continue
                if not href.startswith("http"):
                    href = AFF_BASE + href
                if href in seen_urls:
                    continue
                seen_urls.add(href)
                title = a.get_text(strip=True)
                if not title:
                    continue
                meta = _parse_document_url(href) or {}
                
                aff_ticker = self._ticker_from_title(title)
                if not aff_ticker:
                    aff_ticker = (meta.get("company_slug") or "").upper()
                    
                ngx_ticker = AFF_SLUG_TO_TICKER.get(meta.get("company_slug") or "")
                if not ngx_ticker:
                    ngx_ticker = aff_ticker
                    
                entries.append({
                    "url": href,
                    "title": title,
                    "ticker": aff_ticker,
                    "snapshot_ticker": ngx_ticker,
                    "slug": meta.get("slug"),
                    "year": meta.get("year"),
                    "doc_type": meta.get("doc_type"),
                    "period": meta.get("period"),
                    "index_blurb": "",
                })

        return entries

    @staticmethod
    def _ticker_from_title(title: str) -> Optional[str]:
        """Pull the ticker from '(TICKER.ng)' in an entry title."""
        m = re.search(r"\(([A-Z0-9]+)\.ng\)", title)
        return m.group(1) if m else None

    @staticmethod
    def _matches_wanted(candidate: Optional[str], wanted: set[str]) -> Optional[str]:
        """
        Given an AFF ticker/slug candidate, return the matching NGX ticker
        from `wanted` if any. Tolerant of the common AFF↔NGX transforms:
          - truncation (UNILEVER ↔ UNILEV)
          - dropped suffixes (ZENITHBANK ↔ ZENITH, SOVRENINS ↔ SOVREN, MTNN ↔ MTN)
        """
        if not candidate:
            return None
        c = candidate.upper()
        if c in wanted:
            return c
        suffixes = ("NG", "BK", "BANK", "INS", "INSURE", "GROUP",
                    "HOLDCO", "PLC", "N", "COMM", "NIG")
        for w in wanted:
            if c == w:
                return w
            # Prefix-of either way — catches UNILEV↔UNILEVER and MTN↔MTNN
            if len(c) >= 3 and len(w) >= 3 and (c.startswith(w) or w.startswith(c)):
                return w
            # Strip suffix from NGX ticker then compare
            for sfx in suffixes:
                if w.endswith(sfx) and len(w) > len(sfx) + 2:
                    stem = w[:-len(sfx)]
                    if stem == c or stem.startswith(c) or c.startswith(stem):
                        return w
        return None

    def _select_doc_for_ticker(
        self,
        ticker: str,
        target_year: int,
        index_docs: dict[str, list[dict]],
    ) -> Optional[dict]:
        docs = index_docs.get(ticker.upper(), [])
        if not docs:
            return None
        return self._pick_best_document(docs, target_year)

    @staticmethod
    def _pick_best_document(
        docs: list[dict], target_year: int
    ) -> Optional[dict]:
        if not docs:
            return None

        def sort_key(d: dict) -> tuple:
            year = d.get("year") or 0
            year_match = 0 if year == target_year else 1
            try:
                type_rank = DOC_TYPE_PRIORITY.index(d.get("doc_type", ""))
            except ValueError:
                type_rank = 99
            return (year_match, type_rank, -year)

        return sorted(docs, key=sort_key)[0]

    # ── Fallbacks: company page + direct URL probing ─────────────────

    def _fallback_discover(self, ticker: str, target_year: int) -> Optional[dict]:
        docs = self._get_company_page_docs(ticker)
        best = self._pick_best_document(docs, target_year) if docs else None
        if best:
            return best
        return self._probe_direct_urls(ticker, target_year)

    def _get_company_page_docs(self, ticker: str) -> list[dict]:
        url = _build_company_url(ticker)
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug("Company page fetch failed for %s: %s", ticker, e)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        docs: list[dict] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/document/" not in href:
                continue
            title = a.get_text(strip=True)
            if not title:
                continue
            if not href.startswith("http"):
                href = AFF_BASE + href
            meta = _parse_document_url(href)
            if meta:
                meta["title"] = title
                meta["url"] = href
                
                aff_ticker = self._ticker_from_title(title)
                if not aff_ticker:
                    aff_ticker = (meta.get("company_slug") or "").upper()
                    
                meta["ticker"] = aff_ticker
                meta["snapshot_ticker"] = ticker.upper()
                meta["index_blurb"] = ""
                docs.append(meta)
        return docs

    def _probe_direct_urls(self, ticker: str, target_year: int) -> Optional[dict]:
        slug = _slug_for_ticker(ticker)
        for year in (target_year, target_year - 1, target_year - 2):
            for doc_type in DOC_TYPE_PRIORITY:
                url = f"{AFF_BASE}/document/ng-{slug}-{year}-{doc_type}-00/"
                try:
                    resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                except requests.RequestException:
                    continue
                if resp.status_code == 200:
                    return {
                        "url": url,
                        "slug": f"ng-{slug}-{year}-{doc_type}-00",
                        "year": year,
                        "doc_type": doc_type,
                        "period": "00",
                        "title": "",
                        "ticker": slug.upper(),
                        "snapshot_ticker": ticker.upper(),
                        "index_blurb": "",
                    }
                time.sleep(0.3)
        return None

    # ── Detail page extraction ──────────────────────────────────────

    def _fetch_detail(self, doc: dict, ticker: str) -> Optional[dict]:
        url = doc["url"]
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Detail fetch failed (%s): %s", url, e)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        content_tb = soup.find("div", class_="fusion-content-tb")

        sections = self._extract_sections(content_tb) if content_tb else {}
        kpis_structured = self._extract_kpis_structured(content_tb) if content_tb else {}
        liquidity = self._extract_liquidity(content_tb, soup)
        summary_text = self._legacy_summary_text(content_tb) if content_tb else ""
        kpis_legacy = self._legacy_kpis_regex(summary_text)
        published = self._extract_published_date(soup)
        if not published:
            # Index blurb often contains "Published: Apr 22, 2026"
            blurb = doc.get("index_blurb", "") or ""
            m = re.search(r"Published[:\s]+(\w+\.?\s+\d{1,2},?\s*\d{4})", blurb)
            if m:
                published = m.group(1).strip()

        title = doc.get("title", "")
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        pdf_info = self._extract_pdf_info(soup, doc)

        record = {
            # Existing surface — callers rely on these keys
            "ticker": doc.get("ticker", ticker.upper()),
            "snapshot_ticker": doc.get("snapshot_ticker", ticker.upper()),
            "title": title,
            "url": url,
            "doc_type": doc.get("doc_type", ""),
            "doc_type_label": DOC_TYPE_LABELS.get(doc.get("doc_type", ""), "Unknown"),
            "year": doc.get("year"),
            "period": doc.get("period", ""),
            "published": published,
            "summary": summary_text,
            "kpis": kpis_legacy,
            "fetched_at": datetime.now().isoformat(),
            # New structured fields
            "index_blurb": doc.get("index_blurb", ""),
            "sections": sections,
            "kpis_structured": kpis_structured,
            "liquidity": liquidity,
            "pdf": pdf_info,
        }
        return record

    def _extract_sections(self, container: Tag) -> dict[str, str]:
        """
        Walk headings (h2/h3) inside the Avada content block and collect
        each section's body text into a dict keyed by canonical section id.
        """
        sections: dict[str, str] = {}
        current_key: Optional[str] = None
        current_parts: list[str] = []

        def flush():
            if current_key and current_parts:
                joined = "\n".join(p for p in current_parts if p).strip()
                if joined:
                    # If the same key appears twice, keep the longer text
                    if current_key in sections and len(sections[current_key]) >= len(joined):
                        return
                    sections[current_key] = joined

        for elem in container.find_all(["h2", "h3", "h4", "p", "ul", "ol"]):
            if elem.name in ("h2", "h3", "h4"):
                flush()
                current_parts = []
                current_key = _normalise_heading(elem.get_text(strip=True))
            elif current_key:
                if elem.name in ("ul", "ol"):
                    for li in elem.find_all("li", recursive=False):
                        t = li.get_text(" ", strip=True)
                        if t:
                            current_parts.append(f"• {t}")
                else:
                    t = elem.get_text(" ", strip=True)
                    if t and "generated by an AI" not in t.lower():
                        current_parts.append(t)
        flush()
        return sections

    def _extract_kpis_structured(self, container: Tag) -> dict[str, str]:
        """
        Parse the '<h3>Key Performance Indicators (KPIs)</h3>' block's
        <ul> into {metric_name: value} pairs.

        Each <li> looks like one of:
          "Profit Before Tax: ₦1.23 trillion"
          "Return on Equity (Pre-tax): 40.21%"
          "Gross Written Premium Growth: 20%"
        """
        kpis: dict[str, str] = {}
        for h in container.find_all(["h2", "h3", "h4"]):
            txt = h.get_text(strip=True).lower()
            if "key performance" not in txt and txt != "kpis":
                continue
            # Find the first <ul> that follows this heading
            sib = h.find_next_sibling()
            while sib and sib.name not in ("h2", "h3", "h4"):
                if sib.name == "ul":
                    for li in sib.find_all("li", recursive=False):
                        raw = li.get_text(" ", strip=True)
                        name, value = _split_kpi_line(raw)
                        if name and value:
                            kpis[name] = value
                    break
                sib = sib.find_next_sibling()
            if kpis:
                break
        return kpis

    def _extract_liquidity(self, container: Optional[Tag], soup: BeautifulSoup) -> dict:
        """
        Parse the 'Indicative Share Trading Liquidity' paragraph into
        structured fields.
        """
        raw = ""
        scope = container or soup
        for h in scope.find_all(["h2", "h3", "h4"]):
            if "indicative share trading liquidity" in h.get_text(strip=True).lower():
                sib = h.find_next_sibling()
                while sib and sib.name not in ("h2", "h3", "h4"):
                    if sib.name in ("p", "div"):
                        raw = sib.get_text(" ", strip=True)
                        break
                    sib = sib.find_next_sibling()
                break
        if not raw:
            return {}

        result: dict[str, object] = {"raw": raw}
        # "as of 2nd April 2026"
        m = re.search(r"as of\s+([0-9]{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})", raw, re.I)
        if m:
            result["as_of"] = m.group(1).strip()
        # Total 12m: "is US$422.09M (NGN612.22B)"
        m = re.search(
            r"past 12 months[^$]*?is\s+(US\$[\d.]+[KMB])\s*\(NGN([\d.]+[KMB])\)",
            raw, re.I,
        )
        if m:
            result["total_12m_usd"] = m.group(1)
            result["total_12m_ngn"] = f"NGN{m.group(2)}"
        # Monthly average: "average of US$35.17M (NGN51.02B) per month"
        m = re.search(
            r"average of\s+(US\$[\d.]+[KMB])\s*\(NGN([\d.]+[KMB])\)\s*per month",
            raw, re.I,
        )
        if m:
            result["monthly_avg_usd"] = m.group(1)
            result["monthly_avg_ngn"] = f"NGN{m.group(2)}"
        return result

    def _extract_pdf_info(self, soup: BeautifulSoup, doc: dict) -> dict:
        """Find the Google Drive iframe and (optionally) download/parse the PDF."""
        drive_id = None
        for frame in soup.find_all("iframe"):
            src = frame.get("src", "")
            did = _drive_id_from_src(src)
            if did:
                drive_id = did
                break

        info: dict[str, object] = {
            "drive_id": drive_id,
            "url": _drive_download_url(drive_id) if drive_id else None,
        }
        if not drive_id or not self.download_pdf:
            return info

        slug = doc.get("slug") or re.sub(r"\W+", "_", doc.get("url", "noslug"))
        pdf_path = PDF_DIR / f"{slug}.pdf"
        text_path = PDF_TEXT_DIR / f"{slug}.txt"

        size = _download_pdf(self.session, drive_id, pdf_path)
        if size is None:
            return info
        info["path"] = str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/")
        info["size_bytes"] = size

        if self.parse_pdf:
            excerpt, pages = _extract_pdf_text(pdf_path, text_path)
            if pages is not None:
                info["pages"] = pages
                info["text_path"] = str(text_path.relative_to(BASE_DIR)).replace("\\", "/")
                info["text_excerpt"] = excerpt
        return info

    # ── Legacy summary/KPI extraction (kept for backwards compat) ───

    @staticmethod
    def _legacy_summary_text(container: Tag) -> str:
        SKIP_PHRASES = {
            "sign up for email", "cookie", "privacy", "terms & conditions",
            "advertise", "help desk", "© 2008", "skip to content",
            "additional links", "recent documents", "recent news",
            "share price:", "email alerts",
        }
        parts: list[str] = []
        for elem in container.find_all(["h2", "h3", "h4", "p", "li"]):
            txt = elem.get_text(strip=True)
            if not txt:
                continue
            if "generated by an AI" in txt.lower():
                parts.append("\n[AI-generated summary]")
                break
            if any(skip in txt.lower() for skip in SKIP_PHRASES):
                continue
            if len(txt) < 5 and elem.name not in ("h2", "h3", "h4"):
                continue
            if elem.name in ("h2", "h3", "h4"):
                parts.append(f"\n## {txt}\n")
            elif elem.name == "li":
                parts.append(f"• {txt}")
            else:
                parts.append(txt)
        return "\n".join(parts).strip()

    @staticmethod
    def _legacy_kpis_regex(summary_text: str) -> dict[str, str]:
        kpis: dict[str, str] = {}
        patterns = [
            (r"(?:Group )?Revenue[:\s]+[a-z ]*?([₦N][\d.,]+\s*(?:billion|trillion|million)?)", "revenue"),
            (r"Profit (?:After|Before) Tax[:\s]+[a-z ]*?([₦N][\d.,]+\s*(?:billion|trillion|million)?)", "profit"),
            (r"(?:Group )?(?:Net )?Profit[:\s]+[a-z ]*?([₦N][\d.,]+\s*(?:billion|trillion|million)?)", "net_profit"),
            (r"(?:Group )?(?:Earnings|EPS)[\s]+[Pp]er [Ss]hare[:\s()EPS]*?([₦N]?[\d.,]+)", "eps"),
            (r"(?:Proposed )?Dividend[:\s]+[a-z ]*?([₦N]?[\d.,]+\s*(?:kobo)?(?:\s*per\s*share)?)", "dividend"),
            (r"Total Assets[:\s]+[a-z ]*?([₦N][\d.,]+\s*(?:billion|trillion|million)?)", "total_assets"),
            (r"EBITDA[:\s]+[a-z ]*?([₦N][\d.,]+\s*(?:billion|trillion|million)?)", "ebitda"),
            (r"Market Capitalis?ation[:\s]+[a-z ]*?([₦N][\d.,]+\s*(?:billion|trillion|million)?)", "market_cap"),
            (r"(?:Group )?Revenue (?:Growth|increased|grew)[:\s]*(\d+[%.]?\d*%?)", "revenue_growth"),
            (r"(?:Group )?(?:Net )?Profit (?:Growth|increased|grew|surged)[:\s]*(?:by )?(?:an )?(?:impressive )?(\d+[%.]?\d*%?)", "profit_growth"),
            (r"(?:Group )?EPS (?:increased|grew)[:\s]*(?:by )?(\d+[%.]?\d*%?)", "eps_growth"),
            (r"Revenue Growth[:\s]*(\d+%)", "revenue_growth"),
            (r"Net Profit Growth[:\s]*(\d+%)", "profit_growth"),
            (r"Earnings Per Share[:\s(EPS)]*[:\s]*([₦N]?[\d.,]+)", "eps"),
            (r"Proposed Dividend Per Share[:\s]*([₦N]?[\d.,]+)", "dividend"),
        ]
        for pattern, key in patterns:
            m = re.search(pattern, summary_text, re.IGNORECASE)
            if m and key not in kpis:
                kpis[key] = m.group(1).strip()
        return kpis

    @staticmethod
    def _extract_published_date(soup: BeautifulSoup) -> str:
        text = soup.get_text()
        m = re.search(r"Published[:\s]+(\w+ \d{1,2},?\s*\d{4})", text)
        if m:
            return m.group(1).strip()
        meta = soup.find("meta", {"property": "article:published_time"})
        if meta and meta.get("content"):
            return meta["content"]
        return ""


# ── KPI line splitter ────────────────────────────────────────────────

def _split_kpi_line(raw: str) -> tuple[str, str]:
    """
    Split a KPI <li> text into (name, value).

    Handles both:
      "Profit Before Tax: ₦1.23 trillion"          → ("Profit Before Tax", "₦1.23 trillion")
      "Profit Before Tax:₦1.23 trillion"           → same
      "Gross Written Premium Growth: 20%"          → ("Gross Written Premium Growth", "20%")
      "Share Price (31 Dec 2025): ₦3.79 ..."       → ("Share Price (31 Dec 2025)", "₦3.79 ...")
    Returns ("", "") if it can't be parsed.
    """
    if not raw or ":" not in raw:
        # Some li entries put name and value on separate lines joined by spaces
        return "", ""
    # Split on the FIRST ":" that isn't inside parentheses
    depth = 0
    split_at = -1
    for i, ch in enumerate(raw):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            split_at = i
            break
    if split_at < 0:
        return "", ""
    name = raw[:split_at].strip().rstrip(":")
    value = raw[split_at + 1:].strip()
    return name, value


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    test_tickers = sys.argv[1:] or ["GTCO", "DANGCEM", "MTNN"]

    scraper = ReportScraper(use_cache=False)
    reports = scraper.fetch_all(test_tickers)

    print(f"\n{'='*72}")
    print(f"Fetched {len(reports)} / {len(test_tickers)} reports")
    print(f"{'='*72}")

    for ticker, report in reports.items():
        print(f"\n> {ticker}: {report['title']}")
        print(f"  Type     : {report['doc_type_label']} ({report['year']})")
        print(f"  URL      : {report['url']}")

        kpis = report.get("kpis_structured") or {}
        if kpis:
            print(f"  KPIs     :")
            for k, v in kpis.items():
                print(f"    {k:40s}: {v}")

        liq = report.get("liquidity") or {}
        if liq.get("total_12m_usd"):
            print(f"  Liquidity: {liq.get('total_12m_usd')} / {liq.get('total_12m_ngn')} "
                  f"(avg {liq.get('monthly_avg_usd')} / mo, as of {liq.get('as_of','?')})")

        pdf = report.get("pdf") or {}
        if pdf.get("path"):
            print(f"  PDF      : {pdf['path']} "
                  f"({pdf.get('size_bytes', 0) // 1024} KB, {pdf.get('pages','?')} pages)")

        sections = report.get("sections") or {}
        if sections:
            print(f"  Sections : {', '.join(sections.keys())}")

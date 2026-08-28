"""
assessworth_pull.py — Interactive single-session scraper for Assessworth Premium.
=================================================================================
Auth state on Assessworth does NOT persist across Playwright context restarts.
This module sidesteps that by doing everything in one live session:

    python -m ingest.assessworth_pull --discover    # save raw HTML + screenshots
    python -m ingest.assessworth_pull               # full scrape (TODO once
                                                    # selectors are written)
    python -m ingest.assessworth_pull --insider-only
    python -m ingest.assessworth_pull --dividends-only

Flow:
  1. Open visible Chromium at app.assessworth.com.
  2. You log in via the visible window (Google OAuth or otherwise).
  3. Script polls for logged-in state. When detected, navigates to each target
     page and either dumps HTML (discover mode) or runs the extractors.
  4. Outputs to data/insider/ + data/dividends/.

Cadence: manual, e.g. weekly. NOT wired into daily_ingest.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
INSIDER_DIR = BASE_DIR / "data" / "insider"
DIVIDENDS_DIR = BASE_DIR / "data" / "dividends"
INSIDER_DIR.mkdir(parents=True, exist_ok=True)
DIVIDENDS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://app.assessworth.com"
DASHBOARD_URL = f"{BASE_URL}/dashboard"

# Real Assessworth URL paths aren't yet confirmed; the sidebar in the screenshots
# shows "Insider Trading" and "Dividend Corner" but the URL slugs may differ.
# Try each candidate; first one to render the expected anchor text wins.
INSIDER_URL_CANDIDATES = (
    f"{BASE_URL}/research/insider-trading",
    f"{BASE_URL}/research/insider-dealings",
    f"{BASE_URL}/research/insider",
)
DIVIDEND_URL_CANDIDATES = (
    f"{BASE_URL}/research/dividend-corner",
    f"{BASE_URL}/research/dividends",
    f"{BASE_URL}/research/dividend",
)

INSIDER_READY_TEXT = "Insider Dealings"          # H2/section header in screenshot
DIVIDEND_READY_TEXT = "Upcoming Eligible Dividends"

SCHEMA_VERSION = "2026-05-29"

LOGGED_IN_SELECTORS = (
    'input[placeholder*="Search for stocks"]',
    'a:has-text("Research")',
    'a:has-text("Dashboard")',
)


def _is_logged_in(page, timeout_ms: int = 2000) -> bool:
    for sel in LOGGED_IN_SELECTORS:
        try:
            if page.locator(sel).first.is_visible(timeout=timeout_ms):
                return True
        except Exception:
            pass
    return False


def _looks_like_login_url(url: str) -> bool:
    lower = url.lower().rstrip("/")
    if lower == BASE_URL.lower().rstrip("/"):
        return True
    return any(p in lower for p in ("/login", "/signin", "/auth", "/sign-in"))


def _wait_for_login(page, timeout_s: int = 900) -> bool:
    print(f"  Waiting up to {timeout_s}s for you to complete login in the browser…")
    for elapsed in range(timeout_s):
        time.sleep(1)
        if not _looks_like_login_url(page.url) and _is_logged_in(page):
            print(f"  -> Logged-in state detected at {page.url} (after {elapsed+1}s).")
            return True
        if elapsed and elapsed % 30 == 0:
            print(f"  …still waiting ({elapsed}s elapsed; current url: {page.url})")
    return False


def _try_candidate_urls(page, candidates, ready_text: str) -> str | None:
    """Navigate through candidate URLs until one renders the expected anchor.

    Returns the URL that worked, or None if none did. Saves us from needing to
    know the exact slug ahead of time — the SPA's own routing will surface 404
    or a "page not found" component otherwise.
    """
    for url in candidates:
        print(f"  Trying {url}")
        page.goto(url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except PWTimeoutError:
            pass
        time.sleep(2)
        try:
            page.get_by_text(ready_text, exact=False).first.wait_for(timeout=4000)
            print(f"  -> {ready_text!r} found at {url}")
            return url
        except PWTimeoutError:
            print(f"  …{ready_text!r} not found at {url}")
    return None


def _dump_page(page, out_html: Path, out_png: Path) -> None:
    html = page.content()
    out_html.write_text(html, encoding="utf-8")
    page.screenshot(path=str(out_png), full_page=False)
    print(f"  Saved HTML       ({len(html):,} chars) -> {out_html.name}")
    print(f"  Saved screenshot                   -> {out_png.name}")


def cmd_run(discover: bool, do_insider: bool, do_dividends: bool) -> int:
    print("=" * 64)
    print("  Assessworth Interactive Pull"
          + ("  [DISCOVER]" if discover else ""))
    print(f"  Schema version: {SCHEMA_VERSION}")
    print("=" * 64)
    print("  Opening visible browser for manual login...")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    with sync_playwright() as pw:
        # Instead of launching, connect to the user's running Chrome instance
        browser = pw.chromium.connect_over_cdp("http://localhost:9222")
        # For connect_over_cdp, context is usually browser.contexts[0]
        context = browser.contexts[0] if browser.contexts else browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        from playwright_stealth import Stealth
        Stealth().apply_stealth_sync(page)

        page.goto(BASE_URL)
        if not _is_logged_in(page) and not _wait_for_login(page):
            print("  ! Timed out waiting for login.")
            browser.close()
            return 2
        time.sleep(3)  # let dashboard settle

        if do_insider:
            print("\n  --- Insider Trading ---")
            url = _try_candidate_urls(page, INSIDER_URL_CANDIDATES, INSIDER_READY_TEXT)
            if url is None:
                print("  ! Could not locate the insider page from any candidate URL.")
            elif discover:
                _dump_page(
                    page,
                    INSIDER_DIR / f"_discover_insider_{timestamp}.html",
                    INSIDER_DIR / f"_discover_insider_{timestamp}.png",
                )
            else:
                print("  ! Insider extractor not implemented yet — run --discover first.")

        if do_dividends:
            print("\n  --- Dividend Corner ---")
            url = _try_candidate_urls(page, DIVIDEND_URL_CANDIDATES, DIVIDEND_READY_TEXT)
            if url is None:
                print("  ! Could not locate the dividend page from any candidate URL.")
            elif discover:
                _dump_page(
                    page,
                    DIVIDENDS_DIR / f"_discover_dividend_{timestamp}.html",
                    DIVIDENDS_DIR / f"_discover_dividend_{timestamp}.png",
                )
            else:
                print("  ! Dividend extractor not implemented yet — run --discover first.")

        print("\n  Done. Closing browser.")
        browser.close()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Assessworth scraper.")
    ap.add_argument("--discover", action="store_true",
                    help="Dump HTML + screenshot for selector development.")
    section = ap.add_mutually_exclusive_group()
    section.add_argument("--insider-only", action="store_true")
    section.add_argument("--dividends-only", action="store_true")
    args = ap.parse_args()

    do_insider = not args.dividends_only
    do_dividends = not args.insider_only
    sys.exit(cmd_run(discover=args.discover,
                     do_insider=do_insider,
                     do_dividends=do_dividends))


__all__ = ["BASE_URL", "SCHEMA_VERSION", "cmd_run"]


if __name__ == "__main__":
    main()

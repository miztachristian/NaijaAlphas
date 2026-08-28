"""
assessworth_session.py — Authenticated Playwright session for app.assessworth.com.
================================================================================
Manual-bootstrap + cached-reuse auth pattern, mirroring tv_login.py.

Assessworth Premium requires login; there is no public API. This module owns a
persistent browser profile that other ingest scripts reuse:

    python -m ingest.assessworth_session --login    # one-time, visible browser
    python -m ingest.assessworth_session --check    # headless verify
    python -m ingest.assessworth_session --logout   # wipe the cached profile

The --login command opens a real Chromium window at app.assessworth.com. You log
in (incl MFA); on detecting the post-login Dashboard the browser profile
(cookies, localStorage) is persisted at data/.assessworth_profile/. Subsequent
scrapers (assessworth_insider, assessworth_dividends) reuse that profile in
headless mode — no re-login until Assessworth's session expires.

Public API for scraper modules:

    from ingest.assessworth_session import open_session, AssessworthSessionExpired

    try:
        with open_session() as page:
            page.goto("https://app.assessworth.com/research/insider-trading")
            ...
    except AssessworthSessionExpired:
        ...   # daily_ingest skips with a clear log message
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
from pathlib import Path

from playwright.sync_api import (
    Page,
    TimeoutError as PWTimeoutError,
    sync_playwright,
)

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://app.assessworth.com"
DASHBOARD_URL = f"{BASE_URL}/dashboard"
PROFILE_DIR = Path(__file__).resolve().parent.parent / "data" / ".assessworth_profile"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# Selectors that should exist when logged in. Redundant on purpose — at least
# one should always match. Bump SCHEMA_VERSION whenever this set changes so
# downstream "schema drift" alerts can fire if the app redesigns.
SCHEMA_VERSION = "2026-05-27"
LOGGED_IN_SELECTORS = (
    'input[placeholder*="Search for stocks"]',
    'a:has-text("Research")',
    'a:has-text("Dashboard")',
)

VIEWPORT = {"width": 1440, "height": 900}
LOGIN_TIMEOUT_S = 300            # 5 min for the user to log in manually
CHECK_NAV_TIMEOUT_MS = 20_000
POST_LOGIN_GRACE_MS = 8_000      # give the SPA time to flush auth cookies to disk
NETWORK_IDLE_MS = 5_000
_LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]
_DEBUG = False


def _log(msg: str) -> None:
    if _DEBUG:
        print(f"  [aw-session] {msg}")


class AssessworthSessionExpired(Exception):
    """Raised when a cached profile no longer authenticates against Assessworth."""


def _is_logged_in(page: Page, timeout_ms: int = 2000) -> bool:
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


@contextlib.contextmanager
def open_session(headless: bool = True):
    """Yield an authenticated Page on app.assessworth.com.

    Loads the cached browser profile, opens the dashboard, and verifies the
    session is still valid. Raises AssessworthSessionExpired if not. The caller
    is responsible for navigating to whichever page(s) they actually want.

    The SPA may need a moment after page load for its auth handshake to settle,
    so we wait for networkidle and re-check before giving up.
    """
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=headless,
            viewport=VIEWPORT,
            args=_LAUNCH_ARGS,
        )
        page = ctx.new_page()
        try:
            _log(f"goto {BASE_URL}")
            page.goto(BASE_URL, wait_until="domcontentloaded",
                      timeout=CHECK_NAV_TIMEOUT_MS)
            try:
                page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_MS)
            except PWTimeoutError:
                _log("networkidle wait timed out (continuing)")
            _log(f"landed at {page.url}")

            # If we landed somewhere obviously authenticated, accept it.
            if not _looks_like_login_url(page.url) and _is_logged_in(page, timeout_ms=3000):
                _log("logged in on root page")
                yield page
                return

            # Otherwise try the dashboard explicitly — root may not auto-redirect.
            _log(f"root unclear, trying {DASHBOARD_URL}")
            page.goto(DASHBOARD_URL, wait_until="domcontentloaded",
                      timeout=CHECK_NAV_TIMEOUT_MS)
            try:
                page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_MS)
            except PWTimeoutError:
                _log("networkidle wait timed out on dashboard (continuing)")
            _log(f"landed at {page.url}")

            if _looks_like_login_url(page.url) or not _is_logged_in(page, timeout_ms=3000):
                raise AssessworthSessionExpired(
                    f"Cached session is no longer valid (landed at {page.url}). "
                    f"Run: python -m ingest.assessworth_session --login"
                )
            yield page
        finally:
            ctx.close()


def cmd_login(headless: bool = False) -> int:
    print("=" * 64)
    print("  Assessworth login (manual bootstrap)")
    print("=" * 64)
    print(f"  Profile dir: {PROFILE_DIR}")
    print(f"  Opening: {BASE_URL}")
    print(f"  You have up to {LOGIN_TIMEOUT_S}s to complete login (incl MFA).")
    print()

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=headless,
            viewport=VIEWPORT,
            args=_LAUNCH_ARGS,
        )
        page = ctx.new_page()
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)

        if _is_logged_in(page):
            print("  -> Already logged in (cached session still valid).")
            ctx.close()
            return 0

        print("  Log in via the browser window. Waiting for dashboard...")
        for _ in range(LOGIN_TIMEOUT_S):
            page.wait_for_timeout(1000)
            if not _looks_like_login_url(page.url) and _is_logged_in(page):
                print(f"  -> Detected logged-in state at {page.url}")
                # Wait long enough for the SPA to flush auth cookies / tokens to
                # disk before we tear the context down. The 1.5s default was too
                # short for OAuth flows where the auth cookie is set after the
                # dashboard renders.
                print(f"  Waiting {POST_LOGIN_GRACE_MS}ms for auth tokens to persist...")
                page.wait_for_timeout(POST_LOGIN_GRACE_MS)
                # Also let any in-flight network requests settle.
                try:
                    page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_MS)
                except PWTimeoutError:
                    pass
                ctx.close()
                print(f"\n  Session saved to: {PROFILE_DIR}")
                return 0

        ctx.close()
        print("\n  ! Timed out waiting for login. Try again.")
        return 2


def cmd_check() -> int:
    print(f"  Schema version: {SCHEMA_VERSION}")
    try:
        with open_session(headless=True) as page:
            url = page.url
        print(f"  OK — cached session valid (dashboard at {url}).")
        return 0
    except AssessworthSessionExpired as exc:
        print(f"  ! {exc}")
        return 2
    except PWTimeoutError as exc:
        print(f"  ! Navigation timed out: {exc}")
        return 3


def cmd_logout() -> int:
    if PROFILE_DIR.exists():
        shutil.rmtree(PROFILE_DIR)
        print(f"  Removed: {PROFILE_DIR}")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Assessworth Playwright session helper.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--login", action="store_true",
                     help="Visible browser for manual login (one-time bootstrap).")
    mode.add_argument("--check", action="store_true",
                     help="Headless verify the cached session is still valid.")
    mode.add_argument("--check-visible", action="store_true",
                     help="Same as --check but with a visible browser (diagnostic).")
    mode.add_argument("--logout", action="store_true",
                     help="Wipe the cached profile (forces re-login).")
    ap.add_argument("--debug", action="store_true",
                    help="Verbose URL transitions for troubleshooting.")
    args = ap.parse_args()

    global _DEBUG
    _DEBUG = args.debug

    if args.login:
        sys.exit(cmd_login(headless=False))
    if args.check:
        sys.exit(cmd_check())
    if args.check_visible:
        try:
            with open_session(headless=False) as page:
                print(f"  OK — visible check passed at {page.url}")
            sys.exit(0)
        except AssessworthSessionExpired as exc:
            print(f"  ! {exc}")
            sys.exit(2)
    if args.logout:
        sys.exit(cmd_logout())


__all__ = [
    "AssessworthSessionExpired",
    "BASE_URL",
    "PROFILE_DIR",
    "SCHEMA_VERSION",
    "open_session",
]


if __name__ == "__main__":
    main()

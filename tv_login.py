"""
tv_login.py — One-time TradingView login helper.

Logs into TradingView and persists the session to data/.tv_profile/. After this
runs once successfully, download_bulk_historical.py reuses the cached session.

Credentials are read from env vars (preferred) or CLI args. Nothing is written
to disk except the browser's own cookie store under data/.tv_profile/.

Usage (PowerShell):
    $env:TV_USER='you@example.com'; $env:TV_PASS='secret'; python tv_login.py
    # then unset:  Remove-Item Env:TV_USER, Env:TV_PASS
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

sys.stdout.reconfigure(encoding="utf-8")

PROFILE_DIR = Path(__file__).parent / "data" / ".tv_profile"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def already_logged_in(page) -> bool:
    """Detect a logged-in homepage."""
    for sel in (
        'button[aria-label="Open user menu"]',
        'button[aria-label*="user"]',
        'img.tv-header__user-menu-button--logged--avatar',
        'div[data-name="header-user-menu-button"]',
    ):
        try:
            if page.locator(sel).first.is_visible(timeout=800):
                return True
        except Exception:
            pass
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("TV_USER"))
    ap.add_argument("--password", default=os.environ.get("TV_PASS"))
    ap.add_argument("--headless", action="store_true",
                    help="Run headless (will fail if CAPTCHA appears)")
    args = ap.parse_args()

    if not args.user or not args.password:
        sys.exit("Provide credentials via TV_USER/TV_PASS env vars or --user/--password.")

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=args.headless,
            viewport={"width": 1440, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.new_page()

        page.goto("https://www.tradingview.com/", wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        if already_logged_in(page):
            print("Session already valid — no login needed.")
            ctx.close()
            return

        print("Opening signin page...")
        page.goto("https://www.tradingview.com/accounts/signin/", wait_until="domcontentloaded")
        page.wait_for_timeout(1500)

        # Click the 'Email' tab on the signin modal
        for sel in [
            'button:has-text("Email")',
            'span:has-text("Email")',
            'button[name="Email"]',
        ]:
            try:
                el = page.locator(sel).first
                if el.is_visible(timeout=2500):
                    el.click()
                    page.wait_for_timeout(700)
                    break
            except Exception:
                continue

        # Fill username + password
        try:
            page.fill(
                'input#id_username, input[name="username"], input[autocomplete="username"]',
                args.user,
                timeout=8000,
            )
        except PWTimeoutError:
            sys.exit("Could not find the username input. UI may have changed.")
        page.fill('input[type="password"]', args.password)
        page.wait_for_timeout(400)

        # Submit
        for sel in [
            'button[type="submit"]:has-text("Sign in")',
            'button:has-text("Sign in")',
            'button[type="submit"]',
        ]:
            try:
                page.locator(sel).first.click(timeout=2000)
                break
            except Exception:
                continue

        print("Submitted. Waiting up to 3 minutes for login to complete.")
        print("If a CAPTCHA shows up, solve it in the browser window.")

        for _ in range(180):
            page.wait_for_timeout(1000)
            url = page.url
            if "signin" not in url and "accounts" not in url:
                print(f"  -> Logged in. Now at: {url}")
                break
            if already_logged_in(page):
                print("  -> User menu detected — logged in.")
                break
        else:
            print("Login did not complete within 3 minutes. Aborting.")
            ctx.close()
            sys.exit(2)

        page.wait_for_timeout(2000)
        ctx.close()
        print(f"Session saved to: {PROFILE_DIR}")


if __name__ == "__main__":
    main()

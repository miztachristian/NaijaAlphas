"""
NGX Bulk Historical Data Downloader (via TradingView WebSocket interception)
============================================================================
Drives a real Chromium session against TradingView, opens each ticker's chart,
and intercepts the WebSocket frames that carry the OHLC bars.
No UI clicking, no Pro account needed.

Usage:
  python download_bulk_historical.py --symbols DANGCEM        # smoke test 1
  python download_bulk_historical.py --symbols DANGCEM,MTNN   # 2
  python download_bulk_historical.py                           # full universe
  python download_bulk_historical.py --headed                  # watch it run
  python download_bulk_historical.py --force                   # re-download cached

Output layout matches the JSE script:
  data/historical/<SYMBOL>.parquet   columns: Open, High, Low, Close, Volume
  Index: pandas DatetimeIndex named 'Date'.

Requires the persistent profile at data/.tv_profile to be already logged into
TradingView (run tv_login.py once if not).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.sync_api import (
    Page,
    TimeoutError as PWTimeoutError,
    sync_playwright,
)

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
HISTORY_DIR = DATA_DIR / "historical"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR = DATA_DIR / ".tv_profile"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

TV_EXCHANGE = "NSENG"  # TradingView code for the Nigerian Stock Exchange
TARGET_BARS = 1100      # ~5y daily; stop once we collect >= this many
COLLECT_TIMEOUT_S = 50
MAX_STALL_RETRIES = 2   # after this many no-progress retries, give up and save
MIN_ROWS_CACHED = 400   # MTNN/GTCO etc. have ~536 bars max from TradingView
DELAY_BETWEEN = 0.6

# WebSocket framing: ~m~<LEN>~m~<PAYLOAD>
_FRAME_RX = re.compile(r"~m~(\d+)~m~")


def _split_frames(text: str):
    """Yield each ~m~LEN~m~PAYLOAD payload from a TradingView WS chunk."""
    i = 0
    while i < len(text):
        m = _FRAME_RX.match(text, i)
        if not m:
            return
        length = int(m.group(1))
        start = m.end()
        end = start + length
        if end > len(text):
            return
        yield text[start:end]
        i = end


def _extract_bars_from_message(msg: dict) -> list[list]:
    """Pull OHLC bar rows out of a timescale_update / series_completed message."""
    out: list[list] = []
    if not isinstance(msg, dict):
        return out
    if msg.get("m") not in ("timescale_update", "du", "series_loading"):
        return out
    params = msg.get("p", [])
    if len(params) < 2 or not isinstance(params[1], dict):
        return out
    for _sid, sdata in params[1].items():
        if not isinstance(sdata, dict):
            continue
        bars = sdata.get("s") or []
        for bar in bars:
            v = bar.get("v") if isinstance(bar, dict) else None
            if v and len(v) >= 5:
                out.append(v)
    return out


def load_symbols(snapshot_date: str | None = None) -> list[str]:
    snaps = sorted((DATA_DIR / "snapshots").glob("*/snapshot.parquet"))
    if not snaps:
        raise SystemExit("No snapshots found under data/snapshots/. Run an ingest first.")
    target = snaps[-1] if snapshot_date is None else (
        DATA_DIR / "snapshots" / snapshot_date / "snapshot.parquet"
    )
    if not target.exists():
        raise SystemExit(f"Snapshot file not found: {target}")
    df = pd.read_parquet(target)
    return sorted(df["symbol"].dropna().unique().tolist())


def _accept_cookies(page: Page) -> None:
    for sel in ('button:has-text("Accept all")', 'button:has-text("Accept")'):
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=1500):
                btn.click()
                page.wait_for_timeout(400)
                return
        except Exception:
            pass


def _click_range_button(page: Page, label: str) -> bool:
    """Click a range button in the chart footer ('1Y', '5Y', 'ALL', etc.)."""
    for sel in (
        f'button:has-text("{label}")',
        f'div:has-text("{label}")',
    ):
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=1500):
                el.click()
                page.wait_for_timeout(700)
                return True
        except Exception:
            continue
    return False


def _drag_chart_left(page: Page, drags: int = 5) -> None:
    """Drag the chart canvas leftward to load older bars."""
    try:
        canvas = page.locator("canvas").first
        box = canvas.bounding_box()
        if not box:
            return
        cy = box["y"] + box["height"] / 2
        for _ in range(drags):
            x_start = box["x"] + box["width"] * 0.2
            x_end = box["x"] + box["width"] * 0.85
            page.mouse.move(x_start, cy)
            page.mouse.down()
            page.mouse.move(x_end, cy, steps=12)
            page.mouse.up()
            page.wait_for_timeout(700)
    except Exception:
        pass


def fetch_one(page: Page, symbol: str) -> tuple[bool, str, int]:
    """Open the chart, collect WebSocket bars, write parquet."""
    bars: list[list] = []

    def on_ws(ws):
        def on_frame(payload):
            try:
                text = payload.decode("utf-8", errors="ignore") if isinstance(payload, (bytes, bytearray)) else str(payload)
            except Exception:
                return
            for body in _split_frames(text):
                if not body.startswith("{"):
                    continue
                try:
                    msg = json.loads(body)
                except json.JSONDecodeError:
                    continue
                got = _extract_bars_from_message(msg)
                if got:
                    bars.extend(got)

        ws.on("framereceived", on_frame)

    page.on("websocket", on_ws)
    try:
        url = f"https://www.tradingview.com/chart/?symbol={TV_EXCHANGE}:{symbol}"
        try:
            page.goto(url, timeout=45_000, wait_until="domcontentloaded")
        except PWTimeoutError:
            return False, "page goto timeout", 0

        try:
            page.wait_for_selector("canvas", timeout=15_000)
        except PWTimeoutError:
            return False, "chart canvas not loaded", 0

        _accept_cookies(page)
        page.wait_for_timeout(1500)

        # Force daily interval.
        page.keyboard.press("Escape")
        page.keyboard.type("1D", delay=40)
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)

        # Ask the chart for a wider history. '5Y' triggers a server request for
        # ~5 years of bars in one go.
        _click_range_button(page, "5Y") or _click_range_button(page, "ALL")
        page.wait_for_timeout(1500)

        start = time.time()
        last_progress = time.time()
        last_count = 0
        stall_count = 0
        while True:
            page.wait_for_timeout(500)
            if len(bars) >= TARGET_BARS:
                break
            if time.time() - start > COLLECT_TIMEOUT_S:
                break
            if len(bars) > last_count:
                last_count = len(bars)
                last_progress = time.time()
                stall_count = 0
            elif time.time() - last_progress > 4.0:
                if stall_count >= MAX_STALL_RETRIES:
                    # Two drag-lefts produced no new bars: TradingView has no
                    # more history for this symbol. Save what we have.
                    break
                _drag_chart_left(page, drags=3)
                last_progress = time.time()
                stall_count += 1

        if not bars:
            return False, "no bars received", 0

        # Deduplicate by timestamp; keep last seen (most complete) row
        bymap: dict[int, list] = {}
        for v in bars:
            try:
                ts = int(v[0])
            except (TypeError, ValueError):
                continue
            bymap[ts] = v

        rows = []
        for ts, v in sorted(bymap.items()):
            rows.append({
                "Date": pd.to_datetime(ts, unit="s", utc=True).tz_localize(None),
                "Open":   float(v[1]) if len(v) > 1 and v[1] is not None else float("nan"),
                "High":   float(v[2]) if len(v) > 2 and v[2] is not None else float("nan"),
                "Low":    float(v[3]) if len(v) > 3 and v[3] is not None else float("nan"),
                "Close":  float(v[4]) if len(v) > 4 and v[4] is not None else float("nan"),
                "Volume": float(v[5]) if len(v) > 5 and v[5] is not None else float("nan"),
            })

        df = pd.DataFrame(rows).set_index("Date")

        # Drop placeholder rows (TradingView sends Open=High=Low=Close=0 for
        # non-trading days before the symbol's listing).
        if not df.empty:
            mask = (df[["Open", "High", "Low", "Close"]].sum(axis=1) > 0)
            df = df.loc[mask]

        out_path = HISTORY_DIR / f"{symbol}.parquet"
        df.to_parquet(out_path)
        return True, "OK", len(df)
    finally:
        # detach listener to avoid leaking handlers across symbols
        try:
            page.remove_listener("websocket", on_ws)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="Comma-separated subset (default: all in latest snapshot)")
    ap.add_argument("--snapshot-date", help="YYYY-MM-DD snapshot to read symbols from (default: latest)")
    ap.add_argument("--headed", action="store_true", help="Show the browser (default: headless)")
    ap.add_argument("--force", action="store_true", help="Re-download even if parquet cached")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbols(args.snapshot_date)

    print("=" * 70)
    print(f"NGX Bulk Historical Download — {len(symbols)} symbols")
    print(f"Profile dir: {PROFILE_DIR}")
    print(f"Output dir : {HISTORY_DIR}")
    print(f"Headed     : {args.headed}")
    print("=" * 70)

    results = []
    success = failed = skipped = 0

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=not args.headed,
            viewport={"width": 1440, "height": 900},
            accept_downloads=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.new_page()
        page.goto("https://www.tradingview.com/", wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        _accept_cookies(page)

        for i, sym in enumerate(symbols, 1):
            out_path = HISTORY_DIR / f"{sym}.parquet"
            if out_path.exists() and not args.force:
                try:
                    existing = pd.read_parquet(out_path)
                    if len(existing) >= MIN_ROWS_CACHED:
                        success += 1
                        skipped += 1
                        results.append({"symbol": sym, "status": "CACHED", "rows": len(existing)})
                        continue
                except Exception:
                    pass

            ok, msg, n = fetch_one(page, sym)
            if ok:
                success += 1
                results.append({"symbol": sym, "status": "OK", "rows": n})
                print(f"  [{i:>3d}/{len(symbols)}] {sym:12s} | {n:>5d} bars")
            else:
                failed += 1
                results.append({"symbol": sym, "status": "FAIL", "rows": 0, "msg": msg})
                print(f"  [{i:>3d}/{len(symbols)}] {sym:12s} | FAIL — {msg}")

            time.sleep(DELAY_BETWEEN)

        ctx.close()

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print(f"  Total:   {len(symbols)}")
    print(f"  Success: {success} ({skipped} cached)")
    print(f"  Failed:  {failed}")
    print(f"  Saved:   {HISTORY_DIR}")
    print("=" * 70)

    log = pd.DataFrame(results)
    log_path = HISTORY_DIR / f"bulk_download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log.to_csv(log_path, index=False)
    print(f"Log: {log_path}")

    if failed:
        fails = log[log["status"] == "FAIL"]["symbol"].tolist()
        print(f"\nFailed ({len(fails)}): {', '.join(fails)}")


if __name__ == "__main__":
    main()

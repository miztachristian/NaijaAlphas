"""
NGX Disclosure Event Analyzer
==============================
Turns the raw NGX corporate-disclosure feed (``ingest/fetch_disclosures.py``)
into per-company *event signals*: earnings catalysts, insider activity,
dividend events and filing-health / late-filer flags.

Unlike ``ReportAnalyzer`` — which scores annual-report prose — this analyzer
works purely from disclosure event metadata (company, category, date, title).
No PDF parsing is involved, so it is fast and fully offline once the
disclosure cache exists.

Signals produced per ticker:
- earnings_catalyst   — a board-meeting notice or fresh results filing is due
                        / just landed (a short-term-sleeve catalyst)
- insider_activity    — recent director-dealings filings (count + score)
- dividend_recent     — a dividend was declared this cycle
- late_filer          — the company filed a delay notice, or its last
                        financial statement is overdue (a robustness flag)
- recency context     — days_since_results, last_results_date, etc.

Usage:
    from analysis.disclosure_analyzer import DisclosureAnalyzer
    da = DisclosureAnalyzer()
    signals = da.analyze("JAIZBANK")
    batch   = da.analyze_batch(["JAIZBANK", "GTCO", "DANGCEM"])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DISCLOSURES_DIR = BASE_DIR / "data" / "disclosures"

# ── Signal windows (days) ────────────────────────────────────────────
CATALYST_BOARD_WINDOW = 21     # board-meeting notice => earnings imminent
CATALYST_RESULTS_WINDOW = 14   # a statement filed this recently is "fresh"
INSIDER_WINDOW = 90            # look-back for director-dealings activity
DIVIDEND_WINDOW = 120          # look-back for a dividend declaration
STALE_RESULTS_DAYS = 150       # no statement in this long => overdue (a quarter)
DELAY_NOTICE_WINDOW = 365      # honour an explicit delay notice for a year


def _parse_dt(iso: str) -> Optional[datetime]:
    """Parse a SharePoint ISO timestamp ('2026-05-22T12:05:06Z') to UTC."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_disclosure_adjustment(signals: dict) -> tuple[float, list[str]]:
    """Capped, sign-unambiguous growth-score adjustment from disclosure signals.

    Single source of truth shared by `analysis/growth.py` (folds it into
    `growth_score`) and the notebooks (display the would-be impact). Only
    sign-clear signals count: a late/overdue filer penalises, a recent
    dividend gives a small bonus. Catalyst and insider-activity flags are
    excluded — their sign is ambiguous. Returns (points, reasons).
    """
    from config.settings import DisclosureScoring as cfg
    if not cfg.ENABLED or not signals:
        return 0.0, []

    adjustment = 0.0
    reasons: list[str] = []

    if signals.get("late_filer"):
        adjustment += cfg.LATE_FILER_PENALTY
        reasons.append(f"late filer ({cfg.LATE_FILER_PENALTY:+g})")

    days = signals.get("days_since_results")
    if days is not None and days > cfg.STALE_DAYS:
        adjustment += cfg.STALE_RESULTS_PENALTY
        reasons.append(f"results {days}d old ({cfg.STALE_RESULTS_PENALTY:+g})")

    if signals.get("dividend_recent"):
        adjustment += cfg.DIVIDEND_BONUS
        reasons.append(f"recent dividend ({cfg.DIVIDEND_BONUS:+g})")

    adjustment = max(-cfg.ADJUSTMENT_CAP, min(cfg.ADJUSTMENT_CAP, adjustment))
    return round(adjustment, 2), reasons


class DisclosureAnalyzer:
    """Derives per-company event signals from the NGX disclosure cache."""

    def __init__(self, rows: Optional[list[dict]] = None,
                 as_of: Optional[datetime] = None):
        """
        Parameters
        ----------
        rows  : disclosure records (as produced by ingest.fetch_disclosures).
                If None, the most recent data/disclosures/<date>.json is loaded.
        as_of : reference "now" for recency windows; defaults to current UTC.
        """
        self.as_of = as_of or datetime.now(timezone.utc)
        self._rows = rows if rows is not None else self._load_latest_cache()
        # Index rows by ticker once, newest first.
        self._by_ticker: dict[str, list[dict]] = {}
        for r in self._rows:
            self._by_ticker.setdefault(r.get("ticker", ""), []).append(r)
        for lst in self._by_ticker.values():
            lst.sort(key=lambda r: r.get("created", ""), reverse=True)
        # Parsed earnings forecasts (ingest/parse_forecasts.py), if available.
        self._forecasts = self._load_latest_json("forecasts-*.json")

    # ── cache loading ────────────────────────────────────────────────

    @staticmethod
    def _load_latest_cache() -> list[dict]:
        """Load the newest dated disclosure cache; empty list if none exist."""
        if not DISCLOSURES_DIR.exists():
            logger.warning("[disclosure_analyzer] no data/disclosures/ dir")
            return []
        files = sorted(DISCLOSURES_DIR.glob("[0-9]*.json"), reverse=True)
        if not files:
            logger.warning("[disclosure_analyzer] no disclosure cache found")
            return []
        try:
            with open(files[0], encoding="utf-8") as f:
                rows = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[disclosure_analyzer] bad cache %s: %s", files[0], exc)
            return []
        # Backfill `category` for caches written before that field existed.
        from ingest.fetch_disclosures import _categorise
        for r in rows:
            r.setdefault("category", _categorise(r.get("type", "")))
        return rows

    @staticmethod
    def _load_latest_json(pattern: str) -> dict:
        """Load the newest data/disclosures/<pattern> JSON; {} if none."""
        if not DISCLOSURES_DIR.exists():
            return {}
        files = sorted(DISCLOSURES_DIR.glob(pattern), reverse=True)
        if not files:
            return {}
        try:
            with open(files[0], encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[disclosure_analyzer] bad %s: %s", files[0], exc)
            return {}

    # ── public API ───────────────────────────────────────────────────

    def analyze(self, ticker: str) -> dict:
        """Return the event-signal dict for one ticker."""
        rows = self._by_ticker.get(ticker.upper(), [])
        if not rows:
            signals = self._empty_signals(ticker)
            signals.update(self._forecast_for(ticker))
            return signals

        days = {id(r): self._days_ago(r) for r in rows}

        def recent(category: str, window: int) -> list[dict]:
            return [r for r in rows
                    if r.get("category") == category
                    and days[id(r)] is not None and days[id(r)] <= window]

        # ── earnings catalyst ───────────────────────────────────────
        board = recent("board_meeting", CATALYST_BOARD_WINDOW)
        fresh_results = recent("financial_statement", CATALYST_RESULTS_WINDOW)
        catalyst = bool(board or fresh_results)
        if fresh_results:
            catalyst_reason = f"Results filed {fresh_results[0]['created'][:10]}"
        elif board:
            catalyst_reason = f"Board meeting notice {board[0]['created'][:10]}"
        else:
            catalyst_reason = ""

        # ── insider activity ────────────────────────────────────────
        insider = recent("directors_dealings", INSIDER_WINDOW)
        insider_count = len(insider)
        insider_score = round(min(insider_count / 4.0, 1.0), 3)
        last_insider = insider[0]["created"][:10] if insider else ""

        # ── dividend ────────────────────────────────────────────────
        dividends = [r for r in rows
                     if "DIVIDEND" in r.get("title", "").upper()
                     and days[id(r)] is not None and days[id(r)] <= DIVIDEND_WINDOW]
        dividend_recent = bool(dividends)
        last_dividend = dividends[0]["created"][:10] if dividends else ""

        # ── filing health / late filer ──────────────────────────────
        statements = [r for r in rows if r.get("category") == "financial_statement"]
        last_results_date = statements[0]["created"][:10] if statements else ""
        days_since_results = days[id(statements[0])] if statements else None

        delay_notice = next(
            (r for r in rows
             if "DELAY IN FILING" in r.get("title", "").upper()
             and days[id(r)] is not None and days[id(r)] <= DELAY_NOTICE_WINDOW),
            None,
        )
        stale = days_since_results is not None and days_since_results > STALE_RESULTS_DAYS
        late_filer = bool(delay_notice) or stale
        if delay_notice:
            late_filer_reason = f"Delay notice {delay_notice['created'][:10]}"
        elif stale:
            late_filer_reason = f"No results for {days_since_results}d"
        else:
            late_filer_reason = ""

        disclosure_count_90d = sum(
            1 for r in rows if days[id(r)] is not None and days[id(r)] <= 90
        )

        return {
            "ticker": ticker.upper(),
            "earnings_catalyst": catalyst,
            "catalyst_reason": catalyst_reason,
            "insider_activity": insider_count > 0,
            "insider_count_90d": insider_count,
            "insider_score": insider_score,
            "last_insider_date": last_insider,
            "dividend_recent": dividend_recent,
            "last_dividend_date": last_dividend,
            "late_filer": late_filer,
            "late_filer_reason": late_filer_reason,
            "days_since_results": days_since_results,
            "last_results_date": last_results_date,
            "disclosure_count_90d": disclosure_count_90d,
            **self._forecast_for(ticker),
        }

    def analyze_batch(self, tickers: list[str]) -> dict[str, dict]:
        """Return {ticker: signal dict} for a list of tickers."""
        return {t.upper(): self.analyze(t) for t in tickers}

    # ── helpers ──────────────────────────────────────────────────────

    def _days_ago(self, row: dict) -> Optional[int]:
        dt = _parse_dt(row.get("created", ""))
        if dt is None:
            return None
        return (self.as_of - dt).days

    def _forecast_for(self, ticker: str) -> dict:
        """Earnings-forecast fields for a ticker.

        Forecast figures are exposed only when the parse is trustworthy
        ('ok' or 'partial'); 'suspect' / 'unparseable' yield None values
        but the status is still reported.
        """
        fc = self._forecasts.get(ticker.upper())
        trusted = bool(fc) and fc.get("parse_status") in ("ok", "partial")
        return {
            "forecast_status": fc.get("parse_status", "none") if fc else "none",
            "forecast_period": fc.get("period", "") if trusted else "",
            "forecast_revenue": fc.get("forecast_revenue") if trusted else None,
            "forecast_pbt": fc.get("forecast_pbt") if trusted else None,
            "forecast_pat": fc.get("forecast_pat") if trusted else None,
        }

    @staticmethod
    def _empty_signals(ticker: str) -> dict:
        """Neutral signal dict for a ticker with no disclosures on file."""
        return {
            "ticker": ticker.upper(),
            "earnings_catalyst": False,
            "catalyst_reason": "",
            "insider_activity": False,
            "insider_count_90d": 0,
            "insider_score": 0.0,
            "last_insider_date": "",
            "dividend_recent": False,
            "last_dividend_date": "",
            "late_filer": False,
            "late_filer_reason": "",
            "days_since_results": None,
            "last_results_date": "",
            "disclosure_count_90d": 0,
        }


__all__ = ["DisclosureAnalyzer", "compute_disclosure_adjustment"]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    da = DisclosureAnalyzer()
    targets = sys.argv[1:] or ["JAIZBANK", "GTCO", "DANGCEM", "MTNN"]
    for t in targets:
        s = da.analyze(t)
        print(f"\n{t}")
        print("-" * 50)
        for k, v in s.items():
            if k != "ticker":
                print(f"  {k:<22} {v}")

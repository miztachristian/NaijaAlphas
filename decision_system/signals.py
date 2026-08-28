"""
signals.py — collect every analysis signal for a stock and normalize each to
a 0-100 sub-score.

This is the single source of truth for "what is a signal". It *orchestrates*
the existing analyzers in analysis/ (via a GrowthAnalyzer, whose wiring and
caches it reuses) and adds the two signals the growth engine never used:
news sentiment and a cross-sectional market-context score.

A signal that has no underlying data returns NaN — never 0 — so the conviction
engine can exclude it from the weighted average instead of biasing the stock
down (the `.fillna(0)` fix from the plan).
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import DATA_DIR, AnalysisConfig, get_sector_outlook, get_ticker_sector
from analysis.growth import GrowthAnalyzer
from analysis import quality_scores
from analysis.sentiment import aggregate_sentiment
from decision_system.models import SignalSet


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _finite(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


class SignalCollector:
    """Builds a SignalSet (8 normalized sub-scores) for any ticker."""

    def __init__(self, snapshot_df: Optional[pd.DataFrame] = None,
                 news_date: Optional[str] = None):
        # GrowthAnalyzer gives us pre-wired analyzers + the report cache.
        self.ga = GrowthAnalyzer()

        self._news = self._load_news(news_date)
        self._quality_rows: dict = {}
        self._momentum_pct: dict = {}
        if snapshot_df is not None and not snapshot_df.empty:
            self._prepare_snapshot(snapshot_df)

    # --- preparation ------------------------------------------------------
    def _load_news(self, news_date: Optional[str]) -> dict:
        """Latest (or dated) data/news/<date>.json — {ticker: [articles]}."""
        news_dir = DATA_DIR / "news"
        if not news_dir.exists():
            return {}
        if news_date:
            path = news_dir / f"{news_date}.json"
        else:
            files = sorted(glob.glob(str(news_dir / "*.json")))
            path = Path(files[-1]) if files else None
        if not path or not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _prepare_snapshot(self, df: pd.DataFrame) -> None:
        """Enrich the snapshot with quality scores and a momentum percentile."""
        df = df.copy()
        sym_col = "symbol" if "symbol" in df.columns else df.columns[0]

        # Quality sub-scores (earnings quality / balance sheet / consistency).
        try:
            enriched = quality_scores.enrich(df)
            for _, row in enriched.iterrows():
                self._quality_rows[str(row[sym_col])] = row.to_dict()
        except Exception:
            self._quality_rows = {}

        # Cross-sectional momentum percentile for market context.
        perf_cols = [c for c in ("perf_3m", "perf_6m", "perf_1m", "perf_1y")
                     if c in df.columns]
        if perf_cols:
            blended = df[perf_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            pct = blended.rank(pct=True) * 100.0
            for sym, p in zip(df[sym_col], pct):
                if _finite(p):
                    self._momentum_pct[str(sym)] = float(p)

    # --- individual signals (return None when no data) --------------------
    def _fundamental(self, ticker: str) -> Optional[float]:
        fm = self.ga.fundamental_analyzer.analyze(ticker)
        key = [fm.pe_ratio, fm.roe, fm.earnings_growth, fm.revenue_growth,
               fm.debt_to_equity, fm.profit_margin]
        if sum(1 for v in key if _finite(v)) == 0:
            return None
        scores = self.ga.fundamental_analyzer.score_fundamentals(fm)
        return _clamp(self.ga.fundamental_analyzer.get_composite_score(scores) * 100.0)

    def _technical(self, ticker: str) -> Optional[float]:
        prices = self.ga.loader.get_price_series(ticker)
        if prices is None or len(prices) < AnalysisConfig.MIN_DATA_POINTS:
            return None
        tm = self.ga.technical_analyzer.analyze(ticker)
        scores = self.ga.technical_analyzer.score_technicals(tm)
        return _clamp(self.ga.technical_analyzer.get_composite_score(scores) * 100.0)

    def _quality(self, ticker: str) -> Optional[float]:
        row = self._quality_rows.get(ticker)
        if not row:
            return None
        vals = [row.get("earnings_quality_score"), row.get("balance_sheet_score"),
                row.get("consistency_score")]
        vals = [float(v) for v in vals if _finite(v)]
        if not vals:
            return None
        return _clamp(sum(vals) / len(vals))

    def _seasonality(self, ticker: str) -> Optional[float]:
        sm = self.ga.seasonal_analyzer.analyze(ticker)
        if sm is None or getattr(sm, "years_of_data", 0) < 1:
            return None
        ss = self.ga.seasonal_analyzer.score_seasonal(sm)
        val = (ss.get("seasonal_current_score", 0.5) * 0.70 +
               ss.get("seasonal_consistency_score", 0.5) * 0.30)
        return _clamp(val * 100.0)

    def _disclosure(self, ticker: str) -> Optional[float]:
        sig = self.ga.disclosure_analyzer.analyze(ticker) or {}
        has_data = (sig.get("days_since_results") is not None
                    or sig.get("earnings_catalyst")
                    or sig.get("insider_count_90d", 0)
                    or sig.get("dividend_recent")
                    or sig.get("late_filer"))
        if not has_data:
            return None
        score = 50.0
        if sig.get("earnings_catalyst"):
            score += 12.0
        score += float(sig.get("insider_score", 0.0) or 0.0) * 20.0
        if sig.get("dividend_recent"):
            score += 8.0
        if sig.get("late_filer"):
            score -= 25.0
        dsr = sig.get("days_since_results")
        if dsr is not None and dsr > 200:
            score -= 12.0
        return _clamp(score)

    def _report_tone(self, ticker: str) -> Optional[float]:
        tone_score, label = self.ga._get_report_tone(ticker)
        if label in (None, "N/A"):
            return None
        return _clamp((tone_score + 1.0) / 2.0 * 100.0)

    def _sentiment(self, ticker: str) -> Optional[float]:
        articles = self._news.get(ticker) or []
        if not articles:
            return None
        agg = aggregate_sentiment(articles)
        if agg.get("total", 0) == 0:
            return None
        return _clamp((agg["avg_score"] + 1.0) / 2.0 * 100.0)

    def _market_context(self, ticker: str, sector: str) -> Optional[float]:
        outlook = get_sector_outlook(AnalysisConfig.TARGET_YEAR).get(sector, 0.5)
        sector_pts = outlook * 100.0
        mom_pct = self._momentum_pct.get(ticker)
        if mom_pct is None:
            return _clamp(sector_pts)
        return _clamp(0.5 * sector_pts + 0.5 * mom_pct)

    # --- public -----------------------------------------------------------
    def collect(self, ticker: str, sector: Optional[str] = None) -> SignalSet:
        """Collect all eight signals for `ticker` into a SignalSet."""
        sector = sector or get_ticker_sector(ticker)
        ss = SignalSet(ticker=ticker, sector=sector)

        computations = {
            "fundamental": lambda: self._fundamental(ticker),
            "technical": lambda: self._technical(ticker),
            "quality": lambda: self._quality(ticker),
            "seasonality": lambda: self._seasonality(ticker),
            "disclosure": lambda: self._disclosure(ticker),
            "report_tone": lambda: self._report_tone(ticker),
            "sentiment": lambda: self._sentiment(ticker),
            "market_context": lambda: self._market_context(ticker, sector),
        }

        for name, fn in computations.items():
            try:
                value = fn()
                ss.sub_scores[name] = float(value) if value is not None else float("nan")
            except Exception as exc:  # noqa: BLE001 - one signal failing is non-fatal
                ss.sub_scores[name] = float("nan")
                ss.errors[name] = str(exc)

        return ss

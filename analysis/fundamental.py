"""
Fundamental Analysis Module
===========================
Analyzes financial health, valuation, profitability, and growth metrics.
Fundamental data is sourced from TradingView screener snapshots.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import NGXDataLoader
from core.models import FundamentalMetrics
from config.settings import DisclosureScoring

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """
    Fundamental analysis engine for NGX stocks.
    
    Analyzes:
    - Valuation metrics (P/E, P/B, PEG)
    - Profitability (ROE, ROA, margins)
    - Growth (revenue, earnings)
    - Financial health (debt, liquidity)
    - Dividend characteristics
    """
    
    def __init__(self, loader: NGXDataLoader = None):
        """
        Initialize analyzer.

        Args:
            loader: Data loader instance (creates new if not provided)
        """
        self.loader = loader or NGXDataLoader()
        # Official NGX financial-statement figures (ingest/parse_statements.py),
        # used to enrich TradingView snapshot fundamentals.
        self._statements = self._load_latest_statements()

    @staticmethod
    def _load_latest_statements() -> dict:
        """Load the newest data/disclosures/statements-*.json; {} if none."""
        sdir = Path(__file__).resolve().parent.parent / "data" / "disclosures"
        if not sdir.exists():
            return {}
        files = sorted(sdir.glob("statements-*.json"), reverse=True)
        if not files:
            return {}
        try:
            with open(files[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[fundamental] bad statements cache: %s", exc)
            return {}
    
    def analyze(self, ticker: str) -> FundamentalMetrics:
        """
        Perform complete fundamental analysis on a stock.
        
        Args:
            ticker: NGX ticker symbol
        
        Returns:
            FundamentalMetrics with all calculated values
        """
        metrics = FundamentalMetrics(ticker=ticker)
        
        # Get fundamental data from the latest TradingView snapshot
        info = self.loader.get_snapshot_fundamentals(ticker)
        
        if not info:
            return metrics
        
        # Extract and process metrics
        self._extract_valuation(metrics, info)
        self._extract_profitability(metrics, info)
        self._extract_growth(metrics, info)
        self._extract_financial_health(metrics, info)
        self._extract_dividends(metrics, info)
        self._extract_size_liquidity(metrics, info)

        # Overlay official NGX statement figures where available
        self._apply_ngx_statement(metrics, ticker)

        return metrics

    def _apply_ngx_statement(self, metrics: FundamentalMetrics, ticker: str):
        """Attach official NGX statement figures as an overlay.

        Computes ROE, ROA, profit margin and growth from the latest parsed
        NGX financial statement and stores them in ``metrics.ngx_figures``
        alongside the TradingView snapshot metrics.
        """
        stmt = self._statements.get(ticker.upper())
        if not stmt or stmt.get("parse_status") not in ("ok", "partial"):
            return

        rev = stmt.get("revenue")
        pat = stmt.get("pat")
        equity = stmt.get("total_equity")
        assets = stmt.get("total_assets")

        ngx: Dict[str, float] = {}
        if stmt.get("revenue_growth") is not None:
            ngx["revenue_growth"] = stmt["revenue_growth"]
        if stmt.get("pat_growth") is not None:
            ngx["earnings_growth"] = stmt["pat_growth"]
        if pat and equity and equity > 0:
            ngx["roe"] = round(pat / equity * 100, 2)
        if pat and assets and assets > 0:
            ngx["roa"] = round(pat / assets * 100, 2)
        if pat and rev and rev > 0:
            ngx["profit_margin"] = round(pat / rev * 100, 2)
        if not ngx:
            return

        # Stored as an overlay (ngx_* keys); does not feed score_fundamentals.
        metrics.ngx_figures = {f"ngx_{k}": v for k, v in ngx.items()}
    
    @staticmethod
    def _safe_float(val, default=np.nan) -> float:
        """Safely convert a value to float."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _extract_valuation(self, metrics: FundamentalMetrics, info: Dict):
        """Extract valuation metrics from TradingView snapshot."""
        metrics.pe_ratio = self._safe_float(info.get("pe_ratio"))
        metrics.forward_pe = self._safe_float(info.get("forward_pe"))  # may not exist in TV
        metrics.pb_ratio = self._safe_float(info.get("pb_ratio"))
        metrics.ps_ratio = self._safe_float(info.get("ps_ratio"))
        metrics.peg_ratio = self._safe_float(info.get("peg_ratio"))
        metrics.ev_ebitda = self._safe_float(info.get("ev_ebitda"))
    
    def _extract_profitability(self, metrics: FundamentalMetrics, info: Dict):
        """Extract profitability metrics from TradingView snapshot.
        
        TradingView already reports these as percentages (e.g. 25.5 = 25.5%).
        """
        metrics.roe = self._safe_float(info.get("roe_ttm"))
        metrics.roa = self._safe_float(info.get("roa_ttm"))
        metrics.profit_margin = self._safe_float(info.get("net_margin_ttm"))
        metrics.operating_margin = self._safe_float(info.get("operating_margin_ttm"))
        metrics.gross_margin = self._safe_float(info.get("gross_margin_ttm"))
    
    def _extract_growth(self, metrics: FundamentalMetrics, info: Dict):
        """Extract growth metrics from TradingView snapshot.
        
        TradingView already reports these as percentages.
        """
        metrics.revenue_growth = self._safe_float(info.get("revenue_growth_ttm"))
        metrics.earnings_growth = self._safe_float(info.get("eps_growth_ttm"))
        metrics.eps_growth_5y = self._safe_float(info.get("eps_growth_qoq"))
        metrics.revenue_growth_5y = self._safe_float(info.get("revenue_growth_qoq"))
    
    def _extract_financial_health(self, metrics: FundamentalMetrics, info: Dict):
        """Extract financial health metrics from TradingView snapshot."""
        metrics.debt_to_equity = self._safe_float(info.get("debt_to_equity"))
        metrics.current_ratio = self._safe_float(info.get("current_ratio"))
        metrics.quick_ratio = self._safe_float(info.get("quick_ratio"))
    
    def _extract_dividends(self, metrics: FundamentalMetrics, info: Dict):
        """Extract dividend metrics from TradingView snapshot.
        
        TradingView dividend_yield is already in percentage form.
        """
        metrics.dividend_yield = self._safe_float(info.get("dividend_yield"), default=0.0)
        metrics.payout_ratio = self._safe_float(info.get("dividend_payout"))
    
    def _extract_size_liquidity(self, metrics: FundamentalMetrics, info: Dict):
        """Extract size and liquidity metrics from TradingView snapshot."""
        metrics.market_cap = self._safe_float(info.get("market_cap"))
        metrics.avg_volume = self._safe_float(info.get("avg_volume_10d",
                                info.get("avg_volume_90d",
                                info.get("volume_1d"))))
        metrics.beta = 1.0  # TradingView does not provide beta for NGX
    
    @staticmethod
    def _effective(metrics: FundamentalMetrics, key: str) -> float:
        """Scoring input for `key`, preferring the NGX-official figure.

        NGX statement figures (ingest/parse_statements.py, parse_status ok)
        are more reliable for Nigerian equities than the TradingView snapshot.
        Falls back to the snapshot value when no NGX figure is present, or
        when DisclosureScoring.USE_NGX_FUNDAMENTALS is off.
        """
        if DisclosureScoring.USE_NGX_FUNDAMENTALS:
            ngx_val = metrics.ngx_figures.get(f"ngx_{key}")
            if ngx_val is not None:
                return ngx_val
        return getattr(metrics, key)

    def score_fundamentals(self, metrics: FundamentalMetrics) -> Dict[str, float]:
        """
        Score fundamental metrics on 0-1 scale.

        Higher scores indicate better fundamentals for growth investing.
        ROE, growth and margin are scored from NGX-official figures when
        available (see _effective); other metrics use the snapshot.

        Args:
            metrics: FundamentalMetrics object

        Returns:
            Dictionary of scores for each metric category
        """
        scores = {}
        
        # P/E Score (lower is better for value, moderate for growth)
        scores["pe_score"] = self._score_pe(metrics.pe_ratio)
        
        # PEG Score (lower is better, < 1 is undervalued)
        scores["peg_score"] = self._score_peg(metrics.peg_ratio)
        
        # ROE Score (higher is better) — NGX-official figure preferred
        scores["roe_score"] = self._score_roe(self._effective(metrics, "roe"))

        # ROA Score (higher is better)
        scores["roa_score"] = self._score_roa(metrics.roa)

        # Earnings Growth Score (higher is better) — NGX-official preferred
        scores["earnings_growth_score"] = self._score_earnings_growth(
            self._effective(metrics, "earnings_growth"))

        # Revenue Growth Score (higher is better) — NGX-official preferred
        scores["revenue_growth_score"] = self._score_revenue_growth(
            self._effective(metrics, "revenue_growth"))
        
        # Debt/Equity Score (lower is better for growth)
        scores["debt_score"] = self._score_debt(metrics.debt_to_equity)
        
        # Current Ratio Score (moderate is best)
        scores["liquidity_score"] = self._score_liquidity(metrics.current_ratio)
        
        # Dividend Score (moderate yield with growth potential)
        scores["dividend_score"] = self._score_dividend(metrics.dividend_yield)
        
        # Profit Margin Score (higher is better) — NGX-official preferred
        scores["margin_score"] = self._score_margin(
            self._effective(metrics, "profit_margin"))
        
        return scores
    
    def _score_pe(self, pe: float) -> float:
        """Score P/E ratio"""
        if np.isnan(pe) or pe <= 0:
            return 0.3
        if pe < 5:
            return 0.7  # Might be value trap
        elif pe < 10:
            return 0.9  # Good value
        elif pe < 15:
            return 1.0  # Ideal
        elif pe < 25:
            return 0.7  # Moderate
        elif pe < 40:
            return 0.4  # Expensive
        else:
            return 0.2  # Very expensive
    
    def _score_peg(self, peg: float) -> float:
        """Score PEG ratio"""
        if np.isnan(peg) or peg <= 0:
            return 0.3
        if peg < 0.5:
            return 0.9  # Very undervalued
        elif peg < 1:
            return 1.0  # Undervalued
        elif peg < 1.5:
            return 0.8  # Fair
        elif peg < 2:
            return 0.6  # Slightly expensive
        else:
            return 0.3  # Expensive
    
    def _score_roe(self, roe: float) -> float:
        """Score ROE"""
        if np.isnan(roe):
            return 0.3
        if roe > 25:
            return 1.0  # Excellent
        elif roe > 20:
            return 0.9
        elif roe > 15:
            return 0.8
        elif roe > 10:
            return 0.6
        elif roe > 5:
            return 0.4
        else:
            return 0.2
    
    def _score_roa(self, roa: float) -> float:
        """Score ROA"""
        if np.isnan(roa):
            return 0.3
        if roa > 15:
            return 1.0
        elif roa > 10:
            return 0.9
        elif roa > 7:
            return 0.7
        elif roa > 5:
            return 0.5
        else:
            return 0.3
    
    def _score_earnings_growth(self, growth: float) -> float:
        """Score earnings growth rate"""
        if np.isnan(growth):
            return 0.3
        if growth > 50:
            return 1.0  # Exceptional
        elif growth > 30:
            return 0.9
        elif growth > 20:
            return 0.8
        elif growth > 10:
            return 0.7
        elif growth > 5:
            return 0.6
        elif growth > 0:
            return 0.4
        else:
            return 0.2  # Negative growth
    
    def _score_revenue_growth(self, growth: float) -> float:
        """Score revenue growth rate"""
        if np.isnan(growth):
            return 0.3
        if growth > 30:
            return 1.0
        elif growth > 20:
            return 0.9
        elif growth > 15:
            return 0.8
        elif growth > 10:
            return 0.7
        elif growth > 5:
            return 0.5
        elif growth > 0:
            return 0.4
        else:
            return 0.2
    
    def _score_debt(self, de: float) -> float:
        """Score debt-to-equity ratio (lower is better for growth)"""
        if np.isnan(de):
            return 0.5
        if de < 0:
            return 0.3  # Negative equity
        elif de < 30:
            return 1.0  # Very low debt
        elif de < 50:
            return 0.9
        elif de < 100:
            return 0.7
        elif de < 150:
            return 0.5
        elif de < 200:
            return 0.3
        else:
            return 0.1  # High debt
    
    def _score_liquidity(self, cr: float) -> float:
        """Score current ratio"""
        if np.isnan(cr):
            return 0.5
        if cr < 0.5:
            return 0.2  # Liquidity risk
        elif cr < 1:
            return 0.4
        elif cr < 1.5:
            return 0.7
        elif cr < 2:
            return 1.0  # Ideal
        elif cr < 3:
            return 0.8
        else:
            return 0.6  # Too much idle capital
    
    def _score_dividend(self, div_yield: float) -> float:
        """Score dividend yield for growth (moderate is best)"""
        if div_yield <= 0:
            return 0.5  # No dividend - neutral for growth
        elif div_yield < 2:
            return 0.6
        elif div_yield < 4:
            return 0.8  # Good balance
        elif div_yield < 6:
            return 0.9  # Strong yield
        elif div_yield < 10:
            return 0.7  # Very high - might limit growth
        else:
            return 0.4  # Unsustainably high
    
    def _score_margin(self, margin: float) -> float:
        """Score profit margin"""
        if np.isnan(margin):
            return 0.3
        if margin > 25:
            return 1.0
        elif margin > 20:
            return 0.9
        elif margin > 15:
            return 0.8
        elif margin > 10:
            return 0.7
        elif margin > 5:
            return 0.5
        elif margin > 0:
            return 0.3
        else:
            return 0.1  # Negative margin
    
    def get_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate composite fundamental score.
        
        Weights:
        - Earnings growth: 25%
        - ROE: 20%
        - Revenue growth: 15%
        - P/E: 15%
        - Debt: 10%
        - Margin: 10%
        - Liquidity: 5%
        
        Args:
            scores: Dictionary of individual scores
        
        Returns:
            Composite score (0-1)
        """
        weights = {
            "earnings_growth_score": 0.25,
            "roe_score": 0.20,
            "revenue_growth_score": 0.15,
            "pe_score": 0.15,
            "debt_score": 0.10,
            "margin_score": 0.10,
            "liquidity_score": 0.05,
        }
        
        composite = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in scores:
                composite += scores[key] * weight
                total_weight += weight
        
        return composite / total_weight if total_weight > 0 else 0.0
    
    def generate_analysis_summary(
        self, 
        ticker: str, 
        metrics: FundamentalMetrics, 
        scores: Dict[str, float]
    ) -> str:
        """
        Generate human-readable analysis summary.
        
        Args:
            ticker: Stock ticker
            metrics: FundamentalMetrics object
            scores: Scoring dictionary
        
        Returns:
            Formatted analysis string
        """
        lines = [
            f"\n{'='*50}",
            f"FUNDAMENTAL ANALYSIS: {ticker}",
            f"{'='*50}",
            "",
            "VALUATION:",
            f"  P/E Ratio: {metrics.pe_ratio:.2f}" if not np.isnan(metrics.pe_ratio) else "  P/E Ratio: N/A",
            f"  P/B Ratio: {metrics.pb_ratio:.2f}" if not np.isnan(metrics.pb_ratio) else "  P/B Ratio: N/A",
            f"  PEG Ratio: {metrics.peg_ratio:.2f}" if not np.isnan(metrics.peg_ratio) else "  PEG Ratio: N/A",
            "",
            "PROFITABILITY:",
            f"  ROE: {metrics.roe:.1f}%" if not np.isnan(metrics.roe) else "  ROE: N/A",
            f"  ROA: {metrics.roa:.1f}%" if not np.isnan(metrics.roa) else "  ROA: N/A",
            f"  Profit Margin: {metrics.profit_margin:.1f}%" if not np.isnan(metrics.profit_margin) else "  Profit Margin: N/A",
            "",
            "GROWTH:",
            f"  Earnings Growth: {metrics.earnings_growth:.1f}%" if not np.isnan(metrics.earnings_growth) else "  Earnings Growth: N/A",
            f"  Revenue Growth: {metrics.revenue_growth:.1f}%" if not np.isnan(metrics.revenue_growth) else "  Revenue Growth: N/A",
            "",
            "FINANCIAL HEALTH:",
            f"  Debt/Equity: {metrics.debt_to_equity:.1f}" if not np.isnan(metrics.debt_to_equity) else "  Debt/Equity: N/A",
            f"  Current Ratio: {metrics.current_ratio:.2f}" if not np.isnan(metrics.current_ratio) else "  Current Ratio: N/A",
            "",
            "DIVIDENDS:",
            f"  Dividend Yield: {metrics.dividend_yield:.2f}%",
            "",
            "SCORES:",
        ]
        
        for key, score in scores.items():
            clean_key = key.replace("_score", "").replace("_", " ").title()
            lines.append(f"  {clean_key}: {score:.2f}")
        
        composite = self.get_composite_score(scores)
        lines.extend([
            "",
            f"COMPOSITE FUNDAMENTAL SCORE: {composite:.2f}",
            f"{'='*50}",
        ])
        
        return "\n".join(lines)

"""
Growth Scoring Module
=====================
Combines fundamental and technical analysis into comprehensive growth scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import NGXDataLoader
from core.models import (
    FundamentalMetrics,
    TechnicalMetrics,
    GrowthScore,
    StockAnalysisResult,
)
from config.settings import (
    ScoringWeights,
    AnalysisConfig,
    get_sector_outlook,
    get_ticker_sector,
)
from analysis.fundamental import FundamentalAnalyzer
from analysis.technical import TechnicalAnalyzer
from analysis.report_analyzer import ReportAnalyzer
from analysis.disclosure_analyzer import DisclosureAnalyzer, compute_disclosure_adjustment
from analysis.seasonality import SeasonalAnalyzer


class GrowthAnalyzer:
    """
    Comprehensive growth potential analyzer.
    
    Combines:
    - Fundamental analysis (earnings, valuation, financial health)
    - Technical analysis (momentum, trend, risk metrics)
    - Sector outlook
    
    GROWTH DEFINITION:
    ==================
    "Growth" in this model encompasses multiple dimensions:
    
    1. EARNINGS GROWTH - Company's ability to increase profits
       - EPS growth rate
       - Revenue growth
       - Margin expansion
       - ROE improvement
    
    2. CAPITAL APPRECIATION - Stock price upside potential
       - Undervaluation vs intrinsic value
       - Technical momentum
       - Relative strength
    
    3. RISK-ADJUSTED RETURNS - Quality of returns
       - Sharpe ratio
       - Volatility-adjusted gains
       - Drawdown resilience
    
    4. SECTOR TAILWINDS - Macro factors
       - Industry growth outlook
       - Economic conditions
       - Regulatory environment
    """
    
    def __init__(self, target_year: int = None):
        """
        Initialize growth analyzer.
        
        Args:
            target_year: Year to analyze growth potential for
        """
        self.target_year = target_year or AnalysisConfig.TARGET_YEAR
        self.loader = NGXDataLoader()
        self.fundamental_analyzer = FundamentalAnalyzer(self.loader)
        self.technical_analyzer = TechnicalAnalyzer(self.loader)
        self.seasonal_analyzer = SeasonalAnalyzer(self.loader)
        self.report_analyzer = ReportAnalyzer()

        # Disclosure event signals (overlay — loads data/disclosures/ cache)
        self.disclosure_analyzer = DisclosureAnalyzer()

        # Load latest cached annual-report data (if any)
        self._report_cache: dict = {}       # ticker -> raw report dict
        self._report_analysis: dict = {}    # ticker -> analysis result dict
        self._load_report_cache()

        # Validate scoring weights
        ScoringWeights.validate()

    # ── Report cache helpers ──────────────────────────────────────
    def _load_report_cache(self) -> None:
        """Load the most recent report cache from data/reports/."""
        import json
        reports_dir = Path(__file__).resolve().parent.parent / "data" / "reports"
        if not reports_dir.exists():
            return
        cache_files = sorted(reports_dir.glob("*.json"), reverse=True)
        if not cache_files:
            return
        with open(cache_files[0], "r", encoding="utf-8") as f:
            self._report_cache = json.load(f)

    def _get_report_tone(self, ticker: str) -> tuple[float, str]:
        """Return (tone_score, tone_label) for *ticker* or (0.0, 'N/A')."""
        if ticker in self._report_analysis:
            a = self._report_analysis[ticker]
            return a.get("tone_score", 0.0), a.get("tone", "N/A")

        raw = self._report_cache.get(ticker)
        if not raw:
            return 0.0, "N/A"

        try:
            a = self.report_analyzer.analyze_report(raw)
            self._report_analysis[ticker] = a
            return a.get("tone_score", 0.0), a.get("tone", "N/A")
        except Exception:
            return 0.0, "N/A"
    
    def analyze_stock(self, ticker: str) -> Optional[StockAnalysisResult]:
        """
        Perform complete analysis on a single stock.
        
        Args:
            ticker: NGX ticker symbol
        
        Returns:
            StockAnalysisResult with all metrics and scores
        """
        # Get price data to check data quality
        prices = self.loader.get_price_series(ticker)
        
        if prices is None or len(prices) < AnalysisConfig.MIN_DATA_POINTS:
            return None
        
        # Fundamental analysis
        fundamental_metrics = self.fundamental_analyzer.analyze(ticker)
        fundamental_scores = self.fundamental_analyzer.score_fundamentals(fundamental_metrics)
        
        # Technical analysis
        technical_metrics = self.technical_analyzer.analyze(ticker)
        technical_scores = self.technical_analyzer.score_technicals(technical_metrics)

        # Seasonality analysis
        seasonal_metrics = self.seasonal_analyzer.analyze(ticker)
        seasonal_scores = self.seasonal_analyzer.score_seasonal(seasonal_metrics)

        # Disclosure event signals — computed before scoring so the capped
        # disclosure adjustment can be folded into the growth score.
        disclosure_signals = self.disclosure_analyzer.analyze(ticker)

        # Calculate growth score
        growth_score = self._calculate_growth_score(
            ticker,
            fundamental_metrics,
            technical_metrics,
            fundamental_scores,
            technical_scores,
            seasonal_scores,
            disclosure_signals,
        )

        # Seasonal context labels on the score object
        growth_score.current_month_label = self.seasonal_analyzer.label_current_month(seasonal_metrics)
        growth_score.seasonal_warning = self.seasonal_analyzer.is_warning(seasonal_metrics)

        # Create result
        result = StockAnalysisResult(
            ticker=ticker,
            sector=get_ticker_sector(ticker),
            analysis_date=datetime.now(),
            current_price=technical_metrics.price,
            fundamental=fundamental_metrics,
            technical=technical_metrics,
            seasonal=seasonal_metrics,
            growth=growth_score,
            fundamental_scores=fundamental_scores,
            technical_scores=technical_scores,
            seasonal_scores=seasonal_scores,
            days_of_data=len(prices),
        )
        
        # Assess data quality
        if len(prices) >= 252:
            result.data_quality = "GOOD"
        elif len(prices) >= 126:
            result.data_quality = "PARTIAL"
        else:
            result.data_quality = "POOR"

        # Attach the disclosure signals used in scoring above.
        result.disclosure_signals = disclosure_signals

        # A late/overdue filer is a data-quality risk: downgrade one notch.
        if result.disclosure_signals.get("late_filer") and result.data_quality == "GOOD":
            result.data_quality = "PARTIAL"

        return result
    
    def _calculate_growth_score(
        self,
        ticker: str,
        fundamental: FundamentalMetrics,
        technical: TechnicalMetrics,
        fund_scores: Dict[str, float],
        tech_scores: Dict[str, float],
        seasonal_scores: Optional[Dict[str, float]] = None,
        disclosure_signals: Optional[Dict] = None,
    ) -> GrowthScore:
        """
        Calculate comprehensive growth score.
        
        Components (weights defined in ScoringWeights):
        1. Earnings Growth (25%): EPS/Revenue growth, ROE
        2. Price Appreciation (25%): Valuation, momentum potential
        3. Momentum & Trend (20%): Technical strength
        4. Financial Health (15%): Debt, liquidity
        5. Sector Outlook (10%): Macro tailwinds
        6. Risk-Adjusted (5%): Sharpe, volatility
        
        Args:
            ticker: Stock ticker
            fundamental: FundamentalMetrics
            technical: TechnicalMetrics
            fund_scores: Fundamental scores dict
            tech_scores: Technical scores dict
        
        Returns:
            GrowthScore object
        """
        sector = get_ticker_sector(ticker)
        score = GrowthScore(ticker=ticker, sector=sector)
        
        # 1. Earnings Growth Component (25%)
        earnings_score = (
            fund_scores.get("earnings_growth_score", 0.3) * 0.40 +
            fund_scores.get("revenue_growth_score", 0.3) * 0.30 +
            fund_scores.get("roe_score", 0.3) * 0.20 +
            fund_scores.get("margin_score", 0.3) * 0.10
        )
        score.earnings_growth_score = round(earnings_score * 100, 2)
        
        # 2. Price Appreciation Potential (25%)
        appreciation_score = (
            fund_scores.get("pe_score", 0.3) * 0.30 +
            fund_scores.get("peg_score", 0.3) * 0.20 +
            tech_scores.get("momentum_score", 0.3) * 0.30 +
            tech_scores.get("bb_score", 0.5) * 0.20
        )
        score.price_appreciation_score = round(appreciation_score * 100, 2)
        
        # 3. Momentum & Trend (20%)
        momentum_score = (
            tech_scores.get("momentum_score", 0.3) * 0.40 +
            tech_scores.get("trend_score", 0.3) * 0.35 +
            tech_scores.get("rsi_score", 0.5) * 0.15 +
            tech_scores.get("macd_score", 0.5) * 0.10
        )
        score.momentum_score = round(momentum_score * 100, 2)
        
        # 4. Financial Health (15%)
        health_score = (
            fund_scores.get("debt_score", 0.5) * 0.40 +
            fund_scores.get("liquidity_score", 0.5) * 0.30 +
            fund_scores.get("dividend_score", 0.5) * 0.30
        )
        score.financial_health_score = round(health_score * 100, 2)
        
        # 5. Sector Outlook (10%)
        sector_outlook = get_sector_outlook(self.target_year)
        sector_score = sector_outlook.get(sector, 0.5)
        score.sector_score = round(sector_score * 100, 2)
        
        # 6. Risk-Adjusted (5%)
        risk_score = (
            tech_scores.get("sharpe_score", 0.3) * 0.50 +
            tech_scores.get("volatility_score", 0.5) * 0.25 +
            tech_scores.get("drawdown_score", 0.5) * 0.25
        )
        score.risk_adjusted_score = round(risk_score * 100, 2)

        # 7. Seasonality (5%)
        seasonal_scores = seasonal_scores or {}
        seasonal_score = (
            seasonal_scores.get("seasonal_current_score", 0.5) * 0.70 +
            seasonal_scores.get("seasonal_consistency_score", 0.5) * 0.30
        )
        score.seasonality_score = round(seasonal_score * 100, 2)

        # Calculate total score (weighted average)
        raw_total = round(
            earnings_score * ScoringWeights.EARNINGS_GROWTH +
            appreciation_score * ScoringWeights.PRICE_APPRECIATION +
            momentum_score * ScoringWeights.MOMENTUM_TREND +
            health_score * ScoringWeights.FINANCIAL_HEALTH +
            sector_score * ScoringWeights.SECTOR_OUTLOOK +
            risk_score * ScoringWeights.RISK_ADJUSTED +
            seasonal_score * ScoringWeights.SEASONALITY,
            2
        ) * 100

        # Annual-report tone bonus / penalty (max +/-5 pts)
        tone_score, tone_label = self._get_report_tone(ticker)
        report_bonus = round(tone_score * 5.0, 2)  # -5 … +5

        # NGX disclosure adjustment (Tier A — late filer / stale / dividend)
        disclosure_adjustment, disclosure_reasons = self._calculate_disclosure_adjustment(
            disclosure_signals or {}
        )

        # Combine the weighted total with both adjustments, clamped to [0, 100]
        total = raw_total + report_bonus + disclosure_adjustment
        score.total_score = round(max(0.0, min(100.0, total)), 2)
        score.report_tone = tone_label                 # type: ignore[attr-defined]
        score.report_tone_score = tone_score           # type: ignore[attr-defined]
        score.report_bonus = report_bonus              # type: ignore[attr-defined]
        score.disclosure_adjustment = disclosure_adjustment
        score.disclosure_reasons = disclosure_reasons  # type: ignore[attr-defined]

        # Determine signal
        score = self._determine_signal(score)

        return score

    def _calculate_disclosure_adjustment(
        self, signals: Dict
    ) -> Tuple[float, List[str]]:
        """Capped Tier-A disclosure adjustment folded into the growth score.

        Thin wrapper over the shared
        analysis.disclosure_analyzer.compute_disclosure_adjustment so the
        notebooks and the scorer apply identical logic.
        """
        return compute_disclosure_adjustment(signals)
    
    def _determine_signal(self, score: GrowthScore) -> GrowthScore:
        """
        Determine buy/hold/sell signal based on total score.
        
        Args:
            score: GrowthScore object
        
        Returns:
            Updated GrowthScore with signals
        """
        total = score.total_score
        
        if total >= 75:
            score.signal_strength = "STRONG_BUY"
            score.buy_signal = True
        elif total >= 65:
            score.signal_strength = "BUY"
            score.buy_signal = True
        elif total >= 50:
            score.signal_strength = "HOLD"
            score.hold_signal = True
        elif total >= 40:
            score.signal_strength = "SELL"
            score.sell_signal = True
        else:
            score.signal_strength = "STRONG_SELL"
            score.sell_signal = True
        
        return score
    
    def analyze_all_stocks(
        self,
        tickers: List[str] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze all stocks and return sorted results.
        
        Args:
            tickers: List of tickers (uses all if not provided)
            show_progress: Show progress indicator
        
        Returns:
            DataFrame with analysis results sorted by growth score
        """
        from config.settings import get_all_tickers
        
        tickers = tickers or get_all_tickers()
        results = []
        
        total = len(tickers)
        for i, ticker in enumerate(tickers):
            if show_progress:
                print(f"Analyzing {ticker} ({i+1}/{total})...", end="\r")
            
            try:
                result = self.analyze_stock(ticker)
                if result and result.growth:
                    results.append(self._result_to_dict(result))
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                continue
        
        if show_progress:
            print(f"Completed analysis of {len(results)}/{total} stocks" + " " * 20)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values("growth_score", ascending=False).reset_index(drop=True)
        
        # Add ranks
        df["rank"] = range(1, len(df) + 1)
        
        # Add sector ranks
        df["sector_rank"] = df.groupby("sector")["growth_score"].rank(
            ascending=False, method="min"
        ).astype(int)
        
        return df
    
    def _result_to_dict(self, result: StockAnalysisResult) -> Dict:
        """Convert result to flat dictionary for DataFrame"""
        return {
            # Basic info
            "ticker": result.ticker,
            "sector": result.sector,
            "price": result.current_price,
            "data_quality": result.data_quality,
            
            # Growth scores
            "growth_score": result.growth.total_score,
            "earnings_score": result.growth.earnings_growth_score,
            "appreciation_score": result.growth.price_appreciation_score,
            "momentum_score": result.growth.momentum_score,
            "health_score": result.growth.financial_health_score,
            "sector_score": result.growth.sector_score,
            "risk_score": result.growth.risk_adjusted_score,
            "seasonality_score": result.growth.seasonality_score,
            "disclosure_adjustment": result.growth.disclosure_adjustment,
            "current_month_label": result.growth.current_month_label,
            "seasonal_warning": result.growth.seasonal_warning,
            "signal": result.growth.signal_strength,
            
            # Key fundamentals
            "pe_ratio": result.fundamental.pe_ratio if result.fundamental else np.nan,
            "pb_ratio": result.fundamental.pb_ratio if result.fundamental else np.nan,
            "roe": result.fundamental.roe if result.fundamental else np.nan,
            "dividend_yield": result.fundamental.dividend_yield if result.fundamental else 0,
            "debt_to_equity": result.fundamental.debt_to_equity if result.fundamental else np.nan,
            "earnings_growth": result.fundamental.earnings_growth if result.fundamental else np.nan,
            "revenue_growth": result.fundamental.revenue_growth if result.fundamental else np.nan,

            # NGX-official fundamentals (overlay — does not affect growth_score)
            "ngx_roe": (result.fundamental.ngx_figures.get("ngx_roe")
                        if result.fundamental else None),
            "ngx_revenue_growth": (result.fundamental.ngx_figures.get("ngx_revenue_growth")
                                   if result.fundamental else None),
            "ngx_earnings_growth": (result.fundamental.ngx_figures.get("ngx_earnings_growth")
                                    if result.fundamental else None),
            
            # Key technicals
            "momentum_12m": result.technical.momentum_12m if result.technical else np.nan,
            "momentum_6m": result.technical.momentum_6m if result.technical else np.nan,
            "momentum_3m": result.technical.momentum_3m if result.technical else np.nan,
            "rsi_14": result.technical.rsi_14 if result.technical else np.nan,
            "above_sma200": result.technical.above_sma200 if result.technical else False,
            "golden_cross": result.technical.golden_cross if result.technical else False,
            "volatility": result.technical.volatility_252d if result.technical else np.nan,
            "max_drawdown": result.technical.max_drawdown_1y if result.technical else np.nan,
            "sharpe_ratio": result.technical.sharpe_1y if result.technical else np.nan,

            # Annual-report sentiment (if available)
            "report_tone": getattr(result.growth, "report_tone", "N/A"),
            "report_tone_score": getattr(result.growth, "report_tone_score", 0.0),
            "report_bonus": getattr(result.growth, "report_bonus", 0.0),

            # NGX disclosure event signals (overlay — does not affect growth_score)
            "catalyst_flag": result.disclosure_signals.get("earnings_catalyst", False),
            "catalyst_reason": result.disclosure_signals.get("catalyst_reason", ""),
            "insider_score": result.disclosure_signals.get("insider_score", 0.0),
            "insider_count_90d": result.disclosure_signals.get("insider_count_90d", 0),
            "dividend_recent": result.disclosure_signals.get("dividend_recent", False),
            "late_filer": result.disclosure_signals.get("late_filer", False),
            "days_since_results": result.disclosure_signals.get("days_since_results"),

            # NGX issuer earnings-forecast guidance (overlay)
            "forecast_period": result.disclosure_signals.get("forecast_period", ""),
            "forecast_pbt": result.disclosure_signals.get("forecast_pbt"),
            "forecast_pat": result.disclosure_signals.get("forecast_pat"),
            "forecast_status": result.disclosure_signals.get("forecast_status", "none"),
        }
    
    def get_top_picks(self, df: pd.DataFrame, n: int = None) -> pd.DataFrame:
        """
        Get top N stock picks.
        
        Args:
            df: Analysis results DataFrame
            n: Number of picks (uses config default if not provided)
        
        Returns:
            DataFrame with top picks
        """
        n = n or AnalysisConfig.TOP_N_PICKS
        return df.head(n).copy()
    
    def get_sector_leaders(self, df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """
        Get top N stocks from each sector.
        
        Args:
            df: Analysis results DataFrame
            n: Number of stocks per sector
        
        Returns:
            DataFrame with sector leaders
        """
        sector_leaders = []
        
        for sector in df["sector"].unique():
            sector_df = df[df["sector"] == sector].head(n)
            sector_leaders.append(sector_df)
        
        return pd.concat(sector_leaders).sort_values("growth_score", ascending=False)
    
    def get_buy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get all stocks with buy signals.
        
        Args:
            df: Analysis results DataFrame
        
        Returns:
            DataFrame with buy signals only
        """
        return df[df["signal"].isin(["BUY", "STRONG_BUY"])].copy()
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            df: Analysis results DataFrame
        
        Returns:
            Formatted report string
        """
        top_picks = self.get_top_picks(df)
        
        lines = [
            "\n" + "=" * 70,
            f"NGX STOCK GROWTH ANALYSIS - TARGET YEAR: {self.target_year}",
            "=" * 70,
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Stocks Analyzed: {len(df)}",
            "",
            "GROWTH SCORING METHODOLOGY:",
            "-" * 40,
            f"  1. Earnings Growth:      {ScoringWeights.EARNINGS_GROWTH*100:.0f}%",
            f"  2. Price Appreciation:   {ScoringWeights.PRICE_APPRECIATION*100:.0f}%",
            f"  3. Momentum & Trend:     {ScoringWeights.MOMENTUM_TREND*100:.0f}%",
            f"  4. Financial Health:     {ScoringWeights.FINANCIAL_HEALTH*100:.0f}%",
            f"  5. Sector Outlook:       {ScoringWeights.SECTOR_OUTLOOK*100:.0f}%",
            f"  6. Risk-Adjusted:        {ScoringWeights.RISK_ADJUSTED*100:.0f}%",
            f"  7. Seasonality:          {ScoringWeights.SEASONALITY*100:.0f}%",
            "",
            "=" * 70,
            f"TOP {len(top_picks)} GROWTH PICKS FOR {self.target_year}:",
            "=" * 70,
        ]
        
        for i, row in top_picks.iterrows():
            lines.extend([
                "",
                f"#{row['rank']} {row['ticker']} ({row['sector']})",
                "-" * 40,
                f"  Growth Score: {row['growth_score']:.1f}/100 [{row['signal']}]",
                f"  Price: ₦{row['price']:,.2f}" if not np.isnan(row['price']) else "  Price: N/A",
                "",
                "  Score Breakdown:",
                f"    Earnings:     {row['earnings_score']:.1f}",
                f"    Appreciation: {row['appreciation_score']:.1f}",
                f"    Momentum:     {row['momentum_score']:.1f}",
                f"    Health:       {row['health_score']:.1f}",
                f"    Sector:       {row['sector_score']:.1f}",
                f"    Risk-Adj:     {row['risk_score']:.1f}",
                f"    Seasonality:  {row['seasonality_score']:.1f}  ({row['current_month_label']})",
                "",
                "  Key Metrics:",
                f"    P/E: {row['pe_ratio']:.1f}" if not np.isnan(row['pe_ratio']) else "    P/E: N/A",
                f"    ROE: {row['roe']:.1f}%" if not np.isnan(row['roe']) else "    ROE: N/A",
                f"    12M Momentum: {row['momentum_12m']:.1f}%" if not np.isnan(row['momentum_12m']) else "    12M Momentum: N/A",
                f"    Sharpe: {row['sharpe_ratio']:.2f}" if not np.isnan(row['sharpe_ratio']) else "    Sharpe: N/A",
                "",
                "  Disclosure Signals (NGX):",
                f"    Catalyst: {('YES - ' + row['catalyst_reason']) if row.get('catalyst_flag') else 'none'}",
                f"    Insider filings (90d): {row.get('insider_count_90d', 0)}"
                f"   |   Late filer: {'YES' if row.get('late_filer') else 'no'}"
                f"   |   Recent dividend: {'YES' if row.get('dividend_recent') else 'no'}",
            ])
        
        # Sector summary
        lines.extend([
            "",
            "=" * 70,
            "SECTOR DISTRIBUTION (Top 10):",
            "-" * 40,
        ])
        
        sector_counts = top_picks["sector"].value_counts()
        for sector, count in sector_counts.items():
            lines.append(f"  {sector}: {count} stocks")
        
        # Signal summary
        lines.extend([
            "",
            "=" * 70,
            "SIGNAL DISTRIBUTION:",
            "-" * 40,
        ])
        
        signal_counts = df["signal"].value_counts()
        for signal, count in signal_counts.items():
            lines.append(f"  {signal}: {count} stocks")
        
        # Disclaimer
        lines.extend([
            "",
            "=" * 70,
            "DISCLAIMER:",
            "-" * 40,
            "This analysis is for informational purposes only.",
            "Past performance does not guarantee future results.",
            "Always conduct your own research before investing.",
            "=" * 70,
        ])
        
        return "\n".join(lines)

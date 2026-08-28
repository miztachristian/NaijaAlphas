"""
Data Models for NGX Stock Analysis
==================================
Dataclass definitions for metrics and results.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np


@dataclass
class SeasonalMetrics:
    """Per-stock monthly seasonal patterns derived from historical returns."""

    ticker: str

    # month (1..12) -> stat across the years where that month had data
    monthly_avg_returns: Dict[int, float] = field(default_factory=dict)
    monthly_median_returns: Dict[int, float] = field(default_factory=dict)
    monthly_win_rates: Dict[int, float] = field(default_factory=dict)
    monthly_sample_sizes: Dict[int, int] = field(default_factory=dict)
    monthly_stdev: Dict[int, float] = field(default_factory=dict)

    # Per-year, per-month returns (for matrix display matching TradingView)
    # year -> {month -> return}
    yearly_monthly: Dict[int, Dict[int, float]] = field(default_factory=dict)

    current_month: int = 0
    current_month_score: float = 0.0  # [-1, +1], 0 if insufficient data
    next_month_score: float = 0.0     # [-1, +1]

    best_months: List[Tuple[int, float]] = field(default_factory=list)
    worst_months: List[Tuple[int, float]] = field(default_factory=list)

    years_of_data: int = 0

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "monthly_avg_returns": self.monthly_avg_returns,
            "monthly_median_returns": self.monthly_median_returns,
            "monthly_win_rates": self.monthly_win_rates,
            "monthly_sample_sizes": self.monthly_sample_sizes,
            "monthly_stdev": self.monthly_stdev,
            "yearly_monthly": self.yearly_monthly,
            "current_month": self.current_month,
            "current_month_score": self.current_month_score,
            "next_month_score": self.next_month_score,
            "best_months": self.best_months,
            "worst_months": self.worst_months,
            "years_of_data": self.years_of_data,
        }


@dataclass
class FundamentalMetrics:
    """Fundamental analysis metrics for a stock"""
    
    ticker: str
    
    # Valuation metrics
    pe_ratio: float = np.nan
    forward_pe: float = np.nan
    pb_ratio: float = np.nan
    ps_ratio: float = np.nan
    peg_ratio: float = np.nan
    ev_ebitda: float = np.nan
    
    # Profitability metrics
    roe: float = np.nan
    roa: float = np.nan
    profit_margin: float = np.nan
    operating_margin: float = np.nan
    gross_margin: float = np.nan
    
    # Growth metrics
    revenue_growth: float = np.nan
    earnings_growth: float = np.nan
    eps_growth_5y: float = np.nan
    revenue_growth_5y: float = np.nan
    
    # Financial health
    debt_to_equity: float = np.nan
    current_ratio: float = np.nan
    quick_ratio: float = np.nan
    interest_coverage: float = np.nan
    
    # Dividends
    dividend_yield: float = 0.0
    payout_ratio: float = np.nan
    dividend_growth_5y: float = np.nan
    
    # Size and liquidity
    market_cap: float = np.nan
    enterprise_value: float = np.nan
    avg_volume: float = np.nan
    
    # Risk
    beta: float = 1.0

    # NGX-official statement figures (overlay — see ingest/parse_statements.py).
    # Kept alongside the TradingView snapshot metrics for cross-validation.
    # Keys: ngx_roe, ngx_roa, ngx_revenue_growth, ngx_earnings_growth, ngx_profit_margin.
    ngx_figures: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "pe_ratio": self.pe_ratio,
            "forward_pe": self.forward_pe,
            "pb_ratio": self.pb_ratio,
            "ps_ratio": self.ps_ratio,
            "peg_ratio": self.peg_ratio,
            "ev_ebitda": self.ev_ebitda,
            "roe": self.roe,
            "roa": self.roa,
            "profit_margin": self.profit_margin,
            "operating_margin": self.operating_margin,
            "gross_margin": self.gross_margin,
            "revenue_growth": self.revenue_growth,
            "earnings_growth": self.earnings_growth,
            "eps_growth_5y": self.eps_growth_5y,
            "revenue_growth_5y": self.revenue_growth_5y,
            "debt_to_equity": self.debt_to_equity,
            "current_ratio": self.current_ratio,
            "quick_ratio": self.quick_ratio,
            "interest_coverage": self.interest_coverage,
            "dividend_yield": self.dividend_yield,
            "payout_ratio": self.payout_ratio,
            "dividend_growth_5y": self.dividend_growth_5y,
            "market_cap": self.market_cap,
            "enterprise_value": self.enterprise_value,
            "avg_volume": self.avg_volume,
            "beta": self.beta,
            "ngx_figures": self.ngx_figures,
        }


@dataclass
class TechnicalMetrics:
    """Technical analysis metrics for a stock"""
    
    ticker: str
    
    # Price
    price: float = np.nan
    price_change_1d: float = np.nan
    
    # Moving averages
    sma_20: float = np.nan
    sma_50: float = np.nan
    sma_200: float = np.nan
    ema_12: float = np.nan
    ema_26: float = np.nan
    
    # Trend indicators
    above_sma50: bool = False
    above_sma200: bool = False
    golden_cross: bool = False
    death_cross: bool = False
    
    # Oscillators
    rsi_14: float = np.nan
    macd: float = np.nan
    macd_signal: float = np.nan
    macd_histogram: float = np.nan
    stochastic_k: float = np.nan
    stochastic_d: float = np.nan
    
    # Momentum
    momentum_1m: float = np.nan
    momentum_3m: float = np.nan
    momentum_6m: float = np.nan
    momentum_12m: float = np.nan
    momentum_12_1: float = np.nan  # 12-1 month momentum factor
    
    # Volatility
    volatility_20d: float = np.nan
    volatility_60d: float = np.nan
    volatility_252d: float = np.nan
    atr_14: float = np.nan
    
    # Bollinger Bands
    bb_upper: float = np.nan
    bb_middle: float = np.nan
    bb_lower: float = np.nan
    bb_width: float = np.nan
    bb_position: float = np.nan  # Position within bands (0-1)
    
    # Risk metrics
    max_drawdown_1y: float = np.nan
    sharpe_1y: float = np.nan
    sortino_1y: float = np.nan
    
    # Volume
    volume_sma_20: float = np.nan
    volume_ratio: float = np.nan  # Current volume vs average
    
    # Relative strength
    relative_strength_vs_index: float = np.nan
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "price": self.price,
            "price_change_1d": self.price_change_1d,
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "ema_12": self.ema_12,
            "ema_26": self.ema_26,
            "above_sma50": self.above_sma50,
            "above_sma200": self.above_sma200,
            "golden_cross": self.golden_cross,
            "death_cross": self.death_cross,
            "rsi_14": self.rsi_14,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "stochastic_k": self.stochastic_k,
            "stochastic_d": self.stochastic_d,
            "momentum_1m": self.momentum_1m,
            "momentum_3m": self.momentum_3m,
            "momentum_6m": self.momentum_6m,
            "momentum_12m": self.momentum_12m,
            "momentum_12_1": self.momentum_12_1,
            "volatility_20d": self.volatility_20d,
            "volatility_60d": self.volatility_60d,
            "volatility_252d": self.volatility_252d,
            "atr_14": self.atr_14,
            "bb_upper": self.bb_upper,
            "bb_middle": self.bb_middle,
            "bb_lower": self.bb_lower,
            "bb_width": self.bb_width,
            "bb_position": self.bb_position,
            "max_drawdown_1y": self.max_drawdown_1y,
            "sharpe_1y": self.sharpe_1y,
            "sortino_1y": self.sortino_1y,
            "volume_sma_20": self.volume_sma_20,
            "volume_ratio": self.volume_ratio,
            "relative_strength_vs_index": self.relative_strength_vs_index,
        }


@dataclass
class GrowthScore:
    """Growth potential scores for a stock"""
    
    ticker: str
    sector: str
    
    # Overall score (0-100)
    total_score: float = 0.0
    
    # Component scores (0-100)
    earnings_growth_score: float = 0.0
    price_appreciation_score: float = 0.0
    momentum_score: float = 0.0
    financial_health_score: float = 0.0
    sector_score: float = 0.0
    risk_adjusted_score: float = 0.0
    seasonality_score: float = 0.0

    # Additive NGX-disclosure adjustment folded into total_score (capped ±5;
    # see analysis/growth.py + config.settings.DisclosureScoring).
    disclosure_adjustment: float = 0.0

    # Seasonality context strings
    current_month_label: str = ""   # e.g. "Strong May (+4.3%, 80% win)"
    seasonal_warning: bool = False  # True when current_month_score < -0.3

    # Ranking
    rank: int = 0
    sector_rank: int = 0

    # Signals
    buy_signal: bool = False
    hold_signal: bool = False
    sell_signal: bool = False
    signal_strength: str = "NEUTRAL"  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "sector": self.sector,
            "total_score": self.total_score,
            "earnings_growth_score": self.earnings_growth_score,
            "price_appreciation_score": self.price_appreciation_score,
            "momentum_score": self.momentum_score,
            "financial_health_score": self.financial_health_score,
            "sector_score": self.sector_score,
            "risk_adjusted_score": self.risk_adjusted_score,
            "seasonality_score": self.seasonality_score,
            "disclosure_adjustment": self.disclosure_adjustment,
            "current_month_label": self.current_month_label,
            "seasonal_warning": self.seasonal_warning,
            "rank": self.rank,
            "sector_rank": self.sector_rank,
            "buy_signal": self.buy_signal,
            "hold_signal": self.hold_signal,
            "sell_signal": self.sell_signal,
            "signal_strength": self.signal_strength,
        }


@dataclass
class StockAnalysisResult:
    """Complete analysis result for a stock"""
    
    ticker: str
    sector: str
    analysis_date: datetime
    
    # Current price info
    current_price: float = np.nan
    currency: str = "NGN"
    
    # Metrics
    fundamental: Optional[FundamentalMetrics] = None
    technical: Optional[TechnicalMetrics] = None
    seasonal: Optional[SeasonalMetrics] = None
    growth: Optional[GrowthScore] = None

    # Component scores (0-1 scale)
    fundamental_scores: Dict[str, float] = field(default_factory=dict)
    technical_scores: Dict[str, float] = field(default_factory=dict)
    seasonal_scores: Dict[str, float] = field(default_factory=dict)

    # NGX disclosure event signals (overlay — see analysis/disclosure_analyzer.py).
    # Does not affect growth_score; surfaced as separate columns/flags.
    disclosure_signals: Dict = field(default_factory=dict)

    # Data quality
    data_quality: str = "GOOD"  # GOOD, PARTIAL, POOR
    days_of_data: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            "ticker": self.ticker,
            "sector": self.sector,
            "analysis_date": self.analysis_date.isoformat(),
            "current_price": self.current_price,
            "currency": self.currency,
            "data_quality": self.data_quality,
            "days_of_data": self.days_of_data,
        }
        
        if self.fundamental:
            result["fundamental"] = self.fundamental.to_dict()
        if self.technical:
            result["technical"] = self.technical.to_dict()
        if self.seasonal:
            result["seasonal"] = self.seasonal.to_dict()
        if self.growth:
            result["growth"] = self.growth.to_dict()

        result["fundamental_scores"] = self.fundamental_scores
        result["technical_scores"] = self.technical_scores
        result["seasonal_scores"] = self.seasonal_scores
        result["disclosure_signals"] = self.disclosure_signals

        return result


@dataclass
class BacktestResult:
    """Backtesting result"""
    
    strategy_name: str
    start_date: datetime
    end_date: datetime
    
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Portfolio
    final_value: float = 0.0
    peak_value: float = 0.0
    
    # Holdings history
    holdings_history: List[Dict] = field(default_factory=list)
    portfolio_values: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "final_value": self.final_value,
            "peak_value": self.peak_value,
        }


@dataclass
class PuntCard:
    """
    Per-ticker output of the Gamble Punt screener.

    `buy_zone` is `None` when the ticker is AVOID. It is a `(low, high)` tuple
    for actionable states, and `("deferred", "wait for retrace to EMA-50 +/-3%")`
    when the technical setup says wait (state then caps at SETUP).

    `component_scores` keys: setup, catalyst, insider, business_pulse, liquidity.
    The `insider` value is the ONLY one that may be negative (down to -20).
    """
    ticker: str
    tier: str                                     # "micro" | "low"
    score: float                                  # 0..100 (composite is clipped to 0)
    state: str                                    # "AVOID" | "WATCH" | "SETUP" | "TRIGGER"
    component_scores: Dict[str, float]
    buy_zone: Optional[Tuple[float, float] | Tuple[str, str]]
    stop: Optional[float]
    catalyst: str
    next_rating_change: str
    warnings: List[str]
    evidence: Dict

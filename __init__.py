"""
NGX Stock Analysis System
=========================
A comprehensive system for analyzing Nigerian stocks.

Modules:
- config: Configuration and settings
- core: Data loading and models
- analysis: Fundamental, technical, and growth analysis

Quick Start:
    from ngx_analysis import GrowthAnalyzer
    
    analyzer = GrowthAnalyzer(target_year=2026)
    results = analyzer.analyze_all_stocks()
    print(analyzer.generate_report(results))
"""

from config.settings import (
    AnalysisConfig,
    ScoringWeights,
    TechnicalParams,
    BacktestConfig,
    get_sector_outlook,
    get_ticker_sector,
    get_all_tickers,
)

from core.data_loader import NGXDataLoader
from core.models import (
    FundamentalMetrics,
    TechnicalMetrics,
    GrowthScore,
    StockAnalysisResult,
    BacktestResult,
)

from analysis.fundamental import FundamentalAnalyzer
from analysis.technical import TechnicalAnalyzer
from analysis.growth import GrowthAnalyzer
from analysis.backtest import BacktestEngine, Portfolio

__version__ = "1.0.0"
__author__ = "NGX Analysis System"

__all__ = [
    # Config
    "AnalysisConfig",
    "ScoringWeights",
    "TechnicalParams",
    "BacktestConfig",
    "get_sector_outlook",
    "get_ticker_sector",
    "get_all_tickers",
    # Core
    "NGXDataLoader",
    "FundamentalMetrics",
    "TechnicalMetrics",
    "GrowthScore",
    "StockAnalysisResult",
    "BacktestResult",
    # Analysis
    "FundamentalAnalyzer",
    "TechnicalAnalyzer",
    "GrowthAnalyzer",
    "BacktestEngine",
    "Portfolio",
]

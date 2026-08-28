"""
NGX Stock Analysis System - Configuration Settings
===================================================
Central configuration for the Nigerian stock analysis system.
Modify these settings to customize analysis parameters.
"""

from datetime import datetime
from pathlib import Path


# ==================== DIRECTORIES ====================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"
BACKTEST_DIR = BASE_DIR / "backtests"

# decision_system directories
MACRO_DIR = DATA_DIR / "macro"            # Nigeria macro time series
PORTFOLIO_DIR = DATA_DIR / "portfolio"    # canonical holdings
DECISIONS_DIR = OUTPUT_DIR / "decisions"  # daily decision artifacts

# Create directories
for dir_path in [DATA_DIR, OUTPUT_DIR, CACHE_DIR, BACKTEST_DIR,
                 MACRO_DIR, PORTFOLIO_DIR, DECISIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ==================== ANALYSIS SETTINGS ====================
class AnalysisConfig:
    """Configuration for stock analysis"""
    
    # Target year for growth analysis (change this for future years)
    TARGET_YEAR: int = 2026
    
    # Analysis date (defaults to today)
    ANALYSIS_DATE: datetime = datetime.now()
    
    # Historical data lookback period (years)
    LOOKBACK_YEARS: int = 5
    
    # Minimum data points required for analysis
    MIN_DATA_POINTS: int = 50
    
    # Risk-free rate (Nigeria T-bill rate approximation)
    RISK_FREE_RATE: float = 0.15
    
    # Top N stocks to select
    TOP_N_PICKS: int = 10


# ==================== SCORING WEIGHTS ====================
class ScoringWeights:
    """
    Weights for growth score calculation.
    These should sum to 1.0 (100%)
    """

    EARNINGS_GROWTH: float = 0.25      # 25% - EPS/Revenue growth, ROE
    PRICE_APPRECIATION: float = 0.25   # 25% - Valuation, momentum
    MOMENTUM_TREND: float = 0.15       # 15% - Technical strength
    FINANCIAL_HEALTH: float = 0.15     # 15% - Debt, dividends, liquidity
    SECTOR_OUTLOOK: float = 0.10       # 10% - Sector growth projections
    RISK_ADJUSTED: float = 0.05        # 5%  - Sharpe, volatility
    SEASONALITY: float = 0.05          # 5%  - Calendar-month tailwind/headwind

    @classmethod
    def validate(cls):
        """Ensure weights sum to 1.0"""
        total = (cls.EARNINGS_GROWTH + cls.PRICE_APPRECIATION +
                 cls.MOMENTUM_TREND + cls.FINANCIAL_HEALTH +
                 cls.SECTOR_OUTLOOK + cls.RISK_ADJUSTED +
                 cls.SEASONALITY)
        assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"


# ==================== DISCLOSURE SCORING ====================
class DisclosureScoring:
    """
    Controls how NGX disclosure findings influence the growth score.

    Two effects, both tunable and reversible here:
    1. USE_NGX_FUNDAMENTALS — official NGX statement figures feed
       score_fundamentals instead of the weaker TradingView snapshot.
    2. An additive disclosure adjustment (capped at ADJUSTMENT_CAP points)
       applied to total_score alongside the annual-report tone bonus.

    Only sign-unambiguous signals are used: a late/overdue filer is a
    governance penalty, a recent dividend a small quality bonus. Catalyst
    and insider-activity flags stay overlay-only (ambiguous sign).
    """

    ENABLED: bool = True               # master switch for the score adjustment
    USE_NGX_FUNDAMENTALS: bool = True   # Tier C figures feed score_fundamentals
    ADJUSTMENT_CAP: float = 5.0         # max |disclosure adjustment|, points

    LATE_FILER_PENALTY: float = -3.0
    STALE_RESULTS_PENALTY: float = -2.0  # results older than STALE_DAYS
    STALE_DAYS: int = 200
    DIVIDEND_BONUS: float = 1.0


# ==================== SEASONALITY SETTINGS ====================
class SeasonalityConfig:
    """Configuration for monthly seasonal pattern analysis."""

    MIN_YEARS: int = 3                       # need at least N years per month to score it
    STRONG_MONTH_THRESHOLD: float = 0.03     # avg monthly return >= +3% counts as strong
    WEAK_MONTH_THRESHOLD: float = -0.02      # avg monthly return <= -2% counts as weak
    HIGH_WIN_RATE: float = 0.65              # >= 65% positive years counts as consistent
    WARNING_SCORE: float = -0.30             # current_month_score below this raises a flag


# ==================== TECHNICAL PARAMETERS ====================
class TechnicalParams:
    """Parameters for technical analysis"""
    
    # Moving averages
    SMA_SHORT: int = 50
    SMA_LONG: int = 200
    EMA_SHORT: int = 12
    EMA_LONG: int = 26
    
    # RSI
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    
    # MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    # Bollinger Bands
    BB_PERIOD: int = 20
    BB_STD: int = 2
    
    # Momentum periods (trading days)
    MOMENTUM_1M: int = 21
    MOMENTUM_3M: int = 63
    MOMENTUM_6M: int = 126
    MOMENTUM_12M: int = 252
    
    # Volatility
    VOLATILITY_PERIOD: int = 252


# ==================== BACKTESTING SETTINGS ====================
class BacktestConfig:
    """Configuration for backtesting"""
    
    # Initial portfolio value
    INITIAL_CAPITAL: float = 10_000_000  # 10 million Naira
    
    # Rebalancing frequency
    REBALANCE_FREQUENCY: str = "quarterly"  # monthly, quarterly, annually
    
    # Transaction costs (%)
    TRANSACTION_COST: float = 0.01  # 1% transaction cost
    
    # Slippage (%)
    SLIPPAGE: float = 0.005  # 0.5% slippage
    
    # Maximum position size (% of portfolio)
    MAX_POSITION_SIZE: float = 0.20  # 20% max per stock
    
    # Minimum position size (% of portfolio)
    MIN_POSITION_SIZE: float = 0.05  # 5% min per stock
    
    # Stop loss threshold
    STOP_LOSS: float = -0.15  # -15% stop loss
    
    # Take profit threshold
    TAKE_PROFIT: float = 0.50  # 50% take profit


# ==================== SECTOR OUTLOOK ====================
def get_sector_outlook(year: int) -> dict:
    """
    Get sector growth outlook for a specific year.
    Update these projections annually based on economic forecasts.
    
    Returns scores from 0-1 indicating sector growth potential.
    """
    
    # Base outlook (adjust annually)
    base_outlook = {
        "BANKING": 0.75,      # Moderate - Interest rates, loan growth
        "TELECOM": 0.90,      # High - Digital transformation, 5G
        "OIL_GAS": 0.70,      # Moderate - Oil price dependent
        "CEMENT": 0.80,       # Good - Infrastructure spending
        "CONSUMER": 0.65,     # Moderate - Inflation pressures
        "AGRICULTURE": 0.85,  # High - Food security focus
        "HEALTHCARE": 0.85,   # High - Healthcare investment
        "INSURANCE": 0.80,    # Good - Low penetration, growth runway
        "INDUSTRIAL": 0.70,   # Moderate - Manufacturing push
        "REALESTATE": 0.60,   # Lower - Interest rate sensitivity
        "FINTECH": 0.92,      # Very High - Digital finance boom
        "OTHER": 0.50,        # Neutral
    }
    
    # Year-specific adjustments (add more years as needed)
    year_adjustments = {
        2026: {
            "TELECOM": 0.90,
            "FINTECH": 0.92,
            "BANKING": 0.78,
        },
        2027: {
            "TELECOM": 0.88,
            "OIL_GAS": 0.65,  # Energy transition impact
            "CEMENT": 0.82,
        }
    }
    
    outlook = base_outlook.copy()
    if year in year_adjustments:
        outlook.update(year_adjustments[year])
    
    return outlook


# ==================== TICKER UNIVERSE ====================
ALL_TICKERS = [
    # BANKING SECTOR
    "GTCO", "ZENITHBANK", "ACCESSCORP", "UBA", "STANBIC", "FIDELITYBK",
    "FCMB", "FIRSTHOLDCO", "WEMABANK", "STERLINGNG", "UCAP", "JAIZBANK",
    "UNITYBNK", "LIVINGTRUST",

    # TELECOMMUNICATIONS
    "MTNN", "AIRTELAFRI",

    # OIL & GAS / ENERGY
    "SEPLAT", "OANDO", "TOTAL", "CONOIL", "ETERNA", "GEREGU", "TRANSCORP",

    # CEMENT / BUILDING MATERIALS
    "DANGCEM", "BUACEMENT", "WAPCO",

    # CONSUMER GOODS
    "NESTLE", "NB", "DANGSUGAR", "NASCON", "FLOURMILL", "CADBURY",
    "UNILEVER", "PZ", "GUINNESS", "INTBREW", "CHAMPION", "VITAFOAM",
    "HONYFLOUR", "NNFM",

    # AGRICULTURE
    "PRESCO", "OKOMUOIL", "LIVESTOCK", "ELLAHLAKES",

    # HEALTHCARE / PHARMA
    "FIDSON", "MAYBAKER", "NEIMETH", "GLAXOSMITH", "MORISON",

    # INSURANCE
    "AIICO", "MANSARD", "NEM", "LASACO", "CORNERST", "CHIPLC",
    "REGALINS", "SUNUASSUR",

    # INDUSTRIALS / CONGLOMERATES
    "UACN", "CUTIX", "BERGER", "BETAGLAS", "NGXGROUP", "MEYER", "VANLEER",

    # REAL ESTATE
    "UPDC", "UPDCREIT", "SFSREIT", "UHOMREIT",

    # TECHNOLOGY / FINTECH
    "CWG", "ETRANZACT", "CHAMS", "NSLTECH", "VFDGROUP",

    # SERVICES / HOSPITALITY
    "CAVERTON", "IKEJAHOTEL", "TRANSCOHOT", "TANTALIZER",

    # TRADING / DISTRIBUTION
    "SCOA", "JOHNHOLT", "RTBRISCOE", "NCR", "ALEX",

    # EDUCATION
    "ACADEMY", "LEARNAFRCA",

    # ADDITIONAL STOCKS FROM NGX
    "ABBEYBDS", "ABCTRANS", "AFRINSURE", "AFRIPRUD", "AFROMEDIA",
    "ASOSAVINGS", "AUSTINLAZ", "BAPLC", "CILEASING", "DAARCOMM",
    "DEAPCAP", "DUNLOP", "FTGINSURE", "FTNCOCOA", "GOLDBREW",
    "HMCALL", "IMG", "INFINITY", "INTENEGINS", "JAPAULGOLD",
    "MBENEFIT", "MECURE", "MULTITREX", "MULTIVERSE", "NIDF",
    "NPFMCRFBK", "OMATEK", "PHARMADEKO", "PREMPAINTS", "PRESTIGE",
    "REDSTAREX", "RONCHESS", "ROYALEX", "SKYAVN", "SOVRENINS",
    "STACO", "TIP", "TRANSEXPR", "TRANSPOWER", "TRIPPLEG",
    "UNIVINSURE", "UPL",
]

# Sector classification
SECTOR_MAP = {
    "BANKING": [
        "GTCO", "ZENITHBANK", "ACCESSCORP", "UBA", "STANBIC", "FIDELITYBK",
        "FCMB", "FIRSTHOLDCO", "WEMABANK", "STERLINGNG", "UCAP", "JAIZBANK",
        "UNITYBNK", "LIVINGTRUST", "ASOSAVINGS", "NPFMCRFBK"
    ],
    "TELECOM": ["MTNN", "AIRTELAFRI", "DAARCOMM"],
    "OIL_GAS": ["SEPLAT", "OANDO", "TOTAL", "CONOIL", "ETERNA", "GEREGU", "TRANSCORP"],
    "CEMENT": ["DANGCEM", "BUACEMENT", "WAPCO"],
    "CONSUMER": [
        "NESTLE", "NB", "DANGSUGAR", "NASCON", "FLOURMILL", "CADBURY",
        "UNILEVER", "PZ", "GUINNESS", "INTBREW", "CHAMPION", "VITAFOAM",
        "HONYFLOUR", "NNFM", "GOLDBREW", "TANTALIZER", "MULTITREX"
    ],
    "AGRICULTURE": ["PRESCO", "OKOMUOIL", "LIVESTOCK", "ELLAHLAKES", "FTNCOCOA"],
    "HEALTHCARE": ["FIDSON", "MAYBAKER", "NEIMETH", "GLAXOSMITH", "MORISON", "PHARMADEKO", "MECURE"],
    "INSURANCE": [
        "AIICO", "MANSARD", "NEM", "LASACO", "CORNERST", "CHIPLC", "REGALINS", 
        "SUNUASSUR", "AFRINSURE", "AFRIPRUD", "FTGINSURE", "INTENEGINS",
        "SOVRENINS", "STACO", "UNIVINSURE"
    ],
    "INDUSTRIAL": [
        "UACN", "CUTIX", "BERGER", "BETAGLAS", "NGXGROUP", "MEYER", "VANLEER",
        "DUNLOP", "PREMPAINTS", "AUSTINLAZ", "BAPLC"
    ],
    "REALESTATE": ["UPDC", "UPDCREIT", "SFSREIT", "UHOMREIT"],
    "FINTECH": ["CWG", "ETRANZACT", "CHAMS", "NSLTECH", "VFDGROUP"],
    "HOSPITALITY": ["CAVERTON", "IKEJAHOTEL", "TRANSCOHOT", "TANTALIZER"],
    "TRANSPORT": ["ABCTRANS", "SKYAVN", "TRANSEXPR", "TRANSPOWER"],
    "MINING": ["JAPAULGOLD", "MULTIVERSE"],
    "EDUCATION": ["ACADEMY", "LEARNAFRCA"],
    "OTHER": [
        "SCOA", "JOHNHOLT", "RTBRISCOE", "NCR", "ALEX", "IMG", "ABBEYBDS",
        "AFROMEDIA", "CILEASING", "DEAPCAP", "HMCALL", "INFINITY", "MBENEFIT",
        "NIDF", "OMATEK", "PRESTIGE", "REDSTAREX", "RONCHESS", "ROYALEX",
        "TIP", "TRIPPLEG", "UPL"
    ],
}


def get_ticker_sector(ticker: str) -> str:
    """Get sector for a ticker"""
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "OTHER"


def get_all_tickers() -> list:
    """Get list of all tickers"""
    return list(ALL_TICKERS)


def get_sector_tickers(sector: str) -> list:
    """Get all tickers in a sector"""
    return SECTOR_MAP.get(sector, [])


# ==================== DECISION SYSTEM: CONVICTION WEIGHTS ====================
class ConvictionWeights:
    """
    Weights for the unified conviction score (decision_system/conviction.py).
    Each signal contributes a 0-100 sub-score; the conviction engine combines
    them with these weights, re-normalizing over only the signals that have
    data for a given stock. Weights must sum to 1.0.

    Starting point — re-calibrate with decision_system/calibration.py against
    historical forward returns. The 'market_context' signal carries sector
    outlook and snapshot-relative momentum.
    """

    FUNDAMENTAL: float = 0.30
    TECHNICAL: float = 0.20
    QUALITY: float = 0.15
    DISCLOSURE: float = 0.10
    SENTIMENT: float = 0.08
    SEASONALITY: float = 0.07
    REPORT_TONE: float = 0.05
    MARKET_CONTEXT: float = 0.05

    @classmethod
    def as_dict(cls) -> dict:
        """Signal-name -> weight, keyed to decision_system.models.SIGNAL_NAMES."""
        return {
            "fundamental": cls.FUNDAMENTAL,
            "technical": cls.TECHNICAL,
            "quality": cls.QUALITY,
            "disclosure": cls.DISCLOSURE,
            "sentiment": cls.SENTIMENT,
            "seasonality": cls.SEASONALITY,
            "report_tone": cls.REPORT_TONE,
            "market_context": cls.MARKET_CONTEXT,
        }

    @classmethod
    def validate(cls):
        """Ensure weights sum to 1.0."""
        total = sum(cls.as_dict().values())
        assert abs(total - 1.0) < 0.001, f"Conviction weights must sum to 1.0, got {total}"


# ==================== DECISION SYSTEM: CONVICTION THRESHOLDS ====================
class ConvictionConfig:
    """Action thresholds, macro-tilt cap, and confidence handling."""

    # Action thresholds on the final 0-100 conviction score.
    STRONG_BUY: float = 78.0
    ADD: float = 64.0
    HOLD: float = 48.0
    TRIM: float = 38.0
    # below TRIM => SELL (if held) / AVOID (if not held)

    # Macro tilt is bounded so the regime nudges but never dominates.
    MACRO_TILT_CAP: float = 6.0

    # Confidence shrinkage: final = 50 + (score-50) * confidence_factor, where
    # confidence_factor scales linearly from FLOOR (zero coverage) to 1.0 (full).
    CONFIDENCE_FACTOR_FLOOR: float = 0.60
    # A stock cannot be rated STRONG_BUY below this confidence (downgraded to ADD).
    STRONG_BUY_MIN_CONFIDENCE: float = 0.55


# ==================== DECISION SYSTEM: MACRO ====================
class MacroConfig:
    """
    Nigeria macro ingestion + regime classification (ingest/macro.py,
    decision_system/macro_regime.py).

    Fallback constants are used only when a live source cannot be reached;
    a run that falls back is flagged `degraded`. Update them periodically.
    """

    # --- Fallback levels (as of 2026-05; refresh when stale) ---
    FALLBACK_MPR: float = 27.5          # CBN Monetary Policy Rate, %
    FALLBACK_INFLATION: float = 23.0    # headline CPI YoY, %
    FALLBACK_USDNGN: float = 1550.0     # USD/NGN spot
    FALLBACK_BRENT: float = 65.0        # Brent crude, USD/bbl
    FALLBACK_BOND10Y: float = 14.95     # 10-Year FGN bond yield, %
                                        # (matches Assessworth Market Watch reading)

    # --- Trend detection ---
    TREND_WINDOW_DAYS: int = 60         # lookback for FX / oil slope
    FX_MOVE_THRESHOLD: float = 0.03     # >3% over window => trend (else STABLE)
    OIL_MOVE_THRESHOLD: float = 0.05    # >5% over window => trend (else FLAT)
    YIELD_MOVE_THRESHOLD: float = 0.05  # >5% relative change over window => trend

    # --- Sector tilt magnitudes (conviction points; sign set in macro_regime) ---
    TILT_STRONG: float = 4.0
    TILT_MILD: float = 2.0

    # Source endpoints (best-effort; fall back to constants on failure).
    USDNGN_URL: str = "https://open.er-api.com/v6/latest/USD"
    BRENT_STOOQ_URL: str = "https://stooq.com/q/l/?s=cb.f&f=sd2t2ohlcv&h&e=csv"  # Brent futures quote CSV


# ==================== DECISION SYSTEM: PORTFOLIO CONSTRUCTION ====================
class PortfolioConfig:
    """Position-sizing rules for decision_system/portfolio.py."""

    # Two-sleeve allocation split of deployable capital.
    LONG_TERM_SLEEVE: float = 0.85
    SHORT_TERM_SLEEVE: float = 0.15

    # Position caps as a fraction of total portfolio value.
    MAX_POSITION: float = 0.20          # reuse of BacktestConfig.MAX_POSITION_SIZE
    MIN_POSITION: float = 0.05
    MAX_SECTOR: float = 0.30            # concentration cap per sector

    # A position's naira value may not exceed this fraction of the stock's
    # daily traded value (liquidity_value_1d) — avoids being the whole market.
    MAX_LIQUIDITY_FRACTION: float = 0.15

    CASH_RESERVE: float = 0.07          # keep ~7% uninvested
    MIN_ORDER_NAIRA: float = 5_000.0    # skip dust trades

    HOLDINGS_FILE: str = "holdings.json"  # under PORTFOLIO_DIR


# ==================== NOTIFY / TELEGRAM ====================
class NotifyConfig:
    """Telegram bot + Ollama agent configuration."""

    ENABLED: bool = True

    # Brief section limits
    MAX_SELL_ROWS: int = 5
    MAX_BUY_ROWS: int = 5
    MAX_WATCH_ROWS: int = 4
    MAX_GEM_ROWS: int = 3
    MAX_SEASONAL_ROWS: int = 4
    MAX_PUNT_ROWS: int = 2

    # Ollama
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120

    # Watch list: held positions with conviction in [WATCH_LOWER, WATCH_UPPER)
    WATCH_LOWER: float = 58.0
    WATCH_UPPER: float = 64.0

    # Profile storage
    PROFILES_DIR: str = "data/profiles"


# ==================== CORE CONCENTRATE STRATEGY ====================
class CoreStrategyConfig:
    """Quality+profit+growth core selector, volume-confirmed, 2-name concentrate.

    Drives the long-term core sleeve. See
    docs/superpowers/specs/2026-06-24-core-concentrate-strategy-design.md.
    """

    # --- pillar weights (must sum to 1.0) ---
    W_GROWTH: float = 0.35
    W_PROFIT: float = 0.30
    W_QUALITY: float = 0.35

    # --- volume-confirmation modifier: core = base * (LOW + SPAN*accum/100) ---
    VOL_MOD_LOW: float = 0.85
    VOL_MOD_SPAN: float = 0.30          # -> range 0.85x .. 1.15x

    # --- hard gates ---
    ROE_FLOOR: float = 12.0             # roe_ttm (or roic_ttm fallback) >= this
    REV_GROWTH_FLOOR: float = 8.0       # growth gate: eps_growth>0 OR rev_growth>=this
    MIN_TURNOVER_NAIRA: float = 10_000_000.0   # price*volume_1d floor
    DIST_MAX_1M: float = -10.0          # distribution veto: perf_1m < this ...
    DIST_MIN_RVOL: float = 1.3          # ... AND relative_volume_1d > this

    # --- concentration / sizing ---
    N_CORE: int = 2                     # max core positions
    SPLIT: str = "score_weighted"       # "score_weighted" | "equal"
    CORE_SLEEVE: float = 0.85           # fresh capital fraction into the core
    PUNT_SLEEVE: float = 0.15           # seasonal/momentum sleeve (untouched here)
    CASH_RESERVE: float = 0.07          # keep ~7% of fresh capital uninvested
    MAX_LIQUIDITY_FRACTION: float = 0.15  # per-day deploy cap vs daily turnover
    FORBID_SAME_SECTOR: bool = False    # if True, 2nd pick must differ in sector

    OUTPUT_SUBDIR: str = "core_strategy"   # under OUTPUT_DIR

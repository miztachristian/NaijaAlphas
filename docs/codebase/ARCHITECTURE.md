# Architecture

**Analysis Date:** 2026-02-21

## Pattern Overview

**Overall:** Multi-layered analytical pipeline following a modular analyzer pattern.

**Key Characteristics:**
- Clear separation between data acquisition, analysis computation, and result aggregation
- Composable analyzer classes (FundamentalAnalyzer, TechnicalAnalyzer) combined through GrowthAnalyzer
- Configuration-driven behavior enabling year-over-year updates and parameter tuning
- Data caching layer to reduce external API calls and improve performance

## Layers

**Configuration Layer:**
- Purpose: Centralize all system parameters and mappings
- Location: `config/settings.py`
- Contains: Analysis parameters, scoring weights, technical indicators, ticker mappings, sector classifications
- Dependencies: None (pure data/constants)
- Used by: All analysis layers

**Data Acquisition Layer:**
- Purpose: Fetch and cache stock price and financial data
- Location: `core/data_loader.py`
- Contains: NGXDataLoader class, DataCache class for parquet-based caching
- Depends on: Yahoo Finance API (yfinance), local CSV fallback
- Used by: All analyzer classes

**Data Models Layer:**
- Purpose: Define strongly-typed containers for analysis metrics and results
- Location: `core/models.py`
- Contains: FundamentalMetrics, TechnicalMetrics, GrowthScore, StockAnalysisResult, BacktestResult dataclasses
- Depends on: Configuration (for signal thresholds)
- Used by: Analyzer classes and result aggregation

**Fundamental Analysis Layer:**
- Purpose: Extract and score financial health, valuation, profitability, and growth metrics
- Location: `analysis/fundamental.py`
- Contains: FundamentalAnalyzer class with methods for valuation, profitability, growth, and financial health extraction
- Depends on: NGXDataLoader, FundamentalMetrics model
- Used by: GrowthAnalyzer

**Technical Analysis Layer:**
- Purpose: Calculate technical indicators (moving averages, RSI, MACD, momentum, volatility, risk metrics)
- Location: `analysis/technical.py`
- Contains: TechnicalAnalyzer class with 10+ indicator calculation methods
- Depends on: NGXDataLoader, TechnicalMetrics model
- Used by: GrowthAnalyzer

**Growth Scoring Layer:**
- Purpose: Combine fundamental and technical analyses into weighted composite growth scores
- Location: `analysis/growth.py`
- Contains: GrowthAnalyzer class orchestrating analysis pipeline
- Depends on: FundamentalAnalyzer, TechnicalAnalyzer, GrowthScore model, ScoringWeights configuration
- Used by: CLI and batch processors

**Backtesting Layer:**
- Purpose: Simulate historical portfolio performance based on selection strategies
- Location: `analysis/backtest.py`
- Contains: BacktestEngine, Portfolio, Trade classes; portfolio transaction simulation
- Depends on: NGXDataLoader, BacktestConfig
- Used by: Backtest CLI commands

**Utility Layers:**
- Purpose: Data import and visualization support
- Location: `utils/` directory
- Contains: TradingView imports, NGX data fetchers, visualization helpers
- Key files: `tradingview_snapshot.py`, `ngx_data_fetcher.py`, `visualization.py`

**CLI/Interface Layer:**
- Purpose: Command-line interface for user interaction
- Location: `run_analysis.py`, `run_full_analysis.py`, snapshot ranking scripts
- Contains: Argument parsing, function orchestration, result formatting
- Depends on: All analysis layers

## Data Flow

**Single Stock Analysis Flow:**

1. User initiates analysis via CLI with ticker symbol
2. `NGXDataLoader.get_price_series()` fetches cached or fresh price data
3. `FundamentalAnalyzer.analyze()` extracts financial metrics from Yahoo Finance info
4. `TechnicalAnalyzer.analyze()` calculates 10+ technical indicators from price history
5. `GrowthAnalyzer._calculate_growth_score()` combines metrics using weighted formula
6. `StockAnalysisResult` object returned with all metrics and scores
7. CLI formats and displays results

**Full Portfolio Analysis Flow:**

1. `GrowthAnalyzer.analyze_all_stocks()` iterates over ticker list
2. For each ticker: performs single stock analysis (see above)
3. Results collected into DataFrame with columns: ticker, sector, all metrics, scores
4. DataFrame ranked by growth_score (descending)
5. Report generated summarizing top picks, sector rankings, signal distribution
6. Results saved to `outputs/` as CSV and text report

**Backtesting Flow:**

1. `BacktestEngine.run_backtest()` receives ticker list and date range
2. For each rebalancing period (quarterly):
   - `momentum_selection()` ranks stocks by momentum metric
   - Top N stocks selected for portfolio
3. `Portfolio` class simulates:
   - Buy/sell transactions with slippage and transaction costs
   - Position sizing respecting max/min position limits
   - Stop-loss and take-profit triggers
4. Daily portfolio value updated from price data
5. `BacktestResult` object computed with return metrics, Sharpe/Sortino ratios, drawdowns
6. Results saved to `backtests/` directory

**Snapshot/Ranking Flow:**

1. User provides TradingView CSV exports (multiple technical/fundamental columns)
2. `load_tradingview_exports()` in `utils/tradingview_snapshot.py` loads and merges CSVs
3. `rank_snapshot()` in `analysis/snapshot_ranker.py` scores stocks with config-based rules
4. Rankings saved to dated snapshot directory in `data/snapshots/YYYY-MM-DD/`
5. Later: `evaluate_forward_returns()` compares forward returns between two snapshots

**State Management:**
- No persistent application state; all state is in configuration or data files
- Data cache stored as parquet files in `cache/` with JSON metadata
- Analysis results persisted as CSV and text in `outputs/`
- Snapshots persisted as JSON/pickle in `data/snapshots/`

## Key Abstractions

**NGXDataLoader:**
- Purpose: Abstract away data source details (Yahoo Finance vs local CSV)
- Examples: `core/data_loader.py` (lines 100-200)
- Pattern: Implements caching layer with 24-hour expiry; falls back to local CSV if API unavailable

**Analyzer Pattern (Fundamental/Technical):**
- Purpose: Standardize metric extraction and scoring
- Examples: `analysis/fundamental.py`, `analysis/technical.py`
- Pattern: Each analyzer class has:
  - `analyze(ticker)` → returns Metrics dataclass
  - `score_metrics(metrics)` → returns scores dict
  - `generate_analysis_summary()` → returns formatted text output

**GrowthAnalyzer Orchestrator:**
- Purpose: Compose fundamental + technical analysis with weighted scoring
- Location: `analysis/growth.py` (lines 33-200)
- Pattern: Delegates to sub-analyzers, applies weighting formula, ranks results

**Portfolio Simulator:**
- Purpose: Accurately model transaction costs, slippage, and position sizing
- Location: `analysis/backtest.py` (lines 48-150)
- Pattern: Tracks cash, holdings, trade history; computes daily returns

## Entry Points

**Primary CLI (main entry point):**
- Location: `run_analysis.py` (line 358-554)
- Triggers:
  - `--ticker GTCO`: Single stock detailed analysis
  - `--backtest`: Historical strategy backtesting
  - (no args): Full portfolio analysis for target year
  - `--tv-snapshot *.csv`: TradingView snapshot ranking
- Responsibilities: Argument parsing, function delegation, result formatting

**Full Analysis Runner:**
- Location: `run_full_analysis.py`
- Triggers: Batch analysis of all tickers in configuration
- Responsibilities: Orchestrate GrowthAnalyzer.analyze_all_stocks()

**Backtester Entry:**
- Location: `run_analysis.py::run_backtest()`
- Triggers: `python run_analysis.py --backtest --start YEAR`
- Responsibilities: Initialize engine, run simulation, save results

**Snapshot Ranking Entry:**
- Location: `run_analysis.py::run_snapshot_ranking()`
- Triggers: `python run_analysis.py --tv-snapshot file1.csv file2.csv --rank --save`
- Responsibilities: Load CSVs, merge, apply ranking rules, persist results

## Error Handling

**Strategy:** Defensive data validation; return None/empty structures on failure

**Patterns:**

1. **Insufficient Data:**
   - FundamentalAnalyzer/TechnicalAnalyzer: Return metrics with NaN for unavailable fields
   - GrowthAnalyzer: Skip stocks with <50 days of price history
   - CLI: Display error message and continue to next ticker

2. **Missing Data Source:**
   - NGXDataLoader.get_price_series(): Try Yahoo Finance, fall back to local CSV, return None if both fail
   - Analysis proceeds with NaN values (scores still calculated)

3. **Configuration Errors:**
   - ScoringWeights.validate(): Assert weights sum to 1.0 (raised on import if invalid)
   - TechnicalParams: Used as constants; assumed valid

4. **Caching Failures:**
   - DataCache: Silently return None on parquet read errors, re-fetch from API
   - Failed cache writes logged but don't block analysis

## Cross-Cutting Concerns

**Logging:**
- Approach: Python logging module configured in `run_analysis.py::main()` (line 30-33)
- Format: timestamp - name - level - message
- Used minimally; mostly for import warnings and data availability notices

**Validation:**
- Approach: Implicit via data model types (dataclasses) and API contracts
- Configuration validation: ScoringWeights.validate() called in GrowthAnalyzer.__init__()
- Input validation: Min data points check (50 days) in GrowthAnalyzer.analyze_stock()

**Authentication:**
- Approach: None required; Yahoo Finance API is public
- Environment variables: None used (configuration is static in settings.py)

**Performance:**
- Caching: DataCache with 24-hour TTL prevents repeated API calls for same ticker
- Bulk Analysis: GrowthAnalyzer.analyze_all_stocks() reuses loader instance across tickers
- Backtesting: Vectorized operations using pandas/numpy where possible

---

*Architecture analysis: 2026-02-21*

# Codebase Structure

**Analysis Date:** 2026-02-21

## Directory Layout

```
nigeria_stocks/
├── config/                      # Configuration and settings
│   ├── __init__.py
│   └── settings.py              # Central configuration hub
├── core/                        # Data loading and models
│   ├── __init__.py
│   ├── data_loader.py           # NGXDataLoader and DataCache
│   └── models.py                # Dataclass definitions
├── analysis/                    # Analysis engines
│   ├── __init__.py
│   ├── fundamental.py           # Financial metrics extraction
│   ├── technical.py             # Technical indicators calculation
│   ├── growth.py                # Growth scoring orchestrator
│   ├── backtest.py              # Portfolio backtesting engine
│   ├── hidden_gems.py           # Hidden gems discovery (utility)
│   ├── predictor_2026.py        # Year-specific predictor (utility)
│   ├── snapshot_ranker.py       # TradingView snapshot ranking
│   ├── snapshot_store.py        # Snapshot persistence
│   └── snapshot_tracker.py      # Forward returns evaluation
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── ngx_data_fetcher.py      # NGX data acquisition
│   ├── real_data_importer.py    # CSV data import
│   ├── tradingview_importer.py  # TradingView CSV processing
│   ├── tradingview_snapshot.py  # Snapshot loading and merging
│   └── visualization.py         # Plotting and formatting
├── data/                        # Local data storage
│   └── snapshots/               # Dated snapshot directories
│       ├── 2026-01-14/
│       ├── 2026-01-20/
│       └── 2026-02-20/          # etc.
├── cache/                       # Cached data (auto-generated)
│   ├── cache_meta.json
│   └── *.parquet                # Cached price data
├── outputs/                     # Analysis results
│   ├── ngx_analysis_2026_*.csv
│   ├── ngx_top_picks_2026_*.csv
│   └── ngx_report_2026_*.txt
├── backtests/                   # Backtest results
│   └── *.json                   # Backtest performance records
├── notebooks/                   # Jupyter notebooks for exploration
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_ranker_scores.py
│   ├── test_tracker_returns.py
│   └── test_tradingview_merge.py
├── docs/                        # Documentation
├── run_analysis.py              # Primary CLI entry point
├── run_full_analysis.py         # Batch analysis runner
├── create_snapshot.py           # Snapshot creation utility
├── __init__.py                  # Package initialization
├── README.md                    # Project README
└── requirements.txt             # Python dependencies
```

## Directory Purposes

**`config/`:**
- Purpose: Centralized configuration and mappings
- Contains: AnalysisConfig, ScoringWeights, TechnicalParams, BacktestConfig classes; ticker→Yahoo mappings; sector classifications
- Key files: `config/settings.py` (395 lines)
- Update pattern: Modify settings.py to change year, weights, ticker list, or sector outlook

**`core/`:**
- Purpose: Data access and type definitions
- Contains: NGXDataLoader (Yahoo Finance interface), DataCache (parquet caching), model dataclasses
- Key files:
  - `core/data_loader.py` (200+ lines) - handles fetch, fallback, cache
  - `core/models.py` (250+ lines) - FundamentalMetrics, TechnicalMetrics, GrowthScore dataclasses
- Access pattern: All analysis layers import from here

**`analysis/`:**
- Purpose: Stock analysis computation
- Contains: Analyzer classes and portfolio simulation
- Core files:
  - `analysis/fundamental.py` (400+ lines) - extracts P/E, ROE, debt, growth metrics
  - `analysis/technical.py` (400+ lines) - calculates 10+ indicators: RSI, MACD, moving averages, momentum
  - `analysis/growth.py` (300+ lines) - orchestrates both analyzers, applies weighted scoring
  - `analysis/backtest.py` (400+ lines) - Portfolio/Trade classes, strategy simulation
- Utility files:
  - `analysis/snapshot_ranker.py` - Ranks stocks from TradingView exports
  - `analysis/snapshot_tracker.py` - Evaluates forward returns between dates
  - `analysis/hidden_gems.py` - Alternative discovery methods
  - `analysis/predictor_2026.py` - Year-specific models

**`utils/`:**
- Purpose: Data import and visualization helpers
- Contains: Data fetchers, CSV importers, plotting functions
- Key files:
  - `utils/tradingview_snapshot.py` - Loads and merges TradingView CSVs
  - `utils/ngx_data_fetcher.py` - Fetches from NGX website
  - `utils/real_data_importer.py` - Handles local CSV imports
  - `utils/visualization.py` - Plotting and report formatting

**`data/`:**
- Purpose: Local data storage
- Contains: CSV price data, snapshot JSON/pickle files
- Generated: Yes (populated by import scripts)
- Committed: Yes (critical data files tracked)

**`cache/`:**
- Purpose: Performance optimization via data caching
- Contains: Parquet files (price data) and cache_meta.json metadata
- Generated: Yes (auto-created by DataCache)
- Committed: No (auto-generated, 24-hour expiry)

**`outputs/`:**
- Purpose: Analysis result storage
- Contains: CSV files with all stocks' scores, top picks CSV, text reports
- Generated: Yes (created on each analysis run)
- Committed: No (timestamped, not needed in repo)

**`backtests/`:**
- Purpose: Historical strategy performance records
- Contains: JSON files with portfolio value history, trade records, performance metrics
- Generated: Yes (created by backtest engine)
- Committed: No (ephemeral results)

**`tests/`:**
- Purpose: Validation and quality assurance
- Contains: Unit tests for ranking logic, return tracking, data merging
- Key files: `test_ranker_scores.py`, `test_tracker_returns.py`, `test_tradingview_merge.py`

## Key File Locations

**Entry Points:**
- `run_analysis.py`: Main CLI (346 lines) - single stock, full analysis, backtest, snapshot commands
- `run_full_analysis.py`: Alternative batch runner
- `create_snapshot.py`: Utility for creating snapshots from TradingView exports

**Configuration:**
- `config/settings.py`: All parameters (analysis weights, technical params, backtesting config, ticker mappings)
- `requirements.txt`: Python dependencies (pandas, numpy, yfinance, etc.)

**Core Logic:**
- `core/data_loader.py`: NGXDataLoader class - fetches from Yahoo Finance or local CSV
- `analysis/fundamental.py`: FundamentalAnalyzer - extracts valuation, profitability, financial health
- `analysis/technical.py`: TechnicalAnalyzer - calculates indicators and momentum
- `analysis/growth.py`: GrowthAnalyzer - orchestrates analysis and applies weighted scoring
- `analysis/backtest.py`: BacktestEngine - portfolio simulation with transactions

**Testing:**
- `tests/test_ranker_scores.py`: Tests snapshot ranking logic
- `tests/test_tracker_returns.py`: Tests forward return calculation
- `tests/test_tradingview_merge.py`: Tests CSV merging

## Naming Conventions

**Files:**
- Modules: lowercase with underscores (e.g., `data_loader.py`, `fundamental.py`)
- Classes: PascalCase in modules (e.g., `NGXDataLoader`, `FundamentalAnalyzer`)
- Entry scripts: lowercase underscore (e.g., `run_analysis.py`, `create_snapshot.py`)
- Utilities: descriptive with focus (e.g., `tradingview_snapshot.py`)

**Directories:**
- Package dirs: lowercase plural or module-specific (e.g., `analysis/`, `core/`, `utils/`)
- Data dirs: lowercase (e.g., `data/`, `cache/`, `outputs/`)
- Snapshot dirs: date format YYYY-MM-DD (e.g., `data/snapshots/2026-01-14/`)

**Variables/Functions:**
- Variables: snake_case (e.g., `growth_score`, `pe_ratio`, `technical_metrics`)
- Functions: snake_case (e.g., `analyze_stock()`, `calculate_growth_score()`)
- Constants: UPPER_SNAKE_CASE (e.g., `TARGET_YEAR`, `MIN_DATA_POINTS`)
- Dataclass fields: snake_case (e.g., `pe_ratio`, `roe`, `momentum_12m`)

**Classes:**
- Main analyzers: {Type}Analyzer (e.g., `FundamentalAnalyzer`, `TechnicalAnalyzer`)
- Data containers: {Type}Metrics or {Type}Result (e.g., `FundamentalMetrics`, `StockAnalysisResult`)
- Business logic: {Noun}Engine or {Noun} (e.g., `BacktestEngine`, `Portfolio`)
- Configuration: {Type}Config or {Type}Params (e.g., `AnalysisConfig`, `TechnicalParams`)

## Where to Add New Code

**New Feature (e.g., add sentiment analysis):**
- Primary code: `analysis/sentiment.py` (new file following Analyzer pattern)
- Integration: Import and compose in `analysis/growth.py::GrowthAnalyzer`
- Configuration: Add weights and parameters to `config/settings.py`
- Tests: `tests/test_sentiment.py`

**New Metric/Indicator:**
- If fundamental: Add extraction method to `FundamentalAnalyzer._extract_*()` in `analysis/fundamental.py`
- If technical: Add calculation method to `TechnicalAnalyzer._calculate_*()` in `analysis/technical.py`
- Update corresponding dataclass fields in `core/models.py`
- Update scoring logic in `GrowthAnalyzer._calculate_component_score()` if weighted

**New Data Source:**
- Create new fetcher class in `utils/` (e.g., `utils/bloomberg_fetcher.py`)
- Add fallback option to `NGXDataLoader.get_price_series()` in `core/data_loader.py`
- Update configuration to support new source selection

**Batch Processing Script:**
- Location: Root directory (e.g., `batch_analysis.py`, `process_quarterly.py`)
- Pattern: Import GrowthAnalyzer, call analyze_all_stocks(), save results
- Example: See `run_analysis.py::run_full_analysis()` (lines 116-169)

**Utility Function:**
- Location: `utils/` directory with descriptive module name
- Example: `utils/tradingview_snapshot.py`, `utils/visualization.py`
- Export: Add to `utils/__init__.py` for visibility

**Snapshot Feature:**
- Core logic: `analysis/snapshot_*.py` (ranker, store, tracker pattern)
- Data I/O: `utils/tradingview_*.py` (import, snapshot handling)
- Directory: Results saved to `data/snapshots/YYYY-MM-DD/` subdirectories

## Special Directories

**`data/snapshots/`:**
- Purpose: Version-controlled snapshots for point-in-time stock rankings
- Structure: Each snapshot in dated subdirectory with rankings JSON and merged CSV
- Generated: Yes (by snapshot ranking commands)
- Committed: Yes (snapshots are analysis artifacts)
- Access: Via snapshot_tracker for forward return evaluation

**`cache/`:**
- Purpose: Automatic performance optimization
- Structure: Parquet files with hash-based naming, metadata in JSON
- Generated: Yes (auto-created by DataCache on first access)
- Committed: No (.gitignore includes this)
- Cleanup: 24-hour TTL; older entries re-fetched automatically

**`outputs/` and `backtests/`:**
- Purpose: Temporary result storage
- Structure: Timestamped files (format: `{type}_{year}_{timestamp}.{ext}`)
- Generated: Yes (created by CLI on each run)
- Committed: No (ephemeral results)
- Lifecycle: Can be deleted; will be recreated on next run

---

*Structure analysis: 2026-02-21*

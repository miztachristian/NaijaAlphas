# External Integrations

**Analysis Date:** 2026-02-21

## APIs & External Services

**Financial Data:**
- Yahoo Finance - OHLCV historical price data for NGX stocks
  - SDK/Client: `yfinance` library (`core/data_loader.py`)
  - Symbols: NGX tickers with `.LG` suffix mapping (e.g., `GTCO.LG`, `ZENITHBANK.LG`)
  - Authentication: None required (public API)
  - Fallback: Local CSV files in `data/` directory
  - Implementation: `NGXDataLoader.download_yahoo()` method with 30-second timeout, auto_adjust=False for data integrity

**Alternative Data Sources (Prepared but Not Active):**
- TradingView - Manual CSV export support
  - Client: `utils/tradingview_importer.py` and `utils/tradingview_snapshot.py`
  - Method: Manual file import; no API integration
  - Format: CSV exports with comprehensive fundamentals, technicals, performance metrics

- Investing.com - Mentioned in documentation
  - Status: Not yet implemented
  - Note: `utils/ngx_data_fetcher.py` includes placeholder for future integration

- NGX Group Data Portal - Paid premium data
  - Status: Documented but not integrated
  - Use case: More reliable Nigerian exchange data when available

**HTTP Client:**
- requests library - Used in `utils/ngx_data_fetcher.py` for API calls
  - User-Agent: Standard Mozilla/5.0 string for compatibility
  - Session management: Persistent session with custom headers
  - Timeout: Not explicitly set in current implementation

## Data Storage

**Databases:**
- None (no SQL database)

**File Storage:**
- Local filesystem only
  - **Data Directory** (`data/`): CSV files with stock OHLCV data
  - **Cache Directory** (`cache/`): Parquet binary files with 24-hour expiry
  - **Backtest Directory** (`backtests/`): Historical backtest results
  - **Output Directory** (`outputs/`): Generated analysis reports and visualizations

**Caching:**
- File-based caching system in `core/data_loader.py`
  - Implementation: `DataCache` class using parquet format
  - Expiry: Configurable hours (default: 24 hours)
  - Metadata: JSON tracking at `cache/cache_meta.json`
  - Cache Key: MD5 hash of ticker symbol (16-character prefix)
  - Multi-tier: Memory cache (dict) + file cache (parquet) + source (API/CSV)

## Authentication & Identity

**Auth Provider:**
- None required (all integrations are unauthenticated/public)
- Yahoo Finance: Public API, no credentials needed
- Local data: File system access based on OS permissions

## Monitoring & Observability

**Error Tracking:**
- None (no external service)

**Logs:**
- console output via print statements
  - Data loader progress: `NGXDataLoader.get_multiple_stocks()` prints ticker progress
  - Download warnings: `NGXDataLoader.download_yahoo()` prints errors to console
- Optional logging via Python `logging` module in analysis modules (`analysis/growth.py`, `analysis/snapshot_ranker.py`)
  - Not centrally configured; uses Python's standard logging
  - Log files in various analysis modules but no aggregated logging

## CI/CD & Deployment

**Hosting:**
- Local/desktop application (no hosting platform)
- Designed for manual execution and Jupyter notebook analysis
- `.planning/codebase` suggests GSD (Get Shit Done) integration for future automation

**CI Pipeline:**
- None detected
- Tests exist (`tests/` directory) but no CI/CD configuration found
- Test execution: Manual via pytest

## Environment Configuration

**Required Environment Variables:**
- None currently required
- All configuration via `config/settings.py` Python class-based approach
- YAML configuration in `config/snapshot_ranker.yaml`

**Secrets Location:**
- No secrets management (no API keys, credentials, or sensitive data)
- Project designed for public data only
- Could add `.env` file support if paid data sources are integrated

## Data Acquisition Priority

**Fallback Chain:**
1. **Memory cache** - Fast in-memory Python dict (session-local)
2. **File cache** - Parquet files in `cache/` with expiry checking
3. **Yahoo Finance API** - `yfinance.download()` with yfinance library (requires internet)
4. **Local CSV** - CSV files in `data/` directory (offline mode)

**Source Selection Logic:**
- `NGXDataLoader.get_price_data()` attempts sources in priority order
- If Yahoo Finance unavailable, automatically falls back to CSV
- Cache validity checked before attempting fresh download (24-hour expiry by default)
- No internet connection required if CSV files present

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Feature-Specific Integrations

**Snapshot Ranker:**
- Configuration: `config/snapshot_ranker.yaml`
- Implementation: `analysis/snapshot_ranker.py` and `analysis/snapshot_tracker.py`
- Features: Loads TradingView or manual snapshot data with feature weighting
- Gates: Configurable thresholds for aggressive vs guardrails ranking modes
- Data format: CSV imports with standardized field mapping

**Technical Analysis:**
- Implementation: `analysis/technical.py`
- Indicators: RSI, MACD, Bollinger Bands, moving averages calculated from price data
- Parameters: Configurable in `config/settings.py` `TechnicalParams` class
- No external service dependency

**Backtesting:**
- Implementation: `analysis/backtest.py`
- Mode: Historical simulation using local cached price data
- Capital: Configurable initial portfolio value (default: ₦10M)
- No live trading or broker integration

---

*Integration audit: 2026-02-21*

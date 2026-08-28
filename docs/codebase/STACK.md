# Technology Stack

**Analysis Date:** 2026-02-21

## Languages

**Primary:**
- Python 3.x - Entire application (data analysis, analysis, core logic)

**Secondary:**
- YAML - Configuration files (`config/snapshot_ranker.yaml`)
- Jupyter Notebooks - Interactive analysis and visualization

## Runtime

**Environment:**
- Python 3.x with virtual environment (`.venv/`)

**Package Manager:**
- pip (Python Package Manager)
- Lockfile: Not detected (requirements.txt used for pinning)

## Frameworks

**Core:**
- pandas 1.3.0+ - Data manipulation, time series analysis, OHLCV data handling
- numpy 1.21.0+ - Numerical computations, array operations, financial calculations

**Data Acquisition:**
- yfinance 0.2.0+ - Yahoo Finance API integration for OHLCV stock price data
- requests - HTTP client for API calls and data fetching (used in `utils/ngx_data_fetcher.py`)

**Data Storage & Serialization:**
- pyarrow 8.0.0+ - Parquet file format support for caching binary data
- PyYAML 6.0.0+ - YAML configuration file parsing

**Visualization:**
- matplotlib 3.4.0+ - Static plotting and chart generation
- seaborn 0.11.0+ - Statistical data visualization
- plotly 5.0.0+ - Interactive web-based visualizations

**Jupyter/Notebook Support:**
- jupyter 1.0.0+ - Notebook environment
- ipykernel 6.0.0+ - IPython kernel for notebook execution
- JupyterLab (installed in `.venv/`)

**Testing:**
- pytest 7.0.0+ - Test runner and assertion framework

**Progress & Display:**
- tqdm 4.62.0+ - Progress bars for long-running operations

## Key Dependencies

**Critical:**
- `yfinance` - Primary external API for stock market data; fallback to local CSV when unavailable
- `pandas` - Core data manipulation framework; handles all time series and tabular data
- `numpy` - Numerical computations for financial metrics and technical indicators

**Infrastructure:**
- `pyarrow` - Binary serialization for cache layer; improves data load performance
- `pyyaml` - Configuration management for analysis parameters and ranker settings
- `requests` - HTTP requests library for alternative data source integration (preparing for NGX APIs)

## Configuration

**Environment:**
- Configuration class-based approach in `config/settings.py`
- Settings include analysis parameters, technical indicators, backtesting parameters
- Sector mappings and Yahoo Finance ticker mappings hardcoded in settings
- YAML-based feature weights in `config/snapshot_ranker.yaml`

**Build:**
- No build configuration required (pure Python project)
- Virtual environment managed via `.venv/` directory
- Python path manipulation via `sys.path.insert()` in modules for relative imports

## Data Storage

**Local:**
- CSV files in `data/` directory - Source data for stocks (fallback when API unavailable)
- Parquet files in `cache/` directory - Binary cache for improved loading performance
- JSON metadata in `cache/cache_meta.json` - Cache expiry tracking
- Backtest results in `backtests/` directory - Historical analysis outputs
- Analysis outputs in `outputs/` directory - Generated reports and visualizations

**Caching:**
- Two-tier cache system: memory cache + file-based parquet cache with 24-hour expiry (configurable)
- Cache expiry mechanism with JSON metadata tracking timestamps
- Cache invalidation based on `AnalysisConfig.LOOKBACK_YEARS` (default: 5 years)

## Platform Requirements

**Development:**
- Python 3.6+ (recommended 3.8+)
- pip package manager
- Virtual environment (`.venv/`)
- ~2GB disk space for dependencies and cache
- Internet access for Yahoo Finance API (optional - falls back to local CSV)

**Production:**
- Python 3.6+ runtime
- Sufficient disk space for parquet cache files
- Optional: Internet connectivity for real-time data, but local CSV mode is fully functional offline

## External Data Sources

**Primary:**
- Yahoo Finance API (via `yfinance` library) - OHLCV data for `.LG` suffix symbols
- Local CSV files - Manual imports or broker exports stored in `data/` directory

**Alternative/Planned:**
- TradingView CSV exports - Manual import via `utils/tradingview_importer.py` and `utils/tradingview_snapshot.py`
- Investing.com API - Mentioned as alternative but not implemented
- NGX Group Data Portal - Paid data source mentioned but not integrated

## Dependency Notes

**Version Pinning:**
- All dependencies use minimum version constraints (>=) not specific versions
- Allows flexibility but may require compatibility testing with newer versions
- No lockfile (requirements.lock) present; consider adding for production stability

**Optional Dependencies:**
- Visualization packages (matplotlib, seaborn, plotly) can be excluded for headless analysis
- Jupyter packages only needed for interactive analysis

---

*Stack analysis: 2026-02-21*

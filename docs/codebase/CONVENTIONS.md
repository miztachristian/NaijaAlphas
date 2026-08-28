# Coding Conventions

**Analysis Date:** 2026-02-21

## Naming Patterns

**Files:**
- Module files: `snake_case.py` (e.g., `data_loader.py`, `technical.py`, `snapshot_ranker.py`)
- Test files: `test_*.py` (e.g., `test_ranker_scores.py`, `test_tracker_returns.py`)
- Entry point scripts: `snake_case.py` (e.g., `run_analysis.py`, `run_full_analysis.py`)
- Configuration: `settings.py` in `config/` directory

**Functions:**
- Function names: `snake_case` (e.g., `get_price_data()`, `calculate_moving_averages()`, `score_technicals()`)
- Private functions: Prefix with `_` (e.g., `_calculate_moving_averages()`, `_score_momentum()`)
- Helper functions: Descriptive snake_case names
- Class methods follow same snake_case convention

**Variables:**
- Local variables: `snake_case` (e.g., `price_data`, `moving_avg`, `ticker_symbol`)
- Constants: `UPPER_CASE` (e.g., `LOOKBACK_YEARS`, `RISK_FREE_RATE`, `SMA_SHORT`)
- Private variables (class): `_snake_case` prefix (e.g., `self._cache_dir`, `self._meta_file`)
- DataFrame columns: `snake_case` with underscores (e.g., `perf_3m`, `volume_1d`, `revenue_growth_ttm`)

**Types:**
- Dataclasses: `PascalCase` (e.g., `FundamentalMetrics`, `TechnicalMetrics`, `GrowthScore`)
- Config classes: `PascalCase` (e.g., `AnalysisConfig`, `TechnicalParams`, `BacktestConfig`)

## Code Style

**Formatting:**
- Line length: Implicit ~100 characters (observed in codebase)
- Indentation: 4 spaces (standard Python)
- No explicit formatting tool configured (no `.black`, `.flake8`, `.pylintrc`, `pyproject.toml` detected)
- Code style appears to follow PEP 8 conventions by convention

**Linting:**
- No linter configured (no `.eslintrc`, `.flake8`, or equivalent found)
- Code maintained through manual adherence to conventions

**Docstrings:**
- Format: Google-style docstrings for classes and functions
- Required for public APIs (modules, classes, public methods)
- Example from `core/data_loader.py`:
  ```python
  def download_yahoo(
      self,
      ticker: str,
      yahoo_symbol: str,
      period_years: int = None
  ) -> Optional[pd.DataFrame]:
      """
      Download data from Yahoo Finance.

      Args:
          ticker: NGX ticker symbol
          yahoo_symbol: Yahoo Finance symbol (with .LG suffix)
          period_years: Years of historical data

      Returns:
          DataFrame with OHLCV data or None
      """
  ```

## Import Organization

**Order:**
1. Standard library imports (e.g., `json`, `warnings`, `sys`, `pathlib`)
2. Third-party imports (e.g., `pandas`, `numpy`, `yfinance`)
3. Local imports (e.g., `from config.settings import ...`)

**Pattern observed in `core/data_loader.py`:**
```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import hashlib
import warnings

# Optional dependencies with try/except
try:
    import yfinance as yf
except ImportError:
    yf = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    TICKER_TO_YAHOO,
    DATA_DIR,
    CACHE_DIR,
)
```

**Path aliases:**
- No explicit aliases (no `jsconfig.json` or `tsconfig.json` equivalents)
- Relative imports use `sys.path.insert(0, str(Path(__file__).parent.parent))` pattern
- Prefer absolute imports from project root via sys.path manipulation

## Error Handling

**Patterns:**
- Try/except blocks for external API calls: `yfinance`, CSV loading, parquet read/write
- Graceful degradation: Return `None` on failure instead of raising exceptions
- Silent failures for optional dependencies (e.g., yfinance ImportError)
- Silent failures for cache operations (invalid cached data returns None, not exception)
- Data validation: Check `df.empty`, `None` checks before processing
- No custom exception classes observed; use built-in exceptions only

**Example from `core/data_loader.py`:**
```python
try:
    df = pd.read_parquet(cache_file)
except Exception:
    return None  # Silent fallback
```

**Example from `analysis/technical.py`:**
```python
if prices is None or len(prices) < TechnicalParams.SMA_SHORT:
    return metrics  # Return partial metrics if insufficient data
```

## Logging

**Framework:** No logging framework imported (no `logging` module, no `loguru`, no `print`-based logging)

**Approach:**
- Print-based console output for progress (e.g., `print(f"  Loading {ticker} ({i+1}/{total})...", end="\r")`)
- No structured logging or log levels
- Progress indicators use inline `print()` statements
- Example from `core/data_loader.py`:
  ```python
  if show_progress:
      print(f"  Loaded {len(results)}/{total} stocks" + " " * 20)
  ```

**When to use:**
- Progress output: Use print with `end="\r"` for inline progress
- Warnings: Use `warnings.filterwarnings("ignore")` to suppress (as in `core/data_loader.py`)
- Errors: Print error message directly, return None

## Comments

**When to Comment:**
- Explain complex algorithms (e.g., RSI calculation, momentum scoring)
- Clarify non-obvious business logic (e.g., "12-1 momentum factor (skip most recent month)")
- Document data transformations and assumptions
- Do NOT comment obvious code

**JSDoc/TSDoc:**
- Not applicable (Python project)
- Use Google-style docstrings instead (see "Code Style" section)

## Function Design

**Size:**
- Functions: 10-50 lines typical
- Analytical functions: 50+ lines acceptable for comprehensive calculations
- Private helper methods: 10-30 lines

**Parameters:**
- Use type hints consistently (e.g., `ticker: str`, `period_years: int = None`)
- Return type hints required (e.g., `-> Optional[pd.DataFrame]`, `-> float`)
- Accept DataFrame columns as strings, not raw values
- Use keyword arguments for optional parameters
- Default to `None` for optional parameters

**Return Values:**
- Return `None` on failure (no exceptions)
- Return computed value for success
- Return dictionaries for multi-value results (e.g., scores dictionary)
- Return DataFrames for multi-row results

## Module Design

**Exports:**
- Classes (e.g., `TechnicalAnalyzer`, `NGXDataLoader`, `DataCache`)
- Public functions (e.g., `load_ticker_data()`, `rank_snapshot()`)
- Data models (dataclasses like `TechnicalMetrics`)
- Configuration classes

**Barrel Files:**
- Used in `core/__init__.py` and `analysis/__init__.py`
- Example from `core/__init__.py`:
  ```python
  from .data_loader import NGXDataLoader, DataCache, load_ticker_data, load_all_tickers
  from .models import (
      FundamentalMetrics,
      TechnicalMetrics,
      GrowthScore,
      StockAnalysisResult,
  )
  ```
- Pattern: Import specific classes/functions, re-export for module-level access

**Typical module structure:**
```
module_name/
├── __init__.py         # Barrel file with exports
├── data_loader.py      # Data acquisition
├── models.py          # Data models (dataclasses)
├── analyzer.py        # Analysis logic
└── config.py          # Configuration
```

## Dataclass Usage

**Pattern:** Use `@dataclass` for data models instead of regular classes
- Located in `core/models.py`
- Includes default NaN values for optional metrics
- Provide `to_dict()` method for serialization
- Example:
  ```python
  @dataclass
  class TechnicalMetrics:
      ticker: str
      price: float = np.nan
      sma_50: float = np.nan
      # ... many more fields
  ```

## Configuration Management

**Pattern:** Centralized configuration classes in `config/settings.py`
- `AnalysisConfig`: Core analysis parameters (lookback period, risk-free rate, etc.)
- `TechnicalParams`: Technical indicator parameters (SMA periods, RSI periods, etc.)
- `BacktestConfig`: Backtesting parameters (capital, transaction costs, rebalancing, etc.)
- `ScoringWeights`: Component weights for composite scoring
- Functions: `get_sector_outlook()`, `get_ticker_sector()`, `get_all_tickers()`

**Access pattern:**
```python
from config.settings import AnalysisConfig, TechnicalParams
period_years = AnalysisConfig.LOOKBACK_YEARS
sma_period = TechnicalParams.SMA_SHORT
```

## Type Hints

**Usage:** Type hints used throughout codebase
- Function parameters: Always included
- Return types: Always included
- Local variables: Minimal (not required)
- DataFrame/Series type hints: Use `pd.DataFrame`, `pd.Series`
- Optional values: Use `Optional[T]` from `typing`

**Common patterns:**
```python
def get_price_data(
    self,
    ticker: str,
    force_refresh: bool = False
) -> Optional[pd.DataFrame]:
    ...

def get_multiple_stocks(
    self,
    tickers: List[str],
    show_progress: bool = True
) -> Dict[str, pd.DataFrame]:
    ...
```

---

*Convention analysis: 2026-02-21*

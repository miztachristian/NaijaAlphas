# Codebase Concerns

**Analysis Date:** 2026-02-21

## Tech Debt

**Bare Exception Handlers:**
- Issue: Multiple files use bare `except:` or catch-all `except Exception` without proper logging or re-raising
- Files: `utils\real_data_importer.py` (line 204), `utils\tradingview_snapshot.py`
- Impact: Silent failures make debugging difficult; errors are swallowed with no trace
- Fix approach: Replace bare `except:` with specific exception types and log errors using `logging.exception()` before re-raising or handling gracefully

**Suppress All Warnings:**
- Issue: `warnings.filterwarnings("ignore")` in `core\data_loader.py` (line 17) masks all deprecation and performance warnings
- Impact: Future breaking changes in dependencies go unnoticed until runtime failures occur
- Fix approach: Replace global filter with specific warning suppressions for known safe cases

**Hardcoded Magic Numbers:**
- Issue: Threshold values scattered throughout code (e.g., minimum trade size of 100 in `analysis\backtest.py` line 504, 252 trading days assumption, 30-day rebalance delta in line 399)
- Files: `analysis\backtest.py`, `analysis\snapshot_ranker.py`
- Impact: Difficult to adjust parameters; values not validated; inconsistent with settings module
- Fix approach: Move all magic numbers to `config\settings.py` as class constants with documentation

**sys.path.insert() Anti-pattern:**
- Issue: Multiple files use `sys.path.insert(0, str(Path(__file__).parent.parent))` for imports (e.g., `analysis\growth.py` line 14)
- Files: 10+ files in analysis/, core/, utils/
- Impact: Makes package structure unclear; breaks relative imports; fragile to refactoring
- Fix approach: Use proper package structure with `__init__.py` files and relative imports instead

## Known Bugs

**Portfolio Trade Closing Logic:**
- Symptoms: When selling partially from position, trade matching uses `reversed()` iteration which matches LIFO but doesn't account for FIFO allocation in real trading
- Files: `analysis\backtest.py` (lines 176-179)
- Trigger: Run backtest, sell partial position from same stock bought multiple times
- Workaround: None - affected trades will show incorrect cost basis
- Fix approach: Implement proper trade closing strategy (FIFO/LIFO) as parameter; match partial sells correctly

**Division by Zero in Volatility Calculation:**
- Symptoms: When `downside_vol` equals 0 in Sortino ratio calculation, division by zero occurs silently (NaN result)
- Files: `analysis\backtest.py` (line 556)
- Trigger: Stock with zero downside returns in period (never goes down)
- Workaround: Sortino Ratio just shows NaN, not caught as error
- Fix approach: Check `if downside_vol > 0:` and return sensible default (e.g., use std dev of all returns)

**Missing Data Handling in Rankings:**
- Symptoms: `.fillna(0)` used in score calculations (e.g., `analysis\snapshot_ranker.py` line 305-308) treats missing values as zero, biasing results downward for stocks with incomplete data
- Files: `nvidia_stocks v2\analysis\snapshot_ranker.py` (lines 305-320)
- Trigger: Stock with missing momentum data gets treated as 0 momentum instead of neutral
- Impact: Stocks with sparse data appear worse than they are
- Fix approach: Use proper imputation (sector median) or penalize coverage separately instead of defaulting to zero

## Security Considerations

**No Input Validation:**
- Risk: CSV data imported directly without validating format, types, or malicious content
- Files: `utils\real_data_importer.py` (line 213, 243)
- Current mitigation: None
- Recommendations: Add schema validation before DataFrame operations; validate ticker symbols against whitelist

**Unvalidated External Data:**
- Risk: Yahoo Finance data and TradingView CSVs loaded without checking data ranges or anomalies
- Files: `core\data_loader.py` (lines 160-194)
- Current mitigation: Columns are standardized but no range checks
- Recommendations: Add bounds checking for prices, volumes; flag missing data periods

## Performance Bottlenecks

**Data Loader Inefficiency:**
- Problem: Each analysis pass reloads full historical data even for cached results. No incremental loading.
- Files: `core\data_loader.py`, `analysis\growth.py` (line 94)
- Cause: Cache expires after 24 hours; no date-aware partial loads; loading 5 years of data for daily metrics
- Improvement path: Implement incremental cache updates; cache individual metrics separately from price history; reduce lookback for daily indicators

**Inefficient Backtest Date Processing:**
- Problem: All trading dates stored in memory; rebalance dates generated repeatedly; no indexing
- Files: `analysis\backtest.py` (lines 327-331, 336-356)
- Cause: Loop over 1000+ dates checking prices; O(n*m) complexity for n dates × m tickers
- Improvement path: Use pandas DatetimeIndex for fast lookups; pre-compute trading dates once; vectorize price lookups

**Ranking Computation on Full Dataset:**
- Problem: Winsorizing, ranking, and scoring computed on entire dataset even when only top 10 stocks needed
- Files: `analysis\snapshot_ranker.py` (lines 250-372)
- Cause: No lazy evaluation; sector rank calculation full scan for each feature
- Improvement path: Early filter to sector before ranking; implement streaming rank updates

## Fragile Areas

**Backtesting Portfolio State:**
- Files: `analysis\backtest.py` (Portfolio class)
- Why fragile: Holdings tracking uses dict with floating-point keys; averaging prices not thread-safe; trade history not atomic
- Safe modification: Add preconditions check (assert holdings >= 0); test edge cases (zero shares, NaN prices, negative cash)
- Test coverage: No unit tests for Portfolio class; backtest results not validated against known benchmarks

**Data Loader Caching:**
- Files: `core\data_loader.py` (DataCache class)
- Why fragile: Meta file JSON can get corrupted; cache key collisions possible with MD5 truncation; no locking for concurrent access
- Safe modification: Lock file access; validate cache files on load; implement atomic writes
- Test coverage: No tests for cache expiry, corruption recovery, or concurrency

**Scoring Model Assumptions:**
- Files: `analysis\growth.py` (GrowthScore calculation, lines 172-229)
- Why fragile: Default scores of 0.3-0.5 used when metrics missing; weights don't sum correctly in edge cases; no validation of component scores
- Safe modification: Add assertions that weights sum to 1.0; require minimum data before scoring; use NaN instead of defaults
- Test coverage: Unit tests exist (`tests\test_ranker_scores.py`) but only test happy path

## Scaling Limits

**In-Memory Data Storage:**
- Current capacity: ~200MB for 100 stocks × 5 years daily data
- Limit: Will hit memory ceiling adding more tickers or longer history (>500 stocks)
- Scaling path: Implement batch processing by sector; use data streaming for backtests; cache results to disk with TTL

**Backtesting Time Complexity:**
- Current capacity: 100 stocks, 5 years, monthly rebalance = ~100 seconds
- Limit: 1000+ stocks or daily rebalance becomes unusable
- Scaling path: Vectorize portfolio updates with pandas operations; use numba for indicator calculations; implement parallel backtests by year

## Dependencies at Risk

**yfinance Dependency:**
- Risk: Hard dependency on Yahoo Finance API which breaks periodically; no fallback when API unavailable
- Impact: All analysis fails silently when API returns empty data
- Migration plan: Add AlphaVantage or FRED API as secondary source; implement graceful degradation to CSV-only mode

**Deprecated pandas Operations:**
- Risk: Code uses `.append()` (deprecated) in some places; `.ix` indexer removed in newer pandas
- Impact: Future pandas upgrade (2.0+) will break code
- Migration plan: Audit all pandas operations; use `.concat()` instead of `.append()`; test against pandas 2.0

## Missing Critical Features

**No Data Validation Layer:**
- Problem: Can't tell if analysis result is based on 100 data points or 1; missing data silently propagated
- Blocks: Confidence scoring, result reliability assessment
- Fix approach: Track data quality metrics alongside each result; fail loudly when data below threshold

**No Error Recovery:**
- Problem: Single stock load failure stops entire analysis; no partial results
- Blocks: Large-scale screening with reliable results
- Fix approach: Implement try/except at stock level; collect errors; return partial results with error metadata

**No Configuration Validation:**
- Problem: Weights in ScoringWeights must sum to 1.0 but checked at runtime via `.validate()` call, not enforced
- Blocks: Configuration mistakes not caught until analysis runs
- Fix approach: Use Pydantic models or dataclass validators; validate at import time

## Test Coverage Gaps

**Core Classes Not Tested:**
- What's not tested: Portfolio class (buy/sell logic), DataLoader caching mechanism, technical indicator calculations
- Files: `analysis\backtest.py` (Portfolio), `core\data_loader.py` (NGXDataLoader)
- Risk: Trading logic bugs go undetected; cache corruption silent
- Priority: High - backtest results depend on Portfolio correctness

**No Integration Tests:**
- What's not tested: End-to-end analysis pipeline (load data → analyze → rank → backtest)
- Files: No integration test file exists
- Risk: Individual components pass tests but break together
- Priority: High - critical for validating system behavior

**No Numerical Validation:**
- What's not tested: Extreme values (0.01 stocks, negative prices), NaN handling, division edge cases
- Files: `analysis\technical.py`, `analysis\backtest.py`
- Risk: Undefined behavior with edge data
- Priority: Medium - rare but catastrophic when occurs

**Data Quality Not Tested:**
- What's not tested: CSV parsing with malformed input, missing columns, type mismatches
- Files: `utils\real_data_importer.py`, `utils\tradingview_snapshot.py`
- Risk: Bad data silently produces garbage results
- Priority: Medium - affects data pipeline reliability

---

*Concerns audit: 2026-02-21*

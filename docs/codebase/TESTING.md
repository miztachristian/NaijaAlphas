# Testing Patterns

**Analysis Date:** 2026-02-21

## Test Framework

**Runner:**
- pytest 7.0.0+
- Config: No dedicated `pytest.ini` or `pyproject.toml` config file
- Run via: `pytest tests/` or `python -m pytest`

**Assertion Library:**
- Standard `assert` statements (no additional assertion library like `pytest-assertions`)

**Run Commands:**
```bash
pytest tests/                           # Run all tests
pytest tests/ -v                        # Verbose output
pytest tests/test_ranker_scores.py      # Run single test file
pytest tests/test_ranker_scores.py::TestCoverageScore::test_full_coverage  # Single test
```

## Test File Organization

**Location:**
- Co-located with source code in `tests/` directory at project root
- One test file per module under analysis (not strict 1:1 but common pattern)
- Example: `tests/test_ranker_scores.py` tests `analysis/snapshot_ranker.py`

**Naming:**
- Test files: `test_*.py` (e.g., `test_ranker_scores.py`, `test_tracker_returns.py`)
- Test classes: `Test*` prefix (e.g., `TestCoverageScore`, `TestLiquidityValue`)
- Test methods: `test_*` prefix (e.g., `test_full_coverage`, `test_missing_symbol_handling`)

**Structure:**
```
tests/
├── __init__.py
├── test_ranker_scores.py        # Tests for snapshot ranking logic
├── test_tracker_returns.py       # Tests for forward return tracking
└── test_tradingview_merge.py     # Tests for data merging
```

## Test Structure

**Suite Organization:**
- Organize related tests into classes (e.g., `TestCoverageScore` groups coverage-related tests)
- Each class groups logically related test cases
- Classes inherit from nothing (no base test class required)

**From `tests/test_ranker_scores.py`:**
```python
class TestCoverageScore:
    """Tests for coverage score calculation."""

    def test_full_coverage(self):
        """Test that full coverage returns 1.0."""
        df = pd.DataFrame({...})
        coverage = compute_coverage_score(df)
        assert coverage.iloc[0] == 1.0
```

**Patterns:**
- **Setup:** Use `@pytest.fixture` for reusable test data
  ```python
  @pytest.fixture
  def temp_snapshot_dir(self):
      """Create temporary directory for snapshots."""
      with tempfile.TemporaryDirectory() as tmpdir:
          yield Path(tmpdir)
  ```

- **Teardown:** Context managers (`with`) handle cleanup automatically

- **Assertion:** Simple `assert` statements with descriptive test names
  ```python
  assert coverage.iloc[0] == 1.0
  assert abs(result - expected) < 0.01  # For float comparisons
  ```

## Fixtures

**Test Data:**
- `@pytest.fixture` decorators for reusable test fixtures
- Fixtures passed as method arguments (fixture injection)
- Example from `tests/test_tracker_returns.py`:
  ```python
  @pytest.fixture
  def temp_snapshot_dir(self):
      """Create temporary directory for snapshots."""
      with tempfile.TemporaryDirectory() as tmpdir:
          yield Path(tmpdir)
  ```

**Helper Methods:**
- Use instance methods (not functions) for test helpers
- Example from `tests/test_tracker_returns.py`:
  ```python
  def create_test_snapshot(self, store, date, symbols, prices, scores):
      """Helper to create a test snapshot."""
      merged_df = pd.DataFrame({...})
      ranking_df = pd.DataFrame({...})
      # ... setup code
      store.save_snapshot(date=date, ...)
  ```

- Call helpers from within test methods:
  ```python
  def test_basic_return_calculation(self, temp_snapshot_dir):
      store = SnapshotStore(temp_snapshot_dir)
      self.create_test_snapshot(store, '2026-01-01', ...)
  ```

## Mocking

**Framework:** `tempfile` module (not unittest.mock)

**Patterns:**
- Use `tempfile.TemporaryDirectory()` for file system tests
- Use `pd.DataFrame()` constructor for data mocking
- Create inline test DataFrames rather than fixtures for single-use cases
- Example from `tests/test_ranker_scores.py`:
  ```python
  df = pd.DataFrame({
      'price': [100.0],
      'volume_1d': [1000000],
      'perf_3m': [10.0],
      # ... column setup
  })
  coverage = compute_coverage_score(df)
  ```

**What to Mock:**
- File system operations (use `TemporaryDirectory`)
- External data stores (use in-memory DataFrames or temporary files)
- Optional dependencies are not tested separately (test actual behavior)

**What NOT to Mock:**
- Pandas operations (test real pandas behavior)
- Calculation functions (test actual computation)
- Data validation logic (use real data structures)

## Test Types

**Unit Tests:**
- Scope: Individual functions and methods
- Approach: Test single responsibility with isolated inputs/outputs
- Examples from `tests/test_ranker_scores.py`:
  - `TestCoverageScore.test_full_coverage()` - Tests coverage score calculation
  - `TestWinsorize.test_outlier_capped()` - Tests outlier capping
  - `TestPercentileRank.test_basic_ranking()` - Tests ranking computation

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Test end-to-end workflows with real data flow
- Examples from `tests/test_tracker_returns.py`:
  - `TestForwardReturns.test_basic_return_calculation()` - Tests snapshot creation, storage, and return calculation
  - `TestSnapshotStore.test_save_and_load()` - Tests save/load cycle
  - Tests create snapshots, store them, then retrieve and process them

**E2E Tests:**
- Not implemented (no `tests/e2e/` directory)
- Analysis runs happen through scripts like `run_analysis.py`, `run_full_analysis.py`

## Common Patterns

**Async Testing:**
- Not applicable (no async code in project)

**Error Testing:**
- Test edge cases with missing/invalid data
- Example from `tests/test_ranker_scores.py`:
  ```python
  def test_missing_price(self):
      """Test that missing price returns NaN."""
      df = pd.DataFrame({
          'price': [np.nan],
          'volume_1d': [1000000],
      })
      liquidity = compute_liquidity_value(df)
      assert np.isnan(liquidity.iloc[0])
  ```

- Test boundary conditions
  ```python
  def test_partial_coverage(self):
      """Test that partial coverage returns correct fraction."""
      df = pd.DataFrame({
          'price': [100.0],
          'perf_3m': [10.0],
          'perf_6m': [np.nan],  # Missing
          # Only 3 of 14 features present
      })
      coverage = compute_coverage_score(df)
      expected = 3 / 14
      assert abs(coverage.iloc[0] - expected) < 0.01
  ```

**Numeric Assertions:**
- Use tolerance for float comparisons (not strict equality)
  ```python
  assert abs(result - expected) < 0.01  # Allow 0.01 tolerance
  assert abs(metrics['universe_mean_return'] - 0.0) < 0.1
  ```

**DataFrame Assertions:**
- Check row counts: `assert len(result['aggressive_growth']) > 0`
- Check column presence: `assert 'liquidity_value_1d' in snapshot.columns`
- Check row values: `assert snapshot['coverage_score'].iloc[0] > 0.5`
- Check sorting: `assert full_rank < sparse_rank` (by proxied column values)

## Test Coverage

**Requirements:** Not enforced (no coverage configuration found)

**View Coverage:**
```bash
pip install pytest-cov
pytest --cov=analysis tests/
pytest --cov=core tests/
pytest --cov=config tests/
```

## Imports in Tests

**Standard pattern:**
```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from project
from analysis.snapshot_ranker import (
    rank_snapshot,
    compute_coverage_score,
    _winsorize,
)
from analysis.snapshot_store import SnapshotStore
```

## Test Data Constants

**Inline Creation:**
- Create test DataFrames inline within test methods
- Use realistic but minimal data
- Include edge cases (NaN, missing columns, boundary values)

**Example from `tests/test_tracker_returns.py`:**
```python
self.create_test_snapshot(
    store, '2026-01-01',
    symbols=['A', 'B', 'C'],
    prices=[100.0, 200.0, 50.0],
    scores=[90, 80, 70],
)

# Later snapshot with price changes
self.create_test_snapshot(
    store, '2026-02-01',
    symbols=['A', 'B', 'C'],
    prices=[120.0, 180.0, 60.0],  # A +20%, B -10%, C +20%
    scores=[85, 75, 65],
)
```

## Running Tests from Command Line

**All tests:**
```bash
pytest tests/
```

**Single test file:**
```bash
pytest tests/test_ranker_scores.py
```

**Single test class:**
```bash
pytest tests/test_ranker_scores.py::TestCoverageScore
```

**Single test method:**
```bash
pytest tests/test_ranker_scores.py::TestCoverageScore::test_full_coverage
```

**Verbose output:**
```bash
pytest tests/ -v
pytest tests/ -vv  # More verbose
```

**Show print output:**
```bash
pytest tests/ -s
```

## Test Execution Entry Points

**Via pytest.main() in test module:**
```python
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

*Testing analysis: 2026-02-21*

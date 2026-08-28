"""
Tests for snapshot ranking functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.snapshot_ranker import (
    rank_snapshot,
    compute_coverage_score,
    compute_liquidity_value,
    _winsorize,
    _compute_percentile_rank,
    _compute_sector_ranks,
    DEFAULT_CONFIG,
)


class TestCoverageScore:
    """Tests for coverage score calculation."""
    
    def test_full_coverage(self):
        """Test that full coverage returns 1.0."""
        df = pd.DataFrame({
            'price': [100.0],
            'volume_1d': [1000000],
            'perf_3m': [10.0],
            'perf_6m': [20.0],
            'perf_1y': [30.0],
            'perf_ytd': [15.0],
            'revenue_growth_ttm': [25.0],
            'eps_growth_ttm': [30.0],
            'roe_ttm': [15.0],
            'net_margin_ttm': [10.0],
            'operating_margin_ttm': [12.0],
            'fcf_margin_ttm': [8.0],
            'debt_to_equity': [0.5],
            'vol_1m': [2.0],
        })
        
        coverage = compute_coverage_score(df)
        assert coverage.iloc[0] == 1.0
    
    def test_partial_coverage(self):
        """Test that partial coverage returns correct fraction."""
        df = pd.DataFrame({
            'price': [100.0],
            'volume_1d': [1000000],
            'perf_3m': [10.0],
            'perf_6m': [np.nan],  # Missing
            'perf_1y': [np.nan],  # Missing
            'perf_ytd': [np.nan],  # Missing
            # Only 3 of 14 features present
        })
        
        coverage = compute_coverage_score(df)
        # 3 present / 14 total
        expected = 3 / 14
        assert abs(coverage.iloc[0] - expected) < 0.01
    
    def test_zero_coverage(self):
        """Test that no coverage returns 0."""
        df = pd.DataFrame({
            'symbol': ['TEST'],
            'description': ['Test Stock'],
        })
        
        coverage = compute_coverage_score(df)
        assert coverage.iloc[0] == 0.0


class TestLiquidityValue:
    """Tests for liquidity value calculation."""
    
    def test_basic_calculation(self):
        """Test basic price * volume calculation."""
        df = pd.DataFrame({
            'price': [100.0, 50.0],
            'volume_1d': [1000000, 2000000],
        })
        
        liquidity = compute_liquidity_value(df)
        assert liquidity.iloc[0] == 100_000_000
        assert liquidity.iloc[1] == 100_000_000
    
    def test_missing_price(self):
        """Test that missing price returns NaN."""
        df = pd.DataFrame({
            'price': [np.nan],
            'volume_1d': [1000000],
        })
        
        liquidity = compute_liquidity_value(df)
        assert np.isnan(liquidity.iloc[0])
    
    def test_missing_volume(self):
        """Test that missing volume returns NaN."""
        df = pd.DataFrame({
            'price': [100.0],
            'volume_1d': [np.nan],
        })
        
        liquidity = compute_liquidity_value(df)
        assert np.isnan(liquidity.iloc[0])


class TestWinsorize:
    """Tests for winsorization."""
    
    def test_outlier_capped(self):
        """Test that outliers are capped."""
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        
        result = _winsorize(series, 1, 99)
        
        # Maximum should be capped
        assert result.max() < 100
    
    def test_normal_values_unchanged(self):
        """Test that normal values are unchanged."""
        series = pd.Series([1, 2, 3, 4, 5])
        
        result = _winsorize(series, 1, 99)
        
        # Values should be mostly unchanged
        assert result.iloc[2] == 3
    
    def test_nan_handling(self):
        """Test that NaN values are preserved."""
        series = pd.Series([1, 2, np.nan, 4, 5])
        
        result = _winsorize(series, 1, 99)
        
        assert np.isnan(result.iloc[2])


class TestPercentileRank:
    """Tests for percentile ranking."""
    
    def test_basic_ranking(self):
        """Test basic percentile ranking."""
        series = pd.Series([10, 20, 30, 40, 50])
        
        ranks = _compute_percentile_rank(series, higher_is_better=True)
        
        # Higher values should have higher ranks
        assert ranks.iloc[4] > ranks.iloc[0]
    
    def test_nan_preserved(self):
        """Test that NaN values stay NaN in ranking."""
        series = pd.Series([10, np.nan, 30])
        
        ranks = _compute_percentile_rank(series, higher_is_better=True)
        
        assert np.isnan(ranks.iloc[1])
    
    def test_lower_is_better(self):
        """Test ranking when lower values are better."""
        series = pd.Series([10, 20, 30])
        
        ranks = _compute_percentile_rank(series, higher_is_better=False)
        
        # Lower values should have higher ranks
        assert ranks.iloc[0] > ranks.iloc[2]


class TestSectorRanks:
    """Tests for sector-normalized ranking."""
    
    def test_sector_ranking(self):
        """Test that ranking is done within sector."""
        df = pd.DataFrame({
            'sector': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
        })
        
        ranks = _compute_sector_ranks(df, 'value', higher_is_better=True, min_sector_size=5)
        
        # Best in each sector should have rank 100
        assert ranks.iloc[4] == 100.0  # 50 is best in A
        assert ranks.iloc[9] == 100.0  # 500 is best in B
    
    def test_small_sector_fallback(self):
        """Test that small sectors fall back to global ranking."""
        df = pd.DataFrame({
            'sector': ['A', 'A', 'B'],  # B has only 1 stock
            'value': [10, 20, 100],
        })
        
        ranks = _compute_sector_ranks(df, 'value', higher_is_better=True, min_sector_size=5)
        
        # B should use global rank since sector too small
        # 100 is highest globally, so should have rank 100
        assert ranks.iloc[2] == 100.0
    
    def test_missing_sector(self):
        """Test that missing sector uses global rank."""
        df = pd.DataFrame({
            'sector': ['A', np.nan, 'A'],
            'value': [10, 50, 20],
        })
        
        ranks = _compute_sector_ranks(df, 'value', higher_is_better=True, min_sector_size=2)
        
        # NaN sector should get global rank
        assert not np.isnan(ranks.iloc[1])


class TestRankSnapshot:
    """Tests for full snapshot ranking."""
    
    def test_basic_ranking(self):
        """Test basic ranking produces output."""
        df = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'description': ['Stock A', 'Stock B', 'Stock C'],
            'sector': ['Finance', 'Finance', 'Finance'],
            'price': [100, 200, 300],
            'volume_1d': [1000000, 2000000, 3000000],
            'perf_3m': [10, 20, 30],
            'perf_6m': [15, 25, 35],
            'perf_1y': [20, 40, 60],
            'perf_ytd': [5, 10, 15],
            'revenue_growth_ttm': [10, 20, 30],
            'eps_growth_ttm': [5, 15, 25],
            'roe_ttm': [10, 15, 20],
            'net_margin_ttm': [5, 10, 15],
            'debt_to_equity': [0.5, 1.0, 1.5],
            'vol_1m': [2, 3, 4],
        })
        
        result = rank_snapshot(df)
        
        assert 'aggressive_growth' in result
        assert 'growth_with_guardrails' in result
        assert 'snapshot' in result
        assert len(result['aggressive_growth']) > 0
    
    def test_missingness_penalizes(self):
        """Test that stocks with missing data rank lower."""
        df = pd.DataFrame({
            'symbol': ['FULL', 'SPARSE'],
            'description': ['Full Data', 'Sparse Data'],
            'sector': ['Finance', 'Finance'],
            'price': [100, 100],
            'volume_1d': [1000000, 1000000],
            'perf_3m': [20, 20],  # Same momentum
            'perf_6m': [20, np.nan],  # SPARSE missing
            'perf_1y': [20, np.nan],  # SPARSE missing
            'perf_ytd': [20, np.nan],  # SPARSE missing
            'revenue_growth_ttm': [20, np.nan],  # SPARSE missing
            'eps_growth_ttm': [20, np.nan],  # SPARSE missing
            'roe_ttm': [20, np.nan],  # SPARSE missing
            'net_margin_ttm': [20, np.nan],  # SPARSE missing
            'operating_margin_ttm': [20, np.nan],  # SPARSE missing
            'fcf_margin_ttm': [20, np.nan],  # SPARSE missing
            'debt_to_equity': [0.5, np.nan],  # SPARSE missing
            'vol_1m': [2, np.nan],  # SPARSE missing
        })
        
        result = rank_snapshot(df)
        snapshot = result['snapshot']
        
        # FULL should have higher coverage
        full_row = snapshot[snapshot['symbol'] == 'FULL'].iloc[0]
        sparse_row = snapshot[snapshot['symbol'] == 'SPARSE'].iloc[0]
        
        assert full_row['coverage_score'] > sparse_row['coverage_score']
        
        # FULL should rank higher (lower rank number) in aggressive
        agg = result['aggressive_growth']
        if len(agg) >= 2 and 'FULL' in agg['symbol'].values and 'SPARSE' in agg['symbol'].values:
            full_rank = agg[agg['symbol'] == 'FULL']['rank'].iloc[0]
            sparse_rank = agg[agg['symbol'] == 'SPARSE']['rank'].iloc[0]
            assert full_rank < sparse_rank
    
    def test_gates_filter_stocks(self):
        """Test that stocks not meeting gates are filtered out."""
        config = DEFAULT_CONFIG.copy()
        config['gates'] = {
            'min_coverage_aggressive': 0.8,  # High threshold
            'min_liquidity_value_1d_aggressive': 100_000_000,  # High threshold
            'min_coverage_guardrails': 0.9,
            'min_liquidity_value_1d_guardrails': 500_000_000,
        }
        
        df = pd.DataFrame({
            'symbol': ['A', 'B'],
            'description': ['Stock A', 'Stock B'],
            'price': [10, 100],
            'volume_1d': [100000, 1000000],  # Low volume
            'perf_3m': [10, 20],
        })
        
        result = rank_snapshot(df, config)
        
        # Both should be filtered due to low coverage and liquidity
        assert len(result['aggressive_growth']) == 0
        assert len(result['growth_with_guardrails']) == 0
    
    def test_output_columns(self):
        """Test that output contains expected columns."""
        df = pd.DataFrame({
            'symbol': ['A'],
            'description': ['Stock A'],
            'sector': ['Finance'],
            'price': [100],
            'volume_1d': [10000000],
            'perf_3m': [10],
        })
        
        result = rank_snapshot(df)
        snapshot = result['snapshot']
        
        assert 'liquidity_value_1d' in snapshot.columns
        assert 'coverage_score' in snapshot.columns
        assert 'momentum_score' in snapshot.columns
        assert 'growth_potential_score_aggressive' in snapshot.columns
        assert 'growth_potential_score_guardrails' in snapshot.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

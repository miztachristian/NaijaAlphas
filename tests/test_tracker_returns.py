"""
Tests for snapshot tracking (forward returns).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.snapshot_store import SnapshotStore
from analysis.snapshot_tracker import evaluate_forward_returns


class TestForwardReturns:
    """Tests for forward return calculations."""
    
    @pytest.fixture
    def temp_snapshot_dir(self):
        """Create temporary directory for snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def create_test_snapshot(self, store, date, symbols, prices, scores):
        """Helper to create a test snapshot."""
        merged_df = pd.DataFrame({
            'symbol': symbols,
            'description': [f'{s} Stock' for s in symbols],
            'price': prices,
            'volume_1d': [1000000] * len(symbols),
        })
        
        ranking_df = pd.DataFrame({
            'symbol': symbols,
            'growth_potential_score_aggressive': scores,
            'rank': list(range(1, len(symbols) + 1)),
        })
        
        rankings = {
            'aggressive_growth': ranking_df.sort_values('growth_potential_score_aggressive', ascending=False),
            'growth_with_guardrails': ranking_df.copy(),
        }
        
        # Re-rank after sorting
        rankings['aggressive_growth']['rank'] = range(1, len(symbols) + 1)
        
        store.save_snapshot(
            date=date,
            merged_df=merged_df,
            rankings=rankings,
            metadata={'test': True},
        )
    
    def test_basic_return_calculation(self, temp_snapshot_dir):
        """Test basic forward return calculation."""
        store = SnapshotStore(temp_snapshot_dir)
        
        # Create prior snapshot
        self.create_test_snapshot(
            store, '2026-01-01',
            symbols=['A', 'B', 'C'],
            prices=[100.0, 200.0, 50.0],
            scores=[90, 80, 70],
        )
        
        # Create later snapshot with price changes
        self.create_test_snapshot(
            store, '2026-02-01',
            symbols=['A', 'B', 'C'],
            prices=[120.0, 180.0, 60.0],  # A +20%, B -10%, C +20%
            scores=[85, 75, 65],
        )
        
        result = evaluate_forward_returns(
            prior_date='2026-01-01',
            later_date='2026-02-01',
            top_k=2,
            list_name='aggressive',
            base_dir=temp_snapshot_dir,
        )
        
        assert result is not None
        assert result['prior_date'] == '2026-01-01'
        assert result['later_date'] == '2026-02-01'
        
        # Check returns were computed
        returns = result['returns_detail']
        assert len(returns) == 2  # top_k=2
        
        # A should have +20% return
        a_return = next((r for r in returns if r['symbol'] == 'A'), None)
        assert a_return is not None
        assert abs(a_return['return_pct'] - 20.0) < 0.01
    
    def test_return_formula(self, temp_snapshot_dir):
        """Test that return formula is correct: (later/prior - 1) * 100."""
        store = SnapshotStore(temp_snapshot_dir)
        
        # Create snapshots with known price change
        self.create_test_snapshot(
            store, '2026-01-01',
            symbols=['TEST'],
            prices=[80.0],
            scores=[100],
        )
        
        self.create_test_snapshot(
            store, '2026-02-01',
            symbols=['TEST'],
            prices=[100.0],  # 80 -> 100 = +25%
            scores=[100],
        )
        
        result = evaluate_forward_returns(
            prior_date='2026-01-01',
            later_date='2026-02-01',
            top_k=1,
            list_name='aggressive',
            base_dir=temp_snapshot_dir,
        )
        
        returns = result['returns_detail']
        expected_return = (100.0 / 80.0 - 1) * 100  # 25%
        
        assert abs(returns[0]['return_pct'] - expected_return) < 0.01
    
    def test_missing_symbol_handling(self, temp_snapshot_dir):
        """Test handling of symbols missing in later snapshot."""
        store = SnapshotStore(temp_snapshot_dir)
        
        # Prior snapshot has A, B, C
        self.create_test_snapshot(
            store, '2026-01-01',
            symbols=['A', 'B', 'C'],
            prices=[100.0, 200.0, 50.0],
            scores=[90, 80, 70],
        )
        
        # Later snapshot only has A, C (B missing)
        self.create_test_snapshot(
            store, '2026-02-01',
            symbols=['A', 'C'],
            prices=[120.0, 60.0],
            scores=[85, 65],
        )
        
        result = evaluate_forward_returns(
            prior_date='2026-01-01',
            later_date='2026-02-01',
            top_k=3,
            list_name='aggressive',
            base_dir=temp_snapshot_dir,
        )
        
        # Should report missing symbols
        metrics = result['metrics']
        assert metrics['symbols_missing'] >= 1
    
    def test_missing_snapshot_returns_none(self, temp_snapshot_dir):
        """Test that missing snapshot returns None."""
        result = evaluate_forward_returns(
            prior_date='2026-01-01',
            later_date='2026-02-01',
            top_k=10,
            list_name='aggressive',
            base_dir=temp_snapshot_dir,
        )
        
        assert result is None
    
    def test_universe_metrics(self, temp_snapshot_dir):
        """Test that universe metrics are computed correctly."""
        store = SnapshotStore(temp_snapshot_dir)
        
        # Create snapshots with multiple stocks
        self.create_test_snapshot(
            store, '2026-01-01',
            symbols=['A', 'B', 'C', 'D', 'E'],
            prices=[100.0, 100.0, 100.0, 100.0, 100.0],
            scores=[90, 80, 70, 60, 50],
        )
        
        # Various returns: A +20%, B +10%, C 0%, D -10%, E -20%
        self.create_test_snapshot(
            store, '2026-02-01',
            symbols=['A', 'B', 'C', 'D', 'E'],
            prices=[120.0, 110.0, 100.0, 90.0, 80.0],
            scores=[85, 75, 65, 55, 45],
        )
        
        result = evaluate_forward_returns(
            prior_date='2026-01-01',
            later_date='2026-02-01',
            top_k=2,
            list_name='aggressive',
            base_dir=temp_snapshot_dir,
        )
        
        metrics = result['metrics']
        
        # Universe mean should be 0% (average of +20,+10,0,-10,-20)
        assert abs(metrics['universe_mean_return'] - 0.0) < 0.1
        
        # Universe median should also be ~0%
        assert abs(metrics['universe_median_return'] - 0.0) < 0.1
        
        # Top 2 picks (A, B) have mean +15%
        assert metrics['top_picks_mean_return'] == 15.0
    
    def test_hit_rate_calculation(self, temp_snapshot_dir):
        """Test that hit rate is calculated correctly."""
        store = SnapshotStore(temp_snapshot_dir)
        
        # Create snapshots where universe median is 0%
        self.create_test_snapshot(
            store, '2026-01-01',
            symbols=['A', 'B', 'C'],
            prices=[100.0, 100.0, 100.0],
            scores=[90, 80, 70],
        )
        
        # A +10%, B -5%, C 0% -> universe median = 0%
        self.create_test_snapshot(
            store, '2026-02-01',
            symbols=['A', 'B', 'C'],
            prices=[110.0, 95.0, 100.0],
            scores=[85, 75, 65],
        )
        
        result = evaluate_forward_returns(
            prior_date='2026-01-01',
            later_date='2026-02-01',
            top_k=2,
            list_name='aggressive',
            base_dir=temp_snapshot_dir,
        )
        
        metrics = result['metrics']
        
        # Top 2 are A (+10%) and B (-5%)
        # A beats median (0%), B doesn't
        # Hit rate should be 50%
        assert metrics['hit_rate_vs_universe_median'] == 50.0


class TestSnapshotStore:
    """Tests for snapshot storage."""
    
    @pytest.fixture
    def temp_store_dir(self):
        """Create temporary directory for store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_save_and_load(self, temp_store_dir):
        """Test saving and loading a snapshot."""
        store = SnapshotStore(temp_store_dir)
        
        merged_df = pd.DataFrame({
            'symbol': ['A', 'B'],
            'price': [100, 200],
        })
        
        rankings = {
            'aggressive_growth': pd.DataFrame({
                'symbol': ['A', 'B'],
                'score': [90, 80],
            }),
            'growth_with_guardrails': pd.DataFrame({
                'symbol': ['B', 'A'],
                'score': [85, 75],
            }),
        }
        
        metadata = {'source': 'test'}
        
        store.save_snapshot('2026-01-15', merged_df, rankings, metadata)
        
        # Load it back
        loaded = store.load_snapshot('2026-01-15')
        
        assert loaded is not None
        assert len(loaded['merged']) == 2
        assert len(loaded['aggressive']) == 2
        assert loaded['metadata']['source'] == 'test'
    
    def test_list_snapshots(self, temp_store_dir):
        """Test listing available snapshots."""
        store = SnapshotStore(temp_store_dir)
        
        # Create a couple snapshots
        df = pd.DataFrame({'symbol': ['A'], 'price': [100]})
        rankings = {'aggressive_growth': df, 'growth_with_guardrails': df}
        
        store.save_snapshot('2026-01-01', df, rankings, {})
        store.save_snapshot('2026-01-15', df, rankings, {})
        
        snapshots = store.list_snapshots()
        
        assert '2026-01-01' in snapshots
        assert '2026-01-15' in snapshots
        assert len(snapshots) == 2
    
    def test_load_missing_returns_none(self, temp_store_dir):
        """Test that loading missing snapshot returns None."""
        store = SnapshotStore(temp_store_dir)
        
        result = store.load_snapshot('2026-99-99')
        
        assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

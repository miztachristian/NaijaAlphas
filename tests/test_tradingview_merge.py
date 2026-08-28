"""
Tests for TradingView snapshot merging functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tradingview_snapshot import (
    load_tradingview_exports,
    _clean_numeric,
    _standardize_symbol,
    _normalize_column_name,
)


class TestCleanNumeric:
    """Tests for _clean_numeric function."""
    
    def test_integer(self):
        assert _clean_numeric(100) == 100.0
    
    def test_float(self):
        assert _clean_numeric(3.14) == 3.14
    
    def test_string_number(self):
        assert _clean_numeric("42") == 42.0
    
    def test_string_with_percent(self):
        assert _clean_numeric("15.5%") == 15.5
    
    def test_string_with_comma(self):
        assert _clean_numeric("1,234,567") == 1234567.0
    
    def test_nan(self):
        assert np.isnan(_clean_numeric(np.nan))
    
    def test_none(self):
        assert np.isnan(_clean_numeric(None))
    
    def test_empty_string(self):
        assert np.isnan(_clean_numeric(""))
    
    def test_na_string(self):
        assert np.isnan(_clean_numeric("N/A"))
    
    def test_dash(self):
        assert np.isnan(_clean_numeric("-"))


class TestStandardizeSymbol:
    """Tests for _standardize_symbol function."""
    
    def test_simple(self):
        assert _standardize_symbol("GTCO") == "GTCO"
    
    def test_lowercase(self):
        assert _standardize_symbol("gtco") == "GTCO"
    
    def test_with_exchange_prefix(self):
        assert _standardize_symbol("NGX:GTCO") == "GTCO"
    
    def test_with_spaces(self):
        assert _standardize_symbol("  GTCO  ") == "GTCO"
    
    def test_nan(self):
        assert _standardize_symbol(np.nan) == ""


class TestNormalizeColumnName:
    """Tests for column name normalization."""
    
    def test_performance_3m(self):
        assert _normalize_column_name("Performance % 3 months") == "perf_3m"
    
    def test_pe_ratio(self):
        assert _normalize_column_name("Price to earnings ratio") == "pe_ratio"
    
    def test_debt_to_equity(self):
        assert _normalize_column_name("Debt to equity ratio, Quarterly") == "debt_to_equity"
    
    def test_analyst_rating_dropped(self):
        assert _normalize_column_name("Analyst Rating") == "_analyst_rating"


class TestLoadTradingViewExports:
    """Tests for load_tradingview_exports function."""
    
    @pytest.fixture
    def temp_csv_dir(self):
        """Create temporary directory with test CSVs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def create_csv(self, path: Path, data: dict):
        """Helper to create a CSV file."""
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
    
    def test_single_file_load(self, temp_csv_dir):
        """Test loading a single CSV file."""
        csv_path = temp_csv_dir / "test.csv"
        self.create_csv(csv_path, {
            'Symbol': ['GTCO', 'MTNN', 'DANGCEM'],
            'Description': ['Guaranty Trust', 'MTN Nigeria', 'Dangote Cement'],
            'Price': [45.0, 200.0, 500.0],
        })
        
        df, report = load_tradingview_exports([csv_path])
        
        assert len(df) == 3
        assert 'symbol' in df.columns
        assert 'price' in df.columns
        assert report.total_symbols == 3
    
    def test_merge_two_files(self, temp_csv_dir):
        """Test merging two CSV files with different columns."""
        csv1 = temp_csv_dir / "file1.csv"
        csv2 = temp_csv_dir / "file2.csv"
        
        self.create_csv(csv1, {
            'Symbol': ['GTCO', 'MTNN', 'DANGCEM'],
            'Price': [45.0, 200.0, 500.0],
        })
        
        self.create_csv(csv2, {
            'Symbol': ['GTCO', 'MTNN', 'DANGCEM'],
            'Volume 1 day': [1000000, 500000, 200000],
        })
        
        df, report = load_tradingview_exports([csv1, csv2])
        
        assert len(df) == 3
        assert 'price' in df.columns
        assert 'volume_1d' in df.columns
        assert report.columns_merged >= 1
    
    def test_row_count_stable_on_merge(self, temp_csv_dir):
        """Test that row count stays stable by Symbol after merge."""
        csv1 = temp_csv_dir / "file1.csv"
        csv2 = temp_csv_dir / "file2.csv"
        
        self.create_csv(csv1, {
            'Symbol': ['GTCO', 'MTNN', 'DANGCEM'],
            'Price': [45.0, 200.0, 500.0],
        })
        
        # Second file has extra symbol
        self.create_csv(csv2, {
            'Symbol': ['GTCO', 'MTNN', 'DANGCEM', 'ZENITH'],
            'Volume 1 day': [1000000, 500000, 200000, 300000],
        })
        
        df, report = load_tradingview_exports([csv1, csv2])
        
        # Should have 3 rows (from base file)
        assert len(df) == 3
        assert 'ZENITH' not in df['symbol'].values
    
    def test_duplicate_columns_resolved(self, temp_csv_dir):
        """Test that duplicate columns are resolved correctly."""
        csv1 = temp_csv_dir / "file1.csv"
        csv2 = temp_csv_dir / "file2.csv"
        
        self.create_csv(csv1, {
            'Symbol': ['GTCO', 'MTNN'],
            'Price': [45.0, 200.0],
        })
        
        self.create_csv(csv2, {
            'Symbol': ['GTCO', 'MTNN'],
            'Price': [45.0, 200.0],  # Same values
        })
        
        df, report = load_tradingview_exports([csv1, csv2])
        
        assert len(df) == 2
        assert report.duplicate_columns_resolved >= 1
    
    def test_analyst_rating_dropped(self, temp_csv_dir):
        """Test that Analyst Rating column is dropped."""
        csv_path = temp_csv_dir / "test.csv"
        self.create_csv(csv_path, {
            'Symbol': ['GTCO', 'MTNN'],
            'Price': [45.0, 200.0],
            'Analyst Rating': ['Buy', 'Sell'],
        })
        
        df, report = load_tradingview_exports([csv_path], drop_analyst_rating=True)
        
        # Should not have analyst rating column
        assert '_analyst_rating' not in df.columns
        assert 'analyst_rating' not in df.columns
        assert 'Analyst Rating' not in df.columns
        assert '_analyst_rating' in report.dropped_columns or 'Analyst Rating' in str(report.dropped_columns)
    
    def test_currency_columns_dropped(self, temp_csv_dir):
        """Test that currency helper columns are dropped."""
        csv_path = temp_csv_dir / "test.csv"
        self.create_csv(csv_path, {
            'Symbol': ['GTCO', 'MTNN'],
            'Price': [45.0, 200.0],
            'Price - Currency': ['NGN', 'NGN'],
        })
        
        df, report = load_tradingview_exports([csv_path], drop_currency_columns=True)
        
        # Should not have currency column
        currency_cols = [c for c in df.columns if 'currency' in c.lower()]
        assert len(currency_cols) == 0
    
    def test_no_files_raises_error(self):
        """Test that empty file list raises error."""
        with pytest.raises(ValueError, match="No CSV paths provided"):
            load_tradingview_exports([])
    
    def test_missing_file_raises_error(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_tradingview_exports([Path("/nonexistent/file.csv")])
    
    def test_numeric_conversion(self, temp_csv_dir):
        """Test that numeric columns are converted correctly."""
        csv_path = temp_csv_dir / "test.csv"
        self.create_csv(csv_path, {
            'Symbol': ['GTCO', 'MTNN'],
            'Price': ['45.0', '200.0'],
            'Performance % 3 months': ['15.5%', '25.0%'],
        })
        
        df, report = load_tradingview_exports([csv_path])
        
        assert df['price'].dtype == float
        assert df['perf_3m'].iloc[0] == 15.5
        assert df['perf_3m'].iloc[1] == 25.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

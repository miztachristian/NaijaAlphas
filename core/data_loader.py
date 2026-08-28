"""
NGX Data Loader Module
======================
Handles all data acquisition from TradingView snapshots and local parquet files.
Includes caching for improved performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import hashlib
import logging
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    ALL_TICKERS,
    DATA_DIR,
    CACHE_DIR,
    AnalysisConfig,
)


class DataCache:
    """Simple file-based cache for stock data"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, expiry_hours: int = 24):
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours
        self.cache_dir.mkdir(exist_ok=True)
        self.meta_file = self.cache_dir / "cache_meta.json"
        self._load_meta()
    
    def _load_meta(self):
        """Load cache metadata"""
        if self.meta_file.exists():
            with open(self.meta_file, "r") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}
    
    def _save_meta(self):
        """Save cache metadata"""
        with open(self.meta_file, "w") as f:
            json.dump(self.meta, f)
    
    def _get_cache_key(self, ticker: str) -> str:
        """Generate cache key for ticker"""
        return hashlib.md5(ticker.encode()).hexdigest()[:16]
    
    def is_valid(self, ticker: str) -> bool:
        """Check if cached data is still valid"""
        key = self._get_cache_key(ticker)
        if key not in self.meta:
            return False
        
        cached_time = datetime.fromisoformat(self.meta[key]["timestamp"])
        expiry_time = cached_time + timedelta(hours=self.expiry_hours)
        return datetime.now() < expiry_time
    
    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get cached data for ticker"""
        if not self.is_valid(ticker):
            return None
        
        key = self._get_cache_key(ticker)
        cache_file = self.cache_dir / f"{key}.parquet"
        
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                return None
        return None
    
    def set(self, ticker: str, df: pd.DataFrame):
        """Cache data for ticker"""
        key = self._get_cache_key(ticker)
        cache_file = self.cache_dir / f"{key}.parquet"
        
        df.to_parquet(cache_file)
        self.meta[key] = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "rows": len(df)
        }
        self._save_meta()
    
    def clear(self):
        """Clear all cached data"""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        self.meta = {}
        self._save_meta()


class NGXDataLoader:
    """
    Data loader for NGX (Nigerian Stock Exchange) stocks.
    
    Supports:
    - TradingView historical parquet files (OHLCV)
    - TradingView snapshot data (fundamentals)
    - Local CSV files (legacy)
    - Data caching
    """
    
    def __init__(self, use_cache: bool = True, cache_expiry_hours: int = 24):
        """
        Initialize data loader.
        
        Args:
            use_cache: Whether to use data caching
            cache_expiry_hours: Hours before cached data expires
        """
        self.memory_cache: Dict[str, pd.DataFrame] = {}
        self.use_cache = use_cache
        self._snapshot_cache: Optional[pd.DataFrame] = None
        
        if use_cache:
            self.file_cache = DataCache(expiry_hours=cache_expiry_hours)
        else:
            self.file_cache = None
    
    def load_historical_parquet(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load real OHLCV history from data/historical/<TICKER>.parquet.

        These files are produced by download_bulk_historical.py via TradingView
        WebSocket interception and are the canonical NGX price source.
        """
        path = DATA_DIR / "historical" / f"{ticker}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            # Stored with DatetimeIndex named 'Date' — flatten to a column to
            # match what the rest of the system expects.
            if df.index.name == "Date" or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            return df if not df.empty else None
        except Exception as e:
            print(f"  Warning: Could not load historical parquet for {ticker}: {e}")
            return None

    def load_csv(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from local CSV file.

        Args:
            ticker: NGX ticker symbol

        Returns:
            DataFrame with OHLCV data or None
        """
        csv_path = DATA_DIR / f"{ticker}.csv"

        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)
            df["Date"] = pd.to_datetime(df["Date"])

            # Standardize column names
            if "Adj Close" in df.columns:
                df = df.rename(columns={"Adj Close": "Adj_Close"})

            df = df.sort_values("Date").reset_index(drop=True)
            return df

        except Exception as e:
            print(f"  Warning: Could not load CSV for {ticker}: {e}")
            return None
    
    def get_price_data(
        self,
        ticker: str,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get price data for a ticker.

        Priority:
        1. Memory cache
        2. data/historical/<ticker>.parquet  (real OHLCV from TradingView)
        3. File cache
        4. Local CSV (legacy placeholder data — last resort)
        """
        # Check memory cache
        if not force_refresh and ticker in self.memory_cache:
            return self.memory_cache[ticker]

        # 1. Real historical parquet (primary source for NGX)
        df = self.load_historical_parquet(ticker)

        # 2. File cache (legacy)
        if df is None and not force_refresh and self.file_cache:
            cached = self.file_cache.get(ticker)
            if cached is not None:
                df = cached

        # 3. Local CSV (last resort)
        if df is None:
            df = self.load_csv(ticker)

        # Cache if successful
        if df is not None and not df.empty:
            self.memory_cache[ticker] = df
            if self.file_cache:
                self.file_cache.set(ticker, df)

        return df
    
    def get_price_series(
        self, 
        ticker: str,
        price_col: str = "Adj_Close",
        force_refresh: bool = False
    ) -> Optional[pd.Series]:
        """
        Get adjusted close price series.
        
        Args:
            ticker: NGX ticker symbol
            price_col: Price column to use (Adj_Close or Close)
            force_refresh: Force refresh from source
        
        Returns:
            Series indexed by date or None
        """
        df = self.get_price_data(ticker, force_refresh)
        
        if df is None or df.empty:
            return None
        
        df = df.sort_values("Date").set_index("Date")
        
        # Use adjusted close if available, otherwise close
        if price_col in df.columns:
            series = df[price_col].copy()
        elif "Close" in df.columns:
            series = df["Close"].copy()
        else:
            return None
        
        return series.dropna()
    
    def get_returns(
        self, 
        ticker: str,
        period: str = "daily",
        force_refresh: bool = False
    ) -> Optional[pd.Series]:
        """
        Get returns for a ticker.
        
        Args:
            ticker: NGX ticker symbol
            period: Return period (daily, weekly, monthly)
            force_refresh: Force refresh from source
        
        Returns:
            Series of returns or None
        """
        prices = self.get_price_series(ticker, force_refresh=force_refresh)
        
        if prices is None:
            return None
        
        if period == "daily":
            return prices.pct_change().dropna()
        elif period == "weekly":
            weekly = prices.resample("W").last()
            return weekly.pct_change().dropna()
        elif period == "monthly":
            monthly = prices.resample("M").last()
            return monthly.pct_change().dropna()
        else:
            return prices.pct_change().dropna()

    def _load_latest_snapshot(self) -> Optional[pd.DataFrame]:
        """
        Load the most recent TradingView snapshot (snapshot_merged.csv or
        snapshot.parquet) from data/snapshots/<date>/.

        Caches the result in memory so repeated calls are free.
        """
        if self._snapshot_cache is not None:
            return self._snapshot_cache

        snap_dir = DATA_DIR / "snapshots"
        if not snap_dir.exists():
            return None

        # Find the most recent snapshot directory
        date_dirs = sorted(snap_dir.glob("20*"), reverse=True)
        if not date_dirs:
            return None

        for ddir in date_dirs:
            # Prefer merged CSV (has more columns from TradingView screener)
            merged_csv = ddir / "snapshot_merged.csv"
            if merged_csv.exists():
                try:
                    df = pd.read_csv(merged_csv)
                    self._snapshot_cache = df
                    return df
                except Exception:
                    pass

            # Fall back to parquet
            snap_parquet = ddir / "snapshot.parquet"
            if snap_parquet.exists():
                try:
                    df = pd.read_parquet(snap_parquet)
                    self._snapshot_cache = df
                    return df
                except Exception:
                    pass

        return None

    def get_snapshot_fundamentals(self, ticker: str) -> Dict:
        """
        Get fundamental data for a ticker from the latest TradingView snapshot.

        Returns a dict with keys matching the TradingView snapshot column names
        (pe_ratio, roe_ttm, eps_growth_ttm, etc.).  Returns {} if no snapshot
        data is available for the ticker.
        """
        snap = self._load_latest_snapshot()
        if snap is None:
            return {}

        # Find the row for this ticker
        sym_col = "symbol" if "symbol" in snap.columns else "Symbol"
        if sym_col not in snap.columns:
            return {}

        row = snap[snap[sym_col].str.upper() == ticker.upper()]
        if row.empty:
            return {}

        return row.iloc[0].to_dict()

    def get_multiple_stocks(
        self, 
        tickers: List[str],
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple tickers.
        
        Args:
            tickers: List of NGX ticker symbols
            show_progress: Show progress indicator
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers):
            if show_progress:
                print(f"  Loading {ticker} ({i+1}/{total})...", end="\r")
            
            df = self.get_price_data(ticker)
            if df is not None:
                results[ticker] = df
        
        if show_progress:
            print(f"  Loaded {len(results)}/{total} stocks" + " " * 20)
        
        return results
    
    def clear_cache(self):
        """Clear all caches"""
        self.memory_cache.clear()
        self._snapshot_cache = None
        if self.file_cache:
            self.file_cache.clear()


# Convenience function
def load_ticker_data(ticker: str) -> Optional[pd.DataFrame]:
    """Quick function to load data for a single ticker"""
    loader = NGXDataLoader()
    return loader.get_price_data(ticker)


def load_all_tickers() -> Dict[str, pd.DataFrame]:
    """Load data for all known tickers"""
    loader = NGXDataLoader()
    return loader.get_multiple_stocks(ALL_TICKERS)

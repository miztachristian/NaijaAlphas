"""
TradingView Snapshot Loader
===========================
Load and merge multiple TradingView screener CSV exports into a single snapshot.
Handles duplicate columns, missing data, and standardizes column names.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class MergeReport:
    """Report of merge operations and conflicts."""
    source_files: List[str] = field(default_factory=list)
    total_symbols: int = 0
    columns_merged: int = 0
    duplicate_columns_resolved: int = 0
    conflicts: List[Dict] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)
    symbols_missing_in_later_files: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'source_files': self.source_files,
            'total_symbols': self.total_symbols,
            'columns_merged': self.columns_merged,
            'duplicate_columns_resolved': self.duplicate_columns_resolved,
            'conflicts_count': len(self.conflicts),
            'dropped_columns': self.dropped_columns,
            'symbols_missing_in_later_files': self.symbols_missing_in_later_files,
        }


# Column name mapping from TradingView to internal names
COLUMN_MAP = {
    # Identifiers
    'Symbol': 'symbol',
    'Ticker': 'symbol',
    'Description': 'description',
    'Sector': 'sector',
    
    # Price & Volume
    'Price': 'price',
    'Close': 'price',
    'Last': 'price',
    'Volume 1 day': 'volume_1d',
    'Volume': 'volume_1d',
    'Average Volume (10 day)': 'avg_volume_10d',
    'Average Volume (30 day)': 'avg_volume_30d',
    'Average Volume (60 day)': 'avg_volume_60d',
    'Average Volume (90 day)': 'avg_volume_90d',
    'Relative Volume': 'relative_volume',
    'Relative Volume 1 day': 'relative_volume_1d',

    # Performance (momentum)
    'Performance % 1 week': 'perf_1w',
    'Performance % 1 month': 'perf_1m',
    'Performance % 3 months': 'perf_3m',
    'Performance % 6 months': 'perf_6m',
    'Performance % 1 year': 'perf_1y',
    'Performance % Year to date': 'perf_ytd',
    'Performance % 5 years': 'perf_5y',
    'Performance % All Time': 'perf_all_time',
    'Price Change % 1 day': 'price_change_1d',
    
    # Volatility
    'Volatility 1 week': 'vol_1w',
    'Volatility 1 month': 'vol_1m',
    'Volatility Week': 'vol_1w',
    'Volatility Month': 'vol_1m',
    
    # Growth metrics
    'Revenue growth %, TTM YoY': 'revenue_growth_ttm',
    'Revenue growth rate (Quarterly YoY)': 'revenue_growth_qoq',
    'Revenue Growth, FQ-over-FQ': 'revenue_growth_fq',
    'Earnings per share diluted growth %, TTM YoY': 'eps_growth_ttm',
    'EPS diluted growth %, TTM YoY': 'eps_growth_ttm',
    'Earnings Per Share Growth Rate (YoY), TTM': 'eps_growth_ttm',
    'EPS Diluted Growth %, TTM YoY': 'eps_growth_ttm',
    'EPS growth rate (Quarterly YoY)': 'eps_growth_qoq',
    'EPS Diluted Growth Rate (YoY), TTM': 'eps_growth_ttm',
    'Net Income Growth, FQ-over-FQ': 'net_income_growth_fq',
    
    # Profitability
    'Return on equity %, Trailing 12 months': 'roe_ttm',
    'Return on Equity (TTM)': 'roe_ttm',
    'ROE, TTM': 'roe_ttm',
    'Return on assets %, Trailing 12 months': 'roa_ttm',
    'Return on Assets (TTM)': 'roa_ttm',
    'Return on invested capital %, Trailing 12 months': 'roic_ttm',
    'Net margin %, Trailing 12 months': 'net_margin_ttm',
    'Net Margin (TTM)': 'net_margin_ttm',
    'Net Margin, TTM': 'net_margin_ttm',
    'Operating margin %, Trailing 12 months': 'operating_margin_ttm',
    'Operating Margin (TTM)': 'operating_margin_ttm',
    'Gross margin %, Trailing 12 months': 'gross_margin_ttm',
    'Gross Margin (TTM)': 'gross_margin_ttm',
    'Free cash flow margin %, Trailing 12 months': 'fcf_margin_ttm',
    'Free Cash Flow Margin (TTM)': 'fcf_margin_ttm',
    'FCF Margin, TTM': 'fcf_margin_ttm',
    
    # Valuation
    'Price to earnings ratio': 'pe_ratio',
    'Price/Earnings (TTM)': 'pe_ratio',
    'P/E Ratio (TTM)': 'pe_ratio',
    'Price to book ratio': 'pb_ratio',
    'Price/Book (FY)': 'pb_ratio',
    'P/B Ratio': 'pb_ratio',
    'Price to sales ratio': 'ps_ratio',
    'Price/Sales (TTM)': 'ps_ratio',
    'Price to cash flow ratio': 'pcf_ratio',
    'Price/Cash Flow (TTM)': 'pcf_ratio',
    'Price to free cash flow ratio': 'p_fcf_ratio',
    'Price/Free Cash Flow (TTM)': 'p_fcf_ratio',
    'EV to EBITDA': 'ev_ebitda',
    'Enterprise Value/EBITDA (TTM)': 'ev_ebitda',
    'PEG Ratio': 'peg_ratio',
    'Price/Earnings to Growth': 'peg_ratio',
    
    # Debt & Leverage
    'Debt to equity ratio, Quarterly': 'debt_to_equity',
    'Debt/Equity (MRQ)': 'debt_to_equity',
    'Total Debt to Equity (MRQ)': 'debt_to_equity',
    'Debt to Equity Ratio': 'debt_to_equity',
    'Current ratio, Quarterly': 'current_ratio',
    'Current Ratio (MRQ)': 'current_ratio',
    'Quick ratio, Quarterly': 'quick_ratio',
    'Quick Ratio (MRQ)': 'quick_ratio',
    
    # Per share data
    'Earnings per share diluted, Trailing 12 months': 'eps_diluted_ttm',
    'EPS Diluted (TTM)': 'eps_diluted_ttm',
    'Earnings per share basic, Trailing 12 months': 'eps_basic_ttm',
    'Book value per share, Quarterly': 'bvps',
    'Book/sh (MRQ)': 'bvps',
    'Revenue per share, Trailing 12 months': 'revenue_per_share',
    'Free cash flow per share, Trailing 12 months': 'fcf_per_share',
    
    # Dividends
    'Dividend yield %': 'dividend_yield',
    'Dividend Yield (FY)': 'dividend_yield',
    'Dividends yield (indicated)': 'dividend_yield',
    'Dividend payout ratio %, Trailing 12 months': 'dividend_payout',
    'Payout Ratio (TTM)': 'dividend_payout',
    'Dividends per share (FY)': 'dps',
    
    # Market cap
    'Market capitalization': 'market_cap',
    'Market Cap': 'market_cap',
    
    # Technical (for reference, not scoring)
    'RSI (14)': 'rsi_14',
    'RSI14': 'rsi_14',
    'MACD Level (12, 26)': 'macd',
    'MACD Signal (12, 26)': 'macd_signal',
    'Stochastic %K (14, 3, 3)': 'stoch_k',
    'Stochastic %D (14, 3, 3)': 'stoch_d',
    'CCI (20)': 'cci_20',
    'Momentum (10)': 'momentum_10',
    'ADX (14)': 'adx_14',
    'ATR (14)': 'atr_14',
    
    # Technical indicators (additional TradingView formats)
    'Relative Strength Index (14) 1 month': 'rsi_14_1m',
    'Relative Strength Index (14) 1 day': 'rsi_14',
    'Awesome Oscillator 1 day': 'awesome_oscillator_1d',
    'Commodity Channel Index (20) 1 day': 'cci_20_1d',
    'Technical Rating 1 day': 'technical_rating_1d',
    'Moving Averages Rating 1 day': 'moving_averages_rating_1d',
    'Oscillators Rating 1 day': 'oscillators_rating_1d',
    'Candlestick Pattern 1 day': 'candlestick_pattern_1d',
    'Stochastic (14,3,3) 1 day, %K': 'stochastic_k_1d',
    'Stochastic (14,3,3) 1 day, %D': 'stochastic_d_1d',

    # Volume (additional TradingView formats)
    'Volume 1 week': 'volume_1w',
    'Average Volume 10 days': 'avg_volume_10d',
    'Average Volume 90 days': 'avg_volume_90d',
    'Relative Volume 1 day': 'relative_volume_1d',
    'Relative Volume 1 week': 'relative_volume_1w',

    # Earnings & valuation (additional)
    'Earnings yield %, Trailing 12 months': 'earnings_yield_ttm',
    'Market capitalization performance % 1 month': 'market_cap_perf_1m',
    'Market capitalization performance % 1 year': 'market_cap_perf_1y',
    'Price to cash ratio': 'p_cash_ratio',

    # Growth (additional TradingView formats)
    'Total assets growth %, Quarterly QoQ': 'total_assets_growth_qoq',
    'Net income growth %, TTM YoY': 'net_income_growth_ttm',
    'Net income growth %, Annual YoY': 'net_income_growth_annual',
    'EBITDA growth %, TTM YoY': 'ebitda_growth_ttm',
    'Gross profit growth %, TTM YoY': 'gross_profit_growth_ttm',
    'Free cash flow growth %, TTM YoY': 'fcf_growth_ttm',
    'Total assets growth %, Annual YoY': 'total_assets_growth_annual',
    'Total assets growth %, Quarterly YoY': 'total_assets_growth_quarterly',

    # Balance sheet (additional TradingView formats)
    'Debt to assets ratio, Annual': 'debt_to_assets_annual',
    'Total assets, Annual': 'total_assets_annual',
    'Net debt, Annual': 'net_debt_annual',
    'Total liabilities, Annual': 'total_liabilities_annual',
    'Cash to debt ratio, Quarterly': 'cash_to_debt_ratio',

    # Analyst Rating - to be dropped from scoring
    'Analyst Rating': '_analyst_rating',
    'Recommendation': '_analyst_rating',
    'Technical Rating': '_technical_rating',
}

# Columns to drop from scoring (keep for reference only)
COLUMNS_TO_DROP_FROM_SCORING = [
    '_analyst_rating',
    '_technical_rating',
]

# Key features for coverage calculation
KEY_FEATURES = [
    'price', 'volume_1d',
    'perf_3m', 'perf_6m', 'perf_1y', 'perf_ytd',
    'revenue_growth_ttm', 'eps_growth_ttm',
    'roe_ttm', 'roic_ttm',
    'net_margin_ttm', 'operating_margin_ttm', 'fcf_margin_ttm',
    'debt_to_equity',
    'vol_1m',
    'rsi_14_1m', 'avg_volume_90d', 'earnings_yield_ttm',
]


def _clean_numeric(value) -> Optional[float]:
    """Convert a value to float, handling % signs, commas, and text."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove % signs, commas, spaces
        cleaned = value.replace('%', '').replace(',', '').replace(' ', '').strip()
        if cleaned in ('', '-', 'N/A', 'n/a', 'NA', 'nan', 'NaN', '—'):
            return np.nan
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    return np.nan


def _standardize_symbol(symbol: str) -> str:
    """Standardize symbol format (remove exchange prefix if present)."""
    if pd.isna(symbol):
        return ''
    symbol = str(symbol).strip().upper()
    # Remove exchange prefixes like "NGX:", "NGSE:", etc.
    if ':' in symbol:
        symbol = symbol.split(':')[-1]
    return symbol


def _normalize_column_name(col: str) -> str:
    """Map TradingView column name to internal name."""
    # Direct mapping first
    if col in COLUMN_MAP:
        return COLUMN_MAP[col]
    
    # Try case-insensitive match
    col_lower = col.lower().strip()
    for tv_col, internal_col in COLUMN_MAP.items():
        if tv_col.lower() == col_lower:
            return internal_col

    # Punctuation-insensitive match. TradingView export headers have drifted to
    # include commas (e.g. 'Performance %, 1 month', 'Volume, 1 day') that the
    # COLUMN_MAP keys lack. Match on a loose form that strips %,()$ and collapses
    # whitespace so these still map to canonical names (perf_1m, volume_1d, ...)
    # instead of falling through to the generic snake_case fallback.
    def _loose(s: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[%,()$]', ' ', s)).strip().lower()

    col_loose = _loose(col)
    for tv_col, internal_col in COLUMN_MAP.items():
        if _loose(tv_col) == col_loose:
            return internal_col

    # Fallback: convert to snake_case
    # Remove special chars, replace spaces with underscores
    cleaned = re.sub(r'[%$,()]', '', col)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.lower()


def load_tradingview_exports(
    paths: List[Path],
    drop_analyst_rating: bool = True,
    drop_currency_columns: bool = True,
) -> Tuple[pd.DataFrame, MergeReport]:
    """
    Load and merge multiple TradingView CSV exports.
    
    Args:
        paths: List of CSV file paths to load
        drop_analyst_rating: Whether to drop analyst rating columns
        drop_currency_columns: Whether to drop currency helper columns
        
    Returns:
        Tuple of (merged DataFrame, MergeReport)
    """
    report = MergeReport()
    
    if not paths:
        raise ValueError("No CSV paths provided")
    
    # Convert to Path objects
    paths = [Path(p) for p in paths]
    
    # Validate all files exist
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
    
    report.source_files = [str(p) for p in paths]
    
    # Load first file as base
    base_df = pd.read_csv(paths[0])
    logger.info(f"Loaded base file: {paths[0]} ({len(base_df)} rows, {len(base_df.columns)} cols)")
    
    # Find symbol column
    symbol_col = None
    for col in ['Symbol', 'Ticker', 'symbol', 'ticker']:
        if col in base_df.columns:
            symbol_col = col
            break
    
    if symbol_col is None:
        raise ValueError(f"No Symbol/Ticker column found in {paths[0]}")
    
    # Standardize symbols
    base_df[symbol_col] = base_df[symbol_col].apply(_standardize_symbol)
    base_df = base_df[base_df[symbol_col] != ''].copy()
    base_df = base_df.set_index(symbol_col)
    
    # Remove duplicate index entries (keep first)
    base_df = base_df[~base_df.index.duplicated(keep='first')]
    
    base_symbols = set(base_df.index)
    
    # Merge additional files
    for i, path in enumerate(paths[1:], start=2):
        add_df = pd.read_csv(path)
        logger.info(f"Loading file {i}: {path} ({len(add_df)} rows, {len(add_df.columns)} cols)")
        
        # Find symbol column in this file
        add_symbol_col = None
        for col in ['Symbol', 'Ticker', 'symbol', 'ticker']:
            if col in add_df.columns:
                add_symbol_col = col
                break
        
        if add_symbol_col is None:
            logger.warning(f"No Symbol column in {path}, skipping")
            continue
        
        add_df[add_symbol_col] = add_df[add_symbol_col].apply(_standardize_symbol)
        add_df = add_df[add_df[add_symbol_col] != ''].copy()
        add_df = add_df.set_index(add_symbol_col)
        add_df = add_df[~add_df.index.duplicated(keep='first')]
        
        # Track missing symbols
        missing = base_symbols - set(add_df.index)
        if missing:
            report.symbols_missing_in_later_files[str(path)] = list(missing)[:20]  # Cap at 20
        
        # Collect new columns to add in batch
        new_cols_to_add = {}
        cols_to_update = {}
        
        # Process columns
        for col in add_df.columns:
            if col in ['Description', 'description']:
                # Skip description if already exists
                if 'Description' in base_df.columns or 'description' in base_df.columns:
                    continue
            
            if col in base_df.columns:
                # Handle duplicate column
                report.duplicate_columns_resolved += 1
                
                # Check for conflicts
                base_vals = base_df[col]
                add_vals = add_df[col].reindex(base_df.index)
                
                # Count non-null in each
                base_count = base_vals.notna().sum()
                add_count = add_vals.notna().sum()
                
                # Check for actual conflicts (both non-null but different)
                both_present = base_vals.notna() & add_vals.notna()
                if both_present.any():
                    # For numeric, check if values differ significantly
                    try:
                        base_num = pd.to_numeric(base_vals[both_present], errors='coerce')
                        add_num = pd.to_numeric(add_vals[both_present], errors='coerce')
                        diff = (base_num - add_num).abs()
                        conflicts = diff > 0.01 * base_num.abs()  # More than 1% difference
                        if conflicts.any():
                            report.conflicts.append({
                                'column': col,
                                'conflict_count': int(conflicts.sum()),
                                'kept': 'base' if base_count >= add_count else 'add',
                            })
                    except:
                        pass
                
                # Keep the one with fewer nulls
                if add_count > base_count:
                    # Schedule update: fill base with add where base is null
                    cols_to_update[col] = add_vals
            else:
                # New column - collect for batch add
                new_cols_to_add[col] = add_df[col].reindex(base_df.index)
                report.columns_merged += 1
        
        # Apply updates in batch
        for col, add_vals in cols_to_update.items():
            base_df[col] = base_df[col].fillna(add_vals)
        
        # Add new columns in batch using concat (avoids fragmentation)
        if new_cols_to_add:
            new_cols_df = pd.DataFrame(new_cols_to_add, index=base_df.index)
            base_df = pd.concat([base_df, new_cols_df], axis=1)
    
    # Reset index
    merged_df = base_df.reset_index()
    merged_df = merged_df.rename(columns={merged_df.columns[0]: 'symbol'})
    
    # Normalize column names
    col_mapping = {}
    for col in merged_df.columns:
        new_col = _normalize_column_name(col)
        col_mapping[col] = new_col
    
    merged_df = merged_df.rename(columns=col_mapping)
    
    # Handle duplicate normalized column names (keep first occurrence)
    seen = set()
    cols_to_keep = []
    for col in merged_df.columns:
        if col not in seen:
            seen.add(col)
            cols_to_keep.append(col)
    merged_df = merged_df[cols_to_keep]
    
    # Drop analyst rating if requested
    if drop_analyst_rating:
        for col in list(merged_df.columns):
            if col in COLUMNS_TO_DROP_FROM_SCORING or 'analyst' in col.lower():
                merged_df = merged_df.drop(columns=[col])
                report.dropped_columns.append(col)
    
    # Drop currency columns if requested
    if drop_currency_columns:
        for col in list(merged_df.columns):
            if 'currency' in col.lower() or col.endswith('_currency'):
                merged_df = merged_df.drop(columns=[col])
                report.dropped_columns.append(col)
    
    # Convert numeric columns
    # Identify columns that should be numeric (not symbol, description, sector)
    non_numeric_cols = {'symbol', 'description', 'sector', 'name'}
    for col in merged_df.columns:
        if col.lower() not in non_numeric_cols:
            merged_df[col] = merged_df[col].apply(_clean_numeric)
    
    report.total_symbols = len(merged_df)
    
    logger.info(f"Merged snapshot: {report.total_symbols} symbols, {len(merged_df.columns)} columns")
    
    return merged_df, report


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize column names to snake_case internal names.
    
    Returns:
        Tuple of (normalized DataFrame, mapping of original -> internal names)
    """
    mapping = {}
    new_cols = {}
    
    for col in df.columns:
        internal_name = _normalize_column_name(col)
        mapping[col] = internal_name
        new_cols[col] = internal_name
    
    return df.rename(columns=new_cols), mapping

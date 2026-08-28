"""
NGX Stock Analysis System - Main Runner
=======================================
Command-line interface for running stock analysis.

Usage:
    python run_analysis.py                  # Full analysis for current year
    python run_analysis.py --year 2026      # Specify target year
    python run_analysis.py --ticker GTCO    # Analyze single stock
    python run_analysis.py --backtest       # Run backtest

Annual-Report Commands:
    python run_analysis.py --report                         # Scrape & analyse all ~55 tickers
    python run_analysis.py --report --report-tickers DANGCEM MTNN
    python run_analysis.py --report --no-cache              # Force fresh scrape
    
TradingView Snapshot Commands:
    python run_analysis.py --tv-snapshot *.csv --date 2026-01-14 --rank --save
    python run_analysis.py --tv-eval --from 2026-01-14 --to 2026-02-14 --list aggressive --top 10
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import glob
import re
import logging

import numpy as np
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from config.settings import (
    AnalysisConfig,
    OUTPUT_DIR,
    get_all_tickers,
)
from core.data_loader import NGXDataLoader
from analysis.fundamental import FundamentalAnalyzer
from analysis.technical import TechnicalAnalyzer
from analysis.growth import GrowthAnalyzer
from analysis.backtest import BacktestEngine, momentum_selection
from analysis.seasonality import SeasonalAnalyzer, MONTH_NAMES, _parse_month
from ingest.fetch_annual_reports import ReportScraper, TICKER_TO_AFF_SLUG
from analysis.report_analyzer import ReportAnalyzer


# ─── Report Analysis ────────────────────────────────────────
def run_report_analysis(
    tickers: list[str] | None = None,
    year: int | None = None,
    no_cache: bool = False,
    save: bool = True,
) -> dict:
    """
    Scrape AfricanFinancials annual-report AI summaries and analyse.

    Args:
        tickers: Specific tickers (default: all ~55 mapped tickers)
        year:    Fiscal year to target
        no_cache: Force fresh scrape
        save:    Write results to outputs/

    Returns:
        dict  {ticker: analysis_result}
    """
    tickers = tickers or sorted(TICKER_TO_AFF_SLUG.keys())
    print(f"\n{'=' * 60}")
    print(f"  ANNUAL-REPORT SIGNAL SCAN  ({len(tickers)} tickers)")
    print(f"{'=' * 60}")
    if year:
        print(f"  Target year: {year}")

    scraper = ReportScraper(use_cache=not no_cache)
    reports = scraper.fetch_all(tickers, year=year)

    found = len(reports)
    total = len(tickers)
    missing = sorted(set(tickers) - set(reports.keys()))
    pct = found / total * 100 if total else 0

    print(f"\n  Coverage: {found}/{total} ({pct:.0f}%)")
    if missing:
        print(f"  Missing ({len(missing)}): {', '.join(missing)}")

    if found == 0:
        print("\n  No reports found.")
        return {}

    analyzer = ReportAnalyzer()
    results = analyzer.analyze_batch(reports)
    summary = analyzer.summarize_batch(results)
    print(f"\n{summary}")

    if save:
        outdir = Path("outputs")
        outdir.mkdir(exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        outpath = outdir / f"ngx_report_signals_{today}.txt"
        with open(outpath, "w", encoding="utf-8") as fh:
            fh.write(f"NGX Report Analysis  {today}\n")
            fh.write(f"Coverage: {found}/{total} ({pct:.0f}%)\n")
            if missing:
                fh.write(f"Missing: {', '.join(missing)}\n")
            fh.write(f"\n{summary}\n")
        print(f"\n  Saved -> {outpath}")

    return results


def analyze_single_stock(ticker: str, year: int = None):
    """
    Analyze a single stock with detailed output.
    
    Args:
        ticker: Stock ticker symbol
        year: Target year for analysis
    """
    year = year or AnalysisConfig.TARGET_YEAR
    
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {ticker}")
    print(f"Target Year: {year}")
    print(f"{'='*60}")
    
    loader = NGXDataLoader()
    
    # Check data availability
    prices = loader.get_price_series(ticker)
    if prices is None or len(prices) < 50:
        print(f"Error: Insufficient data for {ticker}")
        return
    
    print(f"\nData available: {len(prices)} days")
    print(f"Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Fundamental analysis
    print("\n" + "-"*40)
    print("FUNDAMENTAL ANALYSIS")
    print("-"*40)
    
    fund_analyzer = FundamentalAnalyzer(loader)
    fund_metrics = fund_analyzer.analyze(ticker)
    fund_scores = fund_analyzer.score_fundamentals(fund_metrics)
    
    print(fund_analyzer.generate_analysis_summary(ticker, fund_metrics, fund_scores))
    
    # Technical analysis
    print("\n" + "-"*40)
    print("TECHNICAL ANALYSIS")
    print("-"*40)
    
    tech_analyzer = TechnicalAnalyzer(loader)
    tech_metrics = tech_analyzer.analyze(ticker)
    tech_scores = tech_analyzer.score_technicals(tech_metrics)
    
    print(tech_analyzer.generate_analysis_summary(ticker, tech_metrics, tech_scores))
    
    # Growth score
    print("\n" + "-"*40)
    print("GROWTH POTENTIAL SCORE")
    print("-"*40)
    
    growth_analyzer = GrowthAnalyzer(target_year=year)
    result = growth_analyzer.analyze_stock(ticker)
    
    if result and result.growth:
        growth = result.growth
        print(f"\n  TOTAL GROWTH SCORE: {growth.total_score:.1f}/100")
        print(f"  Signal: {growth.signal_strength}")
        print(f"\n  Component Scores:")
        print(f"    Earnings Growth:     {growth.earnings_growth_score:.1f}")
        print(f"    Price Appreciation:  {growth.price_appreciation_score:.1f}")
        print(f"    Momentum & Trend:    {growth.momentum_score:.1f}")
        print(f"    Financial Health:    {growth.financial_health_score:.1f}")
        print(f"    Sector Outlook:      {growth.sector_score:.1f}")
        print(f"    Risk-Adjusted:       {growth.risk_adjusted_score:.1f}")
        print(f"    Seasonality:         {growth.seasonality_score:.1f}  ({growth.current_month_label})")
        if growth.seasonal_warning:
            print(f"  [!] Seasonal headwind for current month")


def run_full_analysis(year: int = None, save: bool = True):
    """
    Run full analysis on all stocks.
    
    Args:
        year: Target year for analysis
        save: Whether to save results to files
    """
    year = year or AnalysisConfig.TARGET_YEAR
    
    print(f"\n{'='*60}")
    print(f"NGX FULL STOCK ANALYSIS")
    print(f"Target Year: {year}")
    print(f"{'='*60}\n")
    
    # Initialize analyzer
    analyzer = GrowthAnalyzer(target_year=year)
    
    # Run analysis
    tickers = get_all_tickers()
    print(f"Analyzing {len(tickers)} stocks...")
    
    df = analyzer.analyze_all_stocks(tickers)
    
    if df.empty:
        print("No analysis results available.")
        return None
    
    # Generate report
    report = analyzer.generate_report(df)
    print(report)
    
    # Save results
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full results CSV
        csv_path = OUTPUT_DIR / f"ngx_analysis_{year}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        
        # Top picks CSV
        top_picks = analyzer.get_top_picks(df)
        top_csv_path = OUTPUT_DIR / f"ngx_top_picks_{year}_{timestamp}.csv"
        top_picks.to_csv(top_csv_path, index=False)
        print(f"Saved: {top_csv_path}")
        
        # Report text
        report_path = OUTPUT_DIR / f"ngx_report_{year}_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Saved: {report_path}")
    
    return df


def run_backtest(
    start_year: int = 2020,
    end_year: int = None,
    n_holdings: int = 10
):
    """
    Run backtest on growth strategy.
    
    Args:
        start_year: Backtest start year
        end_year: Backtest end year (defaults to current)
        n_holdings: Number of stocks to hold
    """
    end_year = end_year or datetime.now().year
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    print(f"\n{'='*60}")
    print("BACKTEST: Momentum Growth Strategy")
    print(f"Period: {start_year} - {end_year}")
    print(f"Holdings: {n_holdings} stocks")
    print(f"{'='*60}\n")
    
    # Initialize engine
    engine = BacktestEngine()
    
    # Get tickers
    tickers = get_all_tickers()
    
    # Run backtest
    result = engine.run_backtest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        selection_fn=lambda t, d: momentum_selection(t, d, n_holdings),
        n_holdings=n_holdings,
        strategy_name="Momentum Growth"
    )
    
    # Print report
    report = engine.generate_report(result)
    print(report)
    
    # Save results
    engine.save_results(result)
    
    return result


def print_summary_table(df):
    """Print a nice summary table of top picks"""
    top_10 = df.head(10)
    
    print(f"\n{'='*80}")
    print(f"TOP 10 GROWTH PICKS - QUICK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Ticker':<12} {'Sector':<12} {'Score':<8} {'Signal':<12} {'12M Mom':<10} {'P/E':<8}")
    print("-"*80)
    
    for _, row in top_10.iterrows():
        rank = row['rank']
        mom = f"{row['momentum_12m']:.1f}%" if not np.isnan(row['momentum_12m']) else "N/A"
        pe = f"{row['pe_ratio']:.1f}" if not np.isnan(row['pe_ratio']) else "N/A"
        
        print(f"{rank:<5} {row['ticker']:<12} {row['sector']:<12} {row['growth_score']:<8.1f} {row['signal']:<12} {mom:<10} {pe:<8}")


# ============================================================================
# TradingView Snapshot Commands
# ============================================================================

def parse_date_from_filename(filename: str) -> str:
    """Try to extract date from TradingView filename like 'Nigerian Stocks_2026-01-14 (1).csv'"""
    # Pattern: anything_YYYY-MM-DD or anything_YYYY-MM-DD (N)
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None


def run_snapshot_ranking(csv_paths: list, date: str, save: bool = True, config_path: Path = None):
    """
    Load TradingView CSVs, merge, rank, and optionally save.
    
    Args:
        csv_paths: List of CSV file paths
        date: Snapshot date (YYYY-MM-DD)
        save: Whether to save to disk
        config_path: Optional path to config YAML
    """
    from utils.tradingview_snapshot import load_tradingview_exports
    from analysis.snapshot_ranker import rank_snapshot, print_top_stocks, load_config
    from analysis.snapshot_store import save_snapshot
    
    print(f"\n{'='*70}")
    print("[TRADINGVIEW SNAPSHOT RANKING]")
    print(f"{'='*70}")
    print(f"Date: {date}")
    print(f"Files: {len(csv_paths)}")
    for p in csv_paths[:5]:
        print(f"  - {Path(p).name}")
    if len(csv_paths) > 5:
        print(f"  ... and {len(csv_paths) - 5} more")
    print()
    
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent / 'config' / 'snapshot_ranker.yaml'
    config = load_config(config_path)
    
    # Load and merge
    print("Loading and merging CSV exports...")
    merged_df, report = load_tradingview_exports([Path(p) for p in csv_paths])
    
    print(f"[OK] Merged: {report.total_symbols} symbols, {report.columns_merged} new columns")
    if report.duplicate_columns_resolved > 0:
        print(f"   Resolved {report.duplicate_columns_resolved} duplicate columns")
    if report.dropped_columns:
        print(f"   Dropped columns: {', '.join(report.dropped_columns[:5])}")
    
    # Rank
    print("\nRanking stocks...")
    rankings = rank_snapshot(merged_df, config)
    
    # Print top stocks
    print_top_stocks(rankings, top_k=10)
    
    # Save
    if save:
        print("\nSaving snapshot...")
        metadata = report.to_dict()
        metadata['config'] = config
        
        snapshot_dir = save_snapshot(
            date=date,
            merged_df=rankings['snapshot'],
            rankings=rankings,
            metadata=metadata,
        )
        print(f"[OK] Saved to: {snapshot_dir}")
    
    return rankings


def run_snapshot_evaluation(from_date: str, to_date: str, list_name: str, top_k: int):
    """
    Evaluate forward returns between two snapshots.
    
    Args:
        from_date: Earlier snapshot date
        to_date: Later snapshot date
        list_name: Which ranking list ('aggressive' or 'guardrails')
        top_k: Number of top stocks to evaluate
    """
    from analysis.snapshot_tracker import evaluate_forward_returns, save_evaluation, print_evaluation
    
    print(f"\n{'='*70}")
    print("[FORWARD RETURN EVALUATION]")
    print(f"{'='*70}")
    print(f"From: {from_date}")
    print(f"To:   {to_date}")
    print(f"List: {list_name}")
    print(f"Top K: {top_k}")
    print()
    
    result = evaluate_forward_returns(
        prior_date=from_date,
        later_date=to_date,
        top_k=top_k,
        list_name=list_name,
    )
    
    if result is None:
        print("[ERROR] Failed to evaluate. Check that both snapshots exist.")
        return None
    
    # Print results
    print_evaluation(result)
    
    # Save
    save_evaluation(result)
    
    return result


def run_seasonal_view(ticker: str, save: bool = False):
    """Show year x month seasonal table for one ticker."""
    sa = SeasonalAnalyzer()
    metrics = sa.analyze(ticker.upper())
    if metrics is None:
        print(f"[ERROR] No data available for {ticker}")
        return

    table = sa.generate_seasonal_table(ticker.upper())
    print(f"\n{'=' * 70}")
    print(f"  SEASONAL PATTERNS: {ticker.upper()}")
    print(f"  Years of data: {metrics.years_of_data}")
    print(f"{'=' * 70}\n")

    if table is not None:
        with pd.option_context("display.float_format", lambda v: f"{v:6.2f}"):
            print(table.to_string())

    print(f"\n  Current month: {MONTH_NAMES[metrics.current_month]}")
    print(f"    Score: {metrics.current_month_score:+.2f}  ({sa.label_current_month(metrics)})")
    if sa.is_warning(metrics):
        print(f"    [!] Seasonal headwind")

    print(f"\n  Best months (>= {3} years of data):")
    for m, avg in metrics.best_months:
        win = metrics.monthly_win_rates.get(m, 0) * 100
        n = metrics.monthly_sample_sizes.get(m, 0)
        print(f"    {MONTH_NAMES[m]:<10}  avg {avg*100:+6.2f}%   win {win:5.0f}%   n={n}")

    print(f"\n  Worst months:")
    for m, avg in metrics.worst_months:
        win = metrics.monthly_win_rates.get(m, 0) * 100
        n = metrics.monthly_sample_sizes.get(m, 0)
        print(f"    {MONTH_NAMES[m]:<10}  avg {avg*100:+6.2f}%   win {win:5.0f}%   n={n}")

    if save and table is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"seasonal_{ticker.upper()}_{timestamp}.csv"
        table.to_csv(path)
        print(f"\n  Saved: {path}")


def run_seasonal_month_rank(month, top_n: int = 20, save: bool = True):
    """Rank all tickers by historical avg return in the given month."""
    sa = SeasonalAnalyzer()
    month_int = _parse_month(month)

    print(f"\n{'=' * 70}")
    print(f"  MONTH RANKING: {MONTH_NAMES[month_int]}")
    print(f"{'=' * 70}\n")

    df = sa.rank_by_month(month_int, show_progress=True)
    if df.empty:
        print(f"  No tickers met the minimum sample requirement (>=3 years).")
        return

    print(f"  Top {min(top_n, len(df))} (of {len(df)} qualifying tickers):\n")
    print(f"  {'Rank':<5} {'Ticker':<12} {'Sector':<12} {'Avg %':>8} {'Median %':>10} {'Win %':>8} {'n':>4}")
    print("  " + "-" * 60)
    for _, row in df.head(top_n).iterrows():
        print(
            f"  {row['rank']:<5} {row['ticker']:<12} {row['sector']:<12} "
            f"{row['avg_return_pct']:>8.2f} {row['median_return_pct']:>10.2f} "
            f"{row['win_rate_pct']:>8.1f} {row['samples']:>4}"
        )

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"seasonal_month_{MONTH_NAMES[month_int].lower()}_{timestamp}.csv"
        df.to_csv(path, index=False)
        print(f"\n  Saved: {path}")


def run_seasonal_heatmap(save: bool = True):
    """Generate a tickers x months heatmap of average historical returns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    sa = SeasonalAnalyzer()
    print(f"\n{'=' * 70}")
    print(f"  SEASONAL HEATMAP (all tickers x calendar months)")
    print(f"{'=' * 70}\n")

    matrix = sa.build_universe_matrix(show_progress=True)
    if matrix.empty:
        print("  No qualifying tickers (need >=3 years of monthly data).")
        return

    # Sort tickers by overall mean (best on top)
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]

    fig_h = max(6, 0.22 * len(matrix))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    vmax = float(np.nanpercentile(np.abs(matrix.values), 95)) if matrix.size else 5.0
    vmax = max(vmax, 1.0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Ticker")
    ax.set_title(f"NGX Seasonal Returns Heatmap (avg % return by month, last 5 years)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg monthly return (%)")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d")
    if save:
        out_path = OUTPUT_DIR / f"seasonal_heatmap_{timestamp}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {out_path}")

        csv_path = OUTPUT_DIR / f"seasonal_matrix_{timestamp}.csv"
        matrix.to_csv(csv_path)
        print(f"  Saved: {csv_path}")
    plt.close(fig)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="NGX Stock Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                    # Full analysis for target year
  python run_analysis.py --year 2026        # Specify target year
  python run_analysis.py --ticker GTCO      # Analyze single stock
  python run_analysis.py --ticker GTCO MTNN # Analyze multiple stocks
  python run_analysis.py --backtest         # Run backtest
  python run_analysis.py --backtest --start 2018  # Backtest from 2018

Annual-Report Commands:
  python run_analysis.py --report                              # Scrape all ~55 tickers
  python run_analysis.py --report --report-tickers DANGCEM MTNN  # Specific tickers
  python run_analysis.py --report --year 2024 --no-cache        # Year + fresh scrape

TradingView Snapshot Commands:
  python run_analysis.py --tv-snapshot "data/*.csv" --date 2026-01-14 --rank --save
  python run_analysis.py --tv-eval --from-date 2026-01-14 --to-date 2026-02-14 --list aggressive --top 10
        """
    )
    
    parser.add_argument(
        "--year",
        type=int,
        default=AnalysisConfig.TARGET_YEAR,
        help="Target year for growth analysis"
    )
    
    parser.add_argument(
        "--ticker",
        nargs="+",
        help="Specific ticker(s) to analyze"
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest instead of analysis"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=2020,
        help="Backtest start year"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Backtest end year"
    )
    
    parser.add_argument(
        "--holdings",
        type=int,
        default=10,
        help="Number of holdings for backtest"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )

    # Annual-report scraper arguments
    parser.add_argument(
        "--report",
        action="store_true",
        help="Scrape & analyse AfricanFinancials annual-report summaries"
    )
    parser.add_argument(
        "--report-tickers",
        nargs="+",
        default=None,
        help="Specific tickers for --report (default: all ~55 mapped)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh scrape (skip cache) for --report"
    )
    
    # TradingView Snapshot arguments
    parser.add_argument(
        "--tv-snapshot",
        nargs="+",
        help="TradingView CSV file(s) or glob pattern(s) to load"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        help="Snapshot date (YYYY-MM-DD). If not provided, tries to parse from filename."
    )
    
    parser.add_argument(
        "--rank",
        action="store_true",
        help="Rank stocks from snapshot"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save snapshot and rankings to disk"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--tv-eval",
        action="store_true",
        help="Evaluate forward returns between two snapshots"
    )
    
    parser.add_argument(
        "--from-date",
        type=str,
        help="Earlier snapshot date for evaluation"
    )
    
    parser.add_argument(
        "--to-date",
        type=str,
        help="Later snapshot date for evaluation"
    )
    
    parser.add_argument(
        "--list",
        type=str,
        choices=["aggressive", "guardrails"],
        default="aggressive",
        help="Which ranking list to evaluate"
    )
    
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top stocks to evaluate / list"
    )

    # Seasonality arguments
    parser.add_argument(
        "--seasonality",
        type=str,
        metavar="TICKER",
        help="Show year x month seasonal table for a single ticker"
    )
    parser.add_argument(
        "--seasonal-month",
        type=str,
        metavar="MONTH",
        help="Rank all tickers by historical avg return in a given month (e.g. MAY or 5)"
    )
    parser.add_argument(
        "--seasonal-heatmap",
        action="store_true",
        help="Generate a tickers x months heatmap of average historical returns"
    )

    args = parser.parse_args()

    # Handle seasonality commands first
    if args.seasonality:
        run_seasonal_view(args.seasonality, save=args.save)
        return

    if args.seasonal_month:
        run_seasonal_month_rank(args.seasonal_month, top_n=args.top, save=not args.no_save)
        return

    if args.seasonal_heatmap:
        run_seasonal_heatmap(save=not args.no_save)
        return

    # Handle TradingView snapshot commands
    if args.tv_snapshot:
        # Expand glob patterns
        csv_paths = []
        for pattern in args.tv_snapshot:
            if '*' in pattern or '?' in pattern:
                csv_paths.extend(glob.glob(pattern))
            else:
                csv_paths.append(pattern)
        
        if not csv_paths:
            print("[ERROR] No CSV files found matching the pattern(s)")
            sys.exit(1)
        
        # Determine date
        date = args.date
        if not date:
            # Try to parse from first filename
            date = parse_date_from_filename(csv_paths[0])
            if not date:
                print("[ERROR] Could not determine date. Please provide --date YYYY-MM-DD")
                sys.exit(1)
            print(f"📅 Inferred date from filename: {date}")
        
        # Run ranking
        run_snapshot_ranking(
            csv_paths=csv_paths,
            date=date,
            save=args.save,
        )
        return
    
    if args.tv_eval:
        if not args.from_date or not args.to_date:
            print("[ERROR] Both --from-date and --to-date are required for evaluation")
            sys.exit(1)
        
        run_snapshot_evaluation(
            from_date=args.from_date,
            to_date=args.to_date,
            list_name=args.list,
            top_k=args.top,
        )
        return
    
    # Annual-report scraper command
    if args.report:
        run_report_analysis(
            tickers=[t.upper() for t in args.report_tickers] if args.report_tickers else None,
            year=args.year,
            no_cache=args.no_cache,
            save=not args.no_save,
        )
        return

    # Original commands
    if args.backtest:
        # Run backtest
        run_backtest(
            start_year=args.start,
            end_year=args.end,
            n_holdings=args.holdings
        )
    elif args.ticker:
        # Analyze specific stocks
        for ticker in args.ticker:
            analyze_single_stock(ticker.upper(), args.year)
    else:
        # Full analysis
        df = run_full_analysis(args.year, save=not args.no_save)
        if df is not None and not df.empty:
            print_summary_table(df)


if __name__ == "__main__":
    main()

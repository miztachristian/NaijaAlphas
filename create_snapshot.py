"""
Create a new TradingView snapshot
====================================
This script creates a new snapshot from TradingView CSV export files.

Usage:
    1. Export your TradingView screener data as CSV files
    2. Place them in a folder (or use existing data folder)
    3. Run this script and specify the CSV files
    
Example:
    python create_snapshot.py
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.tradingview_snapshot import load_tradingview_exports
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_snapshot_from_csvs(csv_files: list, output_date: str = None):
    """
    Create a snapshot from CSV files.
    
    Args:
        csv_files: List of paths to TradingView CSV export files
        output_date: Date string (YYYY-MM-DD). If None, uses today.
    """
    if not csv_files:
        print("❌ No CSV files provided")
        return
    
    # Convert to Path objects
    csv_paths = [Path(f) for f in csv_files]
    
    # Validate files exist
    missing = [str(p) for p in csv_paths if not p.exists()]
    if missing:
        print(f"❌ Files not found:")
        for f in missing:
            print(f"   - {f}")
        return
    
    print(f"\n📂 Loading {len(csv_paths)} CSV file(s)...")
    for i, p in enumerate(csv_paths, 1):
        print(f"   {i}. {p.name}")
    
    # Load and merge
    try:
        merged_df, report = load_tradingview_exports(csv_paths)
        
        print(f"\n✅ Successfully merged data!")
        print(f"   Total symbols: {len(merged_df)}")
        print(f"   Total columns: {len(merged_df.columns)}")
        print(f"   Columns merged: {report.columns_merged}")
        print(f"   Duplicate columns resolved: {report.duplicate_columns_resolved}")
        
        if report.conflicts:
            print(f"   ⚠️ Data conflicts found: {len(report.conflicts)}")
        
        # Determine output date
        if output_date is None:
            output_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create snapshot directory
        snapshot_dir = Path(__file__).parent / 'data' / 'snapshots' / output_date
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Add snapshot_date column
        merged_df['snapshot_date'] = output_date
        
        # Save as parquet (efficient format)
        output_path = snapshot_dir / 'snapshot.parquet'
        merged_df.to_parquet(output_path, index=False)
        
        print(f"\n💾 Snapshot saved:")
        print(f"   📁 {output_path}")
        print(f"   📊 {len(merged_df)} stocks × {len(merged_df.columns)} columns")
        
        # Also save as CSV for easy inspection
        csv_path = snapshot_dir / 'snapshot.csv'
        merged_df.to_csv(csv_path, index=False)
        print(f"   📄 {csv_path} (CSV backup)")
        
        # Save merge report
        report_path = snapshot_dir / 'merge_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Snapshot Merge Report\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Date: {output_date}\n")
            f.write(f"Total symbols: {len(merged_df)}\n")
            f.write(f"Total columns: {len(merged_df.columns)}\n\n")
            f.write(f"Source files:\n")
            for i, src in enumerate(report.source_files, 1):
                f.write(f"  {i}. {src}\n")
            f.write(f"\nColumns merged: {report.columns_merged}\n")
            f.write(f"Duplicate columns resolved: {report.duplicate_columns_resolved}\n")
            
            if report.conflicts:
                f.write(f"\nConflicts ({len(report.conflicts)}):\n")
                for c in report.conflicts[:10]:  # Limit to 10
                    f.write(f"  - {c['column']}: {c['conflict_count']} conflicts\n")
        
        print(f"   📝 {report_path} (merge report)")
        
        print(f"\n✨ Done! You can now run your analysis notebooks with this snapshot.")
        
        return merged_df
        
    except Exception as e:
        print(f"\n❌ Error creating snapshot: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Interactive mode - ask user for CSV files."""
    print("=" * 70)
    print("  📸 Nigeria Stocks - Create TradingView Snapshot")
    print("=" * 70)
    
    # Check data folder
    data_folder = Path(__file__).parent / 'data'
    if data_folder.exists():
        csv_files = list(data_folder.glob('*.csv'))
        if csv_files:
            print(f"\n📂 Found {len(csv_files)} CSV file(s) in data folder:")
            for i, f in enumerate(csv_files, 1):
                print(f"   {i}. {f.name}")
            
            use_these = input(f"\nUse these files? (y/n): ").strip().lower()
            if use_these == 'y':
                # Ask for date
                date_input = input(f"Enter snapshot date (YYYY-MM-DD) or press Enter for today: ").strip()
                snapshot_date = date_input if date_input else None
                
                create_snapshot_from_csvs(csv_files, snapshot_date)
                return
    
    # Manual file selection
    print("\n📋 Enter CSV file paths (one per line, empty line to finish):")
    print("   You can drag and drop files here")
    
    csv_paths = []
    while True:
        path_input = input(f"   File {len(csv_paths) + 1}: ").strip()
        if not path_input:
            break
        # Remove quotes if user dragged file
        path_input = path_input.strip('"').strip("'")
        csv_paths.append(path_input)
    
    if not csv_paths:
        print("\n❌ No files provided. Exiting.")
        return
    
    # Ask for date
    date_input = input(f"\nEnter snapshot date (YYYY-MM-DD) or press Enter for today: ").strip()
    snapshot_date = date_input if date_input else None
    
    create_snapshot_from_csvs(csv_paths, snapshot_date)


if __name__ == '__main__':
    main()

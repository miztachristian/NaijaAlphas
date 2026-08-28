"""
Process new Nigeria stocks data from Downloads folder
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from utils.tradingview_snapshot import load_tradingview_exports

# Snapshot to ingest. Pass the TradingView export date as argv[1] (default: today); the
# numbered tab files (NG_Stocks_<date>.csv, " (1).csv" … " (11).csv") are picked
# up automatically from the Downloads folder.
SNAPSHOT_DATE = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
DOWNLOADS = Path.home() / "Downloads"

csv_files = sorted(
    DOWNLOADS.glob(f"NG_Stocks_{SNAPSHOT_DATE}*.csv"),
    key=lambda p: p.name,
)

print("=" * 70)
print("  📸 Nigeria Stocks - Create Snapshot from New Data")
print("=" * 70)
print(f"\n📂 Processing {len(csv_files)} CSV files from Downloads...")

# Convert to Path objects and check existence
csv_paths = [Path(f) for f in csv_files]
existing = [p for p in csv_paths if p.exists()]
missing = [p for p in csv_paths if not p.exists()]

if missing:
    print(f"\n⚠️  Warning: {len(missing)} file(s) not found:")
    for p in missing:
        print(f"   - {p.name}")

if not existing:
    print("\n❌ No files found!")
    sys.exit(1)

print(f"✅ Found {len(existing)} file(s):")
for i, p in enumerate(existing, 1):
    print(f"   {i}. {p.name}")

# Load and merge
print(f"\n🔄 Merging data...")
try:
    merged_df, report = load_tradingview_exports(existing)
    
    print(f"\n✅ Successfully merged!")
    print(f"   📊 Total symbols: {len(merged_df)}")
    print(f"   📋 Total columns: {len(merged_df.columns)}")
    print(f"   ➕ Columns merged: {report.columns_merged}")
    print(f"   🔀 Duplicate columns resolved: {report.duplicate_columns_resolved}")
    
    if report.conflicts:
        print(f"   ⚠️  Data conflicts: {len(report.conflicts)}")
    
    # Drop unwanted columns
    if 'analyst_rating' in merged_df.columns:
        merged_df = merged_df.drop(columns=['analyst_rating'])
        print(f"   🗑️  Dropped: analyst_rating")
    
    # Create snapshot directory - extract date from filename
    import re
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', existing[0].name)
    snapshot_date = date_match.group(1) if date_match else datetime.now().strftime('%Y-%m-%d')
    snapshot_dir = Path(__file__).parent / 'data' / 'snapshots' / snapshot_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Add snapshot_date column
    merged_df['snapshot_date'] = snapshot_date
    
    # Save as parquet
    output_path = snapshot_dir / 'snapshot.parquet'
    merged_df.to_parquet(output_path, index=False)
    
    print(f"\n💾 Snapshot saved:")
    print(f"   📁 {output_path}")
    print(f"   📊 {len(merged_df)} stocks × {len(merged_df.columns)} columns")
    
    # Also save as CSV
    csv_path = snapshot_dir / 'snapshot.csv'
    merged_df.to_csv(csv_path, index=False)
    print(f"   📄 {csv_path} (CSV backup)")
    
    # Save merge report
    report_path = snapshot_dir / 'merge_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Snapshot Merge Report\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Date: {snapshot_date}\n")
        f.write(f"Total symbols: {len(merged_df)}\n")
        f.write(f"Total columns: {len(merged_df.columns)}\n\n")
        f.write(f"Source files:\n")
        for i, src in enumerate(report.source_files, 1):
            f.write(f"  {i}. {Path(src).name}\n")
        f.write(f"\nColumns merged: {report.columns_merged}\n")
        f.write(f"Duplicate columns resolved: {report.duplicate_columns_resolved}\n")
        
        if report.conflicts:
            f.write(f"\nConflicts ({len(report.conflicts)}):\n")
            for c in report.conflicts[:20]:
                f.write(f"  - {c['column']}: {c['conflict_count']} conflicts\n")
    
    print(f"   📝 {report_path}")
    
    # Show sample of columns
    print(f"\n📋 Sample columns:")
    sample_cols = list(merged_df.columns)[:15]
    for col in sample_cols:
        non_null = merged_df[col].notna().sum()
        print(f"   • {col:30} : {non_null:3} stocks")
    if len(merged_df.columns) > 15:
        print(f"   ... and {len(merged_df.columns) - 15} more columns")
    
    print(f"\n✨ Done! Snapshot created at: {snapshot_dir}")
    print(f"\n💡 Next steps:")
    print(f"   1. Open hidden_gems.ipynb notebook")
    print(f"   2. Run the cells to analyze this snapshot")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

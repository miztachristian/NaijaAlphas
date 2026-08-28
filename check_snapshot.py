import pandas as pd
from pathlib import Path

# Check the snapshot
snapshot_path = Path('data/snapshots/2026-03-17/snapshot.parquet')
if snapshot_path.exists():
    df = pd.read_parquet(snapshot_path)
    print(f"✅ Snapshot loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)[:20]}")
    print(f"\nData sample:\n{df.head(2)}")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values: {df.isnull().sum().sum()} total")
    # Check for required columns
    required = ['Price', 'RSI', 'Change%', 'Volume']
    found = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]
    print(f"\nRequired columns: Found {len(found)}/{len(required)}")
    if missing:
        print(f"Missing: {missing}")
else:
    print(f"❌ Snapshot not found at {snapshot_path}")

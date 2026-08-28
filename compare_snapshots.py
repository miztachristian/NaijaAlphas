import pandas as pd
from pathlib import Path

# Compare snapshots
snapshots = sorted(Path('data/snapshots').glob('*/snapshot.parquet'))
print(f"Available snapshots: {len(snapshots)}")
print(f"Latest: {snapshots[-1].parent.name}")
print(f"Previous: {snapshots[-2].parent.name if len(snapshots) > 1 else 'N/A'}")

if len(snapshots) > 1:
    df_new = pd.read_parquet(str(snapshots[-1]))
    df_old = pd.read_parquet(str(snapshots[-2]))
    
    rsi_cols_new = [c for c in df_new.columns if 'rsi' in c.lower()]
    rsi_cols_old = [c for c in df_old.columns if 'rsi' in c.lower()]
    
    print(f"\nNew snapshot RSI columns: {rsi_cols_new}")
    print(f"Old snapshot RSI columns: {rsi_cols_old}")
    
    # Check what get_col would find
    print(f"\nColumn search patterns:")
    for pattern in ['rsi', 'relative', 'strength', 'rsi_1d']:
        found_new = [c for c in df_new.columns if pattern.lower() in c.lower()]
        found_old = [c for c in df_old.columns if pattern.lower() in c.lower()]
        print(f"  '{pattern}': new={found_new}, old={found_old}")

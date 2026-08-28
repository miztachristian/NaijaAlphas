"""
Snapshot Store
==============
Store and load snapshots in a reproducible folder structure.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SnapshotStore:
    """Store and retrieve snapshot data."""
    
    def __init__(self, base_dir: Path = None):
        """
        Initialize snapshot store.
        
        Args:
            base_dir: Base directory for snapshots (default: data/snapshots)
        """
        self.base_dir = Path(base_dir) if base_dir else Path('data/snapshots')
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_snapshot_dir(self, date: str) -> Path:
        """Get directory for a specific snapshot date."""
        return self.base_dir / date
    
    def save_snapshot(
        self,
        date: str,
        merged_df: pd.DataFrame,
        rankings: Dict[str, pd.DataFrame],
        metadata: dict,
    ) -> Path:
        """
        Save a snapshot to disk.
        
        Args:
            date: Snapshot date (YYYY-MM-DD format)
            merged_df: Merged snapshot DataFrame
            rankings: Dict with 'aggressive_growth' and 'growth_with_guardrails' DataFrames
            metadata: Metadata dict (source files, conflicts, etc.)
            
        Returns:
            Path to snapshot directory
        """
        snapshot_dir = self._get_snapshot_dir(date)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save merged snapshot
        merged_path = snapshot_dir / 'snapshot_merged.csv'
        merged_df.to_csv(merged_path, index=False)
        logger.info(f"Saved merged snapshot: {merged_path}")
        
        # Save rankings
        if 'aggressive_growth' in rankings:
            agg_path = snapshot_dir / 'ranking_aggressive.csv'
            rankings['aggressive_growth'].to_csv(agg_path, index=False)
            logger.info(f"Saved aggressive ranking: {agg_path}")
        
        if 'growth_with_guardrails' in rankings:
            guard_path = snapshot_dir / 'ranking_guardrails.csv'
            rankings['growth_with_guardrails'].to_csv(guard_path, index=False)
            logger.info(f"Saved guardrails ranking: {guard_path}")
        
        # Save metadata
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['snapshot_date'] = date
        
        meta_path = snapshot_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {meta_path}")
        
        return snapshot_dir
    
    def load_snapshot(self, date: str) -> Optional[Dict]:
        """
        Load a snapshot from disk.
        
        Args:
            date: Snapshot date (YYYY-MM-DD format)
            
        Returns:
            Dict with 'merged', 'aggressive', 'guardrails', 'metadata' or None if not found
        """
        snapshot_dir = self._get_snapshot_dir(date)
        
        if not snapshot_dir.exists():
            logger.warning(f"Snapshot not found for date: {date}")
            return None
        
        result = {}
        
        # Load merged snapshot (new format: snapshot_merged.csv, old format: snapshot.csv / snapshot.parquet)
        merged_path = snapshot_dir / 'snapshot_merged.csv'
        if merged_path.exists():
            result['merged'] = pd.read_csv(merged_path)
        else:
            # Fallback: old format snapshot files
            parquet_path = snapshot_dir / 'snapshot.parquet'
            csv_path = snapshot_dir / 'snapshot.csv'
            if parquet_path.exists():
                result['merged'] = pd.read_parquet(parquet_path)
                logger.info(f"Loaded old-format parquet snapshot for {date}")
            elif csv_path.exists():
                result['merged'] = pd.read_csv(csv_path)
                logger.info(f"Loaded old-format CSV snapshot for {date}")
        
        # Load rankings
        agg_path = snapshot_dir / 'ranking_aggressive.csv'
        if agg_path.exists():
            result['aggressive'] = pd.read_csv(agg_path)
        
        guard_path = snapshot_dir / 'ranking_guardrails.csv'
        if guard_path.exists():
            result['guardrails'] = pd.read_csv(guard_path)
        
        # Load metadata
        meta_path = snapshot_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                result['metadata'] = json.load(f)
        
        logger.info(f"Loaded snapshot for {date}")
        return result if result else None
    
    def list_snapshots(self) -> list:
        """List all available snapshot dates."""
        if not self.base_dir.exists():
            return []
        
        dates = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                # New format or old format
                has_data = (
                    (item / 'snapshot_merged.csv').exists() or
                    (item / 'snapshot.parquet').exists() or
                    (item / 'snapshot.csv').exists()
                )
                if has_data:
                    dates.append(item.name)
        
        return sorted(dates)
    
    def get_latest_snapshot_date(self) -> Optional[str]:
        """Get the most recent snapshot date."""
        snapshots = self.list_snapshots()
        return snapshots[-1] if snapshots else None
    
    def delete_snapshot(self, date: str) -> bool:
        """Delete a snapshot."""
        snapshot_dir = self._get_snapshot_dir(date)
        
        if not snapshot_dir.exists():
            return False
        
        import shutil
        shutil.rmtree(snapshot_dir)
        logger.info(f"Deleted snapshot for {date}")
        return True


def save_snapshot(
    date: str,
    merged_df: pd.DataFrame,
    rankings: Dict[str, pd.DataFrame],
    metadata: dict,
    base_dir: Path = None,
) -> Path:
    """
    Convenience function to save a snapshot.
    
    Args:
        date: Snapshot date (YYYY-MM-DD format)
        merged_df: Merged snapshot DataFrame
        rankings: Dict with 'aggressive_growth' and 'growth_with_guardrails' DataFrames
        metadata: Metadata dict
        base_dir: Base directory for snapshots
        
    Returns:
        Path to snapshot directory
    """
    store = SnapshotStore(base_dir)
    return store.save_snapshot(date, merged_df, rankings, metadata)


def load_snapshot(date: str, base_dir: Path = None) -> Optional[Dict]:
    """
    Convenience function to load a snapshot.
    
    Args:
        date: Snapshot date (YYYY-MM-DD format)
        base_dir: Base directory for snapshots
        
    Returns:
        Dict with snapshot data or None
    """
    store = SnapshotStore(base_dir)
    return store.load_snapshot(date)

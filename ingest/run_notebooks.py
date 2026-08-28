"""
run_notebooks.py — execute the curated daily notebooks in place.

Each notebook is run headless in its own subprocess via
`python -m nbconvert --to notebook --execute --inplace`, with cwd set to the
notebooks/ directory (the notebooks resolve the project root as Path.cwd().parent
/ via nb_helpers, so this is the cwd they expect). One notebook failing or timing
out is recorded and skipped; the rest still run.

Usage:
    python -m ingest.run_notebooks                 # all curated notebooks
    python -m ingest.run_notebooks --only gamble_punt.ipynb
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"

# Curated daily notebooks, in run order. Bot-feed producers first
# (analysis_notebook -> momentum picks, gamble_punt -> punt cards), then the
# read-only notebooks. personal-portfolio notebooks are intentionally excluded.
DAILY_NOTEBOOKS = [
    "analysis_notebook.ipynb",
    "gamble_punt.ipynb",
    "decision_system.ipynb",
    "bluechip_quality.ipynb",
    "hidden_gems.ipynb",
    "momentum_ranking.ipynb",
    "seasonality_analysis.ipynb",
    "portfolio_tracker.ipynb",
    "stop_loss_tracker.ipynb",
]

NOTEBOOK_TIMEOUT = 600  # seconds, per notebook


@dataclass
class NotebookResult:
    name: str
    status: str          # ok | failed | missing
    message: str = ""


def execute_notebook(name: str, timeout: int = NOTEBOOK_TIMEOUT) -> NotebookResult:
    """Execute one notebook in place. Never raises — returns a NotebookResult."""
    path = NOTEBOOK_DIR / name
    if not path.exists():
        return NotebookResult(name, "missing", "file not found")

    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        name,
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(
            cmd, cwd=str(NOTEBOOK_DIR), env=env,
            capture_output=True, text=True, timeout=timeout + 60,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return NotebookResult(name, "failed", f"timed out after {timeout}s")

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return NotebookResult(name, "failed", " | ".join(tail) or f"exit {proc.returncode}")
    return NotebookResult(name, "ok", "executed")


def run_notebooks(only: list[str] | None = None) -> list[NotebookResult]:
    """Execute the curated set (or a subset) in order."""
    names = only if only else DAILY_NOTEBOOKS
    return [execute_notebook(n) for n in names]


def main() -> None:
    ap = argparse.ArgumentParser(description="Execute curated daily notebooks in place.")
    ap.add_argument("--only", nargs="*", metavar="NOTEBOOK.ipynb",
                    help="subset of notebook filenames to run")
    args = ap.parse_args()

    results = run_notebooks(args.only)
    for r in results:
        print(f"  [{r.status:7s}] {r.name:28s} {r.message}")
    failed = [r.name for r in results if r.status == "failed"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""One-shot helper: append the NGX Disclosure Intelligence section to the
four analysis notebooks. Idempotent — re-running replaces the section in
place rather than duplicating it.

Mirrors notebooks/_append_quality_intelligence.py.
"""
from pathlib import Path
import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent

SECTION_MARK = "# === NGX DISCLOSURE INTELLIGENCE LAYER ==="
MARKDOWN_MARK = "NGX Disclosure Intelligence"

NOTEBOOKS = (
    "analysis_notebook.ipynb",
    "bluechip_quality.ipynb",
    "hidden_gems.ipynb",
    "portfolio_tracker.ipynb",
)

INTELLIGENCE_MARKDOWN = """\
## 🏛️ NGX Disclosure Intelligence

Live signals from the NGX corporate-disclosure pipeline (`ingest/fetch_disclosures.py`),
cross-referenced against the current top picks:

| Signal | Tier | Meaning |
|---|---|---|
| `catalyst` | A | Board-meeting notice or fresh results — a near-term catalyst |
| `late_filer` | A | Delay-in-filing notice or overdue results — a governance risk |
| `dividend` | A | Dividend declared in the last ~120 days |
| `insider_90d` | A | Director-dealings filings in the last 90 days |
| `forecast` | B | The issuer's own forward PAT guidance |
| `ngx_roe` / `ngx_rev_growth` | C | Official NGX financial-statement figures |
| `score_impact` | — | Capped ±5 adjustment folded into `growth_score` |

Refresh the caches with `ingest/fetch_disclosures.py`, `ingest/parse_forecasts.py`
and `ingest/parse_statements.py`.
"""

INTELLIGENCE_CODE = '''\
# === NGX DISCLOSURE INTELLIGENCE LAYER ===
from notebooks.nb_helpers import get_batch_disclosure_insights

# Resolve the tickers to inspect — prefer actual holdings (portfolio
# notebook), else the snapshot/ranking DataFrame (notebooks bind it
# under different names).
if 'portfolio_df' in dir() and hasattr(portfolio_df, 'columns'):
    _base = portfolio_df
elif 'df' in dir():
    _base = df
elif 'rankings' in dir() and isinstance(rankings, dict) and 'snapshot' in rankings:
    _base = rankings['snapshot']
elif 'merged_df' in dir():
    _base = merged_df
elif 'snapshot' in dir():
    _base = snapshot
else:
    raise RuntimeError("Could not find a DataFrame (portfolio_df / df / rankings / merged_df / snapshot).")

_sym = next((c for c in ('symbol', 'ticker', 'Symbol', 'Ticker')
             if c in _base.columns), _base.columns[0])
_tickers = _base[_sym].dropna().astype(str).str.upper().head(25).tolist()

disc = get_batch_disclosure_insights(_tickers)

print(f"\\n{'='*100}")
print(f"🏛️  NGX DISCLOSURE INTELLIGENCE — top {len(disc)} picks")
print(f"{'='*100}")

# 1. Red flags first — late / overdue filers (governance risk + score penalty)
_late = disc[disc['late_filer'] != '—']
print(f"\\n🔴 LATE / OVERDUE FILERS — {len(_late)}")
if len(_late):
    print(_late[['symbol', 'late_filer', 'score_impact']].to_string(index=False))
else:
    print("  none among the top picks")

# 2. Near-term earnings catalysts (board meeting / fresh results)
_cat = disc[disc['catalyst'] != '—']
print(f"\\n⚡ EARNINGS CATALYSTS — {len(_cat)}")
if len(_cat):
    print(_cat[['symbol', 'catalyst']].to_string(index=False))
else:
    print("  none among the top picks")

# 3. Issuer forward guidance (Tier B earnings forecasts)
_fc = disc[disc['forecast'] != '—']
print(f"\\n🔮 FORWARD GUIDANCE — {len(_fc)}")
if len(_fc):
    print(_fc[['symbol', 'forecast']].to_string(index=False))
else:
    print("  no parsed forecasts among the top picks")

# 4. Full disclosure table
print(f"\\n📋 FULL DISCLOSURE TABLE")
_cols = ['symbol', 'catalyst', 'insider_90d', 'dividend', 'late_filer',
         'forecast', 'ngx_roe', 'ngx_rev_growth', 'score_impact']
print(disc[_cols].to_string(index=False))
'''


def append_or_replace(nb_path: Path) -> None:
    if not nb_path.exists():
        print(f"Skipped (not found): {nb_path.name}")
        return

    nb = nbf.read(nb_path, as_version=4)

    # Drop any existing disclosure cells (idempotent re-run).
    keep = []
    for cell in nb.cells:
        src = cell.source
        if SECTION_MARK in src:
            continue
        if cell.cell_type == "markdown" and MARKDOWN_MARK in src:
            continue
        keep.append(cell)
    nb.cells = keep

    nb.cells.append(nbf.v4.new_markdown_cell(INTELLIGENCE_MARKDOWN))
    nb.cells.append(nbf.v4.new_code_cell(INTELLIGENCE_CODE))

    nbf.write(nb, nb_path)
    print(f"Updated: {nb_path.name}  (total cells: {len(nb.cells)})")


if __name__ == "__main__":
    for name in NOTEBOOKS:
        append_or_replace(NB_DIR / name)

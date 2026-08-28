"""One-shot helper: append the v2.3 Quality Intelligence section to both
notebooks. Idempotent — re-running replaces the section in place rather than
duplicating it.
"""
from pathlib import Path
import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent

SECTION_MARK = "# === v2.3 QUALITY INTELLIGENCE LAYER ==="

INTELLIGENCE_MARKDOWN = """\
## 🧠 Quality Intelligence Layer (v2.3)

Five new scores from `analysis/quality_scores.py`, computed from the expanded TV financials:

| Score | What it measures | Range |
|---|---|---|
| `earnings_quality_score` | Net income backed by operating cash flow (anti-accrual) | 0–100 |
| `balance_sheet_score`    | Current ratio + cash/debt + leverage + LT debt/assets | 0–100 |
| `consistency_score`      | 5y revenue/EPS CAGR + dividend streak + direction match | 0–100 |
| `technical_signal`       | TV's aggregated MA + oscillator rating + ADX trend | label |
| `float_risk_flag`        | Sub-15% free float / concentrated holder warnings | string |

The four blocks below cross these scores to surface the highest-conviction setups
on the current snapshot.
"""

INTELLIGENCE_CODE = '''\
# === v2.3 QUALITY INTELLIGENCE LAYER ===
from analysis.quality_scores import enrich as _enrich_q

# Resolve the source snapshot — different notebooks bind it under different names.
if 'df' in dir():
    _base = df
elif 'rankings' in dir() and isinstance(rankings, dict) and 'snapshot' in rankings:
    _base = rankings['snapshot']
elif 'merged_df' in dir():
    _base = merged_df
elif 'snapshot' in dir():
    _base = snapshot
else:
    raise RuntimeError("Could not find a snapshot DataFrame (df / rankings / merged_df / snapshot).")

df_q = _enrich_q(_base)

print(f"\\n{'='*100}")
print(f"🧠 QUALITY INTELLIGENCE LAYER")
print(f"{'='*100}")

# 1. Compounders with Fortress/Healthy balance sheets + high earnings quality
elite = df_q[
    (df_q['consistency_tier'] == '⭐ Compounder')
    & (df_q['balance_sheet_tier'].isin(['🟢 Fortress', '🟢 Healthy']))
    & (df_q['earnings_quality_score'].fillna(0) >= 60)
]
print(f"\\n⭐ COMPOUNDERS with Fortress/Healthy balance sheets and earnings quality ≥60 — {len(elite)}")
if len(elite):
    cols = ['symbol', 'sector', 'consistency_score', 'balance_sheet_score',
            'earnings_quality_score', 'technical_signal', 'technical_trend',
            'float_risk_flag']
    avail = [c for c in cols if c in elite.columns]
    out = elite[avail].sort_values('consistency_score', ascending=False).head(20).copy()
    for c in ['consistency_score', 'balance_sheet_score', 'earnings_quality_score']:
        if c in out.columns:
            out[c] = out[c].round(0).astype('Int64')
    print(out.to_string(index=False))

# 2. Earnings-quality red flags — paper earnings, not cash
bad_eq = df_q[df_q['earnings_quality_score'].fillna(100) < 35]
print(f"\\n🔴 LOW EARNINGS QUALITY (NI not backed by cash) — {len(bad_eq)} stocks")
if len(bad_eq):
    cols = ['symbol', 'sector', 'earnings_quality_score', 'net_margin_ttm',
            'fcf_margin_ttm', 'perf_1y']
    avail = [c for c in cols if c in bad_eq.columns]
    out = bad_eq[avail].sort_values('earnings_quality_score').head(15).copy()
    for c in ['earnings_quality_score', 'net_margin_ttm', 'fcf_margin_ttm', 'perf_1y']:
        if c in out.columns:
            out[c] = out[c].round(1)
    print(out.to_string(index=False))

# 3. Bullish technical confluence — Buy+ AND ADX trending
bullish = df_q[df_q['technical_is_bullish'] == True]
print(f"\\n📈 BULLISH TECHNICAL CONFLUENCE (Buy+ on trending tape) — {len(bullish)}")
if len(bullish):
    cols = ['symbol', 'sector', 'technical_signal', 'technical_score',
            'technical_trend', 'rsi_14', 'perf_1m']
    avail = [c for c in cols if c in bullish.columns]
    out = bullish[avail].sort_values('technical_score', ascending=False).head(20).copy()
    for c in ['technical_score', 'rsi_14', 'perf_1m']:
        if c in out.columns:
            out[c] = out[c].round(1)
    print(out.to_string(index=False))

# 4. Thin-float watchlist — small moves get amplified
thin = df_q[df_q['float_risk_flag'].astype(str) != '']
print(f"\\n🟡 THIN-FLOAT WATCH — {len(thin)} stocks")
if len(thin):
    cols = ['symbol', 'sector', 'float_risk_flag', 'liquidity_value_1d', 'perf_1m']
    avail = [c for c in cols if c in thin.columns]
    out = thin[avail].copy()
    if 'liquidity_value_1d' in out.columns:
        out['liquidity_value_1d'] = (out['liquidity_value_1d'].fillna(0) / 1e6).round(1).astype(str) + 'M'
    if 'perf_1m' in out.columns:
        out['perf_1m'] = out['perf_1m'].round(1)
    print(out.head(15).to_string(index=False))
'''


def append_or_replace(nb_path: Path) -> None:
    nb = nbf.read(nb_path, as_version=4)

    # Drop any existing v2.3 cells (idempotent re-run).
    keep = []
    for cell in nb.cells:
        src = cell.source
        if SECTION_MARK in src:
            continue
        if cell.cell_type == "markdown" and "Quality Intelligence Layer (v2.3)" in src:
            continue
        keep.append(cell)
    nb.cells = keep

    nb.cells.append(nbf.v4.new_markdown_cell(INTELLIGENCE_MARKDOWN))
    nb.cells.append(nbf.v4.new_code_cell(INTELLIGENCE_CODE))

    nbf.write(nb, nb_path)
    print(f"Updated: {nb_path.name}  (total cells: {len(nb.cells)})")


if __name__ == "__main__":
    for name in ("hidden_gems.ipynb", "bluechip_quality.ipynb", "analysis_notebook.ipynb"):
        append_or_replace(NB_DIR / name)

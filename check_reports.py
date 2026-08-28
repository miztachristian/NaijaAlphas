import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebooks.nb_helpers import load_report_cache, get_report_insights

cache = load_report_cache()
for t in ['AFRIPRUD', 'MECURE', 'SKYAVN']:
    ins = get_report_insights(t, cache)
    tone = ins["tone"]
    score = ins["tone_score"]
    div = ins["dividend_display"]
    strat = ins["strategic_signals"]
    has = ins["has_report"]
    gov = ins["governance_flags"]
    gm = ins["growth_metrics"]
    print(f"\n{'='*60}")
    print(f"  {t} Annual Report")
    print(f"{'='*60}")
    print(f"  Has Report:  {has}")
    print(f"  Tone:        {tone} (score: {score:+.2f})")
    print(f"  Dividend:    {div}")
    print(f"  Strategic:   {strat}")
    print(f"  Governance:  {gov}")
    print(f"  Growth:      {gm}")

"""Unit + regression tests for the Core Concentrate strategy."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import CoreStrategyConfig, DATA_DIR  # noqa: E402
from analysis.core_strategy import compute_core_scores, eligible_ranked  # noqa: E402
from decision_system.concentrate import size_core  # noqa: E402
from decision_system.consolidate import consolidate  # noqa: E402


def _stock(symbol, **over):
    """A strong, fully-eligible base stock; override fields per test."""
    base = dict(
        symbol=symbol, sector="Finance", price=100.0, volume_1d=1_000_000,  # ₦100M turnover
        roe_ttm=40.0, roic_ttm=30.0, roa_ttm=15.0,
        net_margin_ttm=25.0, operating_margin_ttm=25.0, gross_margin_ttm=50.0, fcf_margin_ttm=15.0,
        eps_growth_ttm=50.0, revenue_growth_ttm=20.0,
        debt_to_equity=0.3, current_ratio=2.0,
        net_income_trailing_12_months=1e9,
        relative_volume_1d=1.2, perf_1m=5.0, is_etf=False,
    )
    base.update(over)
    return base


def _df(*stocks):
    return pd.DataFrame(list(stocks))


# ---------------- gates ----------------

def test_strong_stock_is_eligible():
    s = compute_core_scores(_df(_stock("AAA")))
    assert bool(s.iloc[0]["eligible"]) is True
    assert s.iloc[0]["gate_fails"] == ""


def test_loss_maker_fails_profit_gate():
    s = compute_core_scores(_df(_stock("LOSS", net_margin_ttm=-5.0, net_income_trailing_12_months=-1e8)))
    assert not s.iloc[0]["eligible"]
    assert "profit" in s.iloc[0]["gate_fails"]


def test_no_growth_fails_growth_gate():
    s = compute_core_scores(_df(_stock("FLAT", eps_growth_ttm=-3.0, revenue_growth_ttm=2.0)))
    assert "growth" in s.iloc[0]["gate_fails"]


def test_low_quality_fails_quality_gate():
    s = compute_core_scores(_df(_stock("LOWQ", roe_ttm=8.0, roic_ttm=5.0)))
    assert "quality" in s.iloc[0]["gate_fails"]


def test_thin_liquidity_fails():
    s = compute_core_scores(_df(_stock("THIN", volume_1d=10)))  # ₦1k turnover
    assert "liquidity" in s.iloc[0]["gate_fails"]


def test_distribution_veto():
    # down >10% on high relative volume = being sold
    s = compute_core_scores(_df(_stock("DIST", perf_1m=-15.0, relative_volume_1d=2.0)))
    assert "distribution" in s.iloc[0]["gate_fails"]


def test_falling_on_low_volume_not_vetoed():
    # same drop but quiet volume -> not distribution
    s = compute_core_scores(_df(_stock("QUIET", perf_1m=-15.0, relative_volume_1d=0.6)))
    assert "distribution" not in s.iloc[0]["gate_fails"]


def test_roic_substitutes_for_missing_roe():
    s = compute_core_scores(_df(_stock("NOROE", roe_ttm=float("nan"), roic_ttm=25.0)))
    assert bool(s.iloc[0]["eligible"]) is True  # passes quality on ROIC


def test_etf_excluded():
    s = compute_core_scores(_df(_stock("ETF", is_etf=True)))
    assert "etf" in s.iloc[0]["gate_fails"]


# ---------------- sizing ----------------

def test_score_weighted_split_sums_to_core_budget():
    scores = compute_core_scores(_df(_stock("AAA", eps_growth_ttm=80, perf_1m=8),
                                     _stock("BBB", eps_growth_ttm=20, perf_1m=1, price=50)))
    plan = size_core(scores, 1_000_000, CoreStrategyConfig)
    assert len(plan.positions) == 2
    total = sum(p.target_naira for p in plan.positions)
    assert total == pytest.approx(plan.core_budget, rel=1e-9)
    # #1 (higher score) gets the bigger slice
    assert plan.positions[0].target_naira > plan.positions[1].target_naira


def test_single_qualifier_takes_full_core():
    scores = compute_core_scores(_df(_stock("ONLY"),
                                     _stock("BAD", net_margin_ttm=-1, net_income_trailing_12_months=-1)))
    plan = size_core(scores, 1_000_000, CoreStrategyConfig)
    assert len(plan.positions) == 1
    assert plan.positions[0].target_naira == pytest.approx(plan.core_budget, rel=1e-9)
    assert any("only one name" in f for f in plan.flags)


def test_same_sector_pair_flagged():
    scores = compute_core_scores(_df(_stock("AAA", sector="Banks"),
                                     _stock("BBB", sector="Banks", price=50)))
    plan = size_core(scores, 1_000_000, CoreStrategyConfig)
    assert any("same sector" in f for f in plan.flags)


def test_forbid_same_sector_picks_different():
    class Cfg(CoreStrategyConfig):
        FORBID_SAME_SECTOR = True
    scores = compute_core_scores(_df(
        _stock("AAA", sector="Banks", eps_growth_ttm=80),
        _stock("BBB", sector="Banks", eps_growth_ttm=70, price=50),
        _stock("CCC", sector="Oil", eps_growth_ttm=60, price=70)))
    plan = size_core(scores, 1_000_000, Cfg)
    sectors = {p.sector for p in plan.positions}
    assert sectors == {"Banks", "Oil"}


def test_zero_capital_rejected():
    scores = compute_core_scores(_df(_stock("AAA")))
    with pytest.raises(ValueError):
        size_core(scores, 0, CoreStrategyConfig)


# ---------------- consolidation ----------------

def test_consolidation_buckets():
    scores = compute_core_scores(_df(
        _stock("WIN", eps_growth_ttm=80, perf_1m=8),
        _stock("OKAY"),
        _stock("BADQ", roe_ttm=5, roic_ttm=4)))
    holdings = {"WIN": {"shares": 100}, "OKAY": {"shares": 100}, "BADQ": {"shares": 100}}
    cons = consolidate(scores, holdings, core_symbols=["WIN"])
    bucket = dict(zip(cons["symbol"], cons["bucket"]))
    assert bucket["WIN"] == "CORE"
    assert bucket["OKAY"] == "HOLD-ELIGIBLE"
    assert bucket["BADQ"] == "CONSIDER-TRIM"


# ---------------- regression on the real snapshot ----------------

@pytest.mark.skipif(not (DATA_DIR / "snapshots/2026-06-23/snapshot.parquet").exists(),
                    reason="2026-06-23 snapshot not present")
def test_regression_top2_is_wapco_mtnn():
    df = pd.read_parquet(DATA_DIR / "snapshots/2026-06-23/snapshot.parquet")
    elig = eligible_ranked(compute_core_scores(df))
    assert list(elig["symbol"].head(2)) == ["WAPCO", "MTNN"]

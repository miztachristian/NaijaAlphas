"""
Tests for the decision_system package.

Pure-logic, no network: the conviction engine, confidence shrinkage, macro
regime classification, portfolio construction and calibration helpers — all
driven by synthetic inputs. Directly covers the concern mitigations from the
plan (fillna(0) bias, weight validation, graceful degradation, untested
portfolio sizing).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import ConvictionWeights, ConvictionConfig, PortfolioConfig
from decision_system.models import SignalSet, ConvictionScore, Decision, SIGNAL_NAMES
from decision_system.confidence import compute_confidence, confidence_factor
from decision_system.conviction import ConvictionEngine
from decision_system.macro_regime import MacroRegime
from decision_system.portfolio import PortfolioConstructor
from decision_system.calibration import _spearman, evaluate_pair


NAN = float("nan")


def _signal_set(ticker="TEST", **scores) -> SignalSet:
    """Build a SignalSet; unspecified signals are NaN (missing)."""
    ss = SignalSet(ticker=ticker)
    for name in SIGNAL_NAMES:
        ss.sub_scores[name] = float(scores.get(name, NAN))
    return ss


# ====================== weights / config ======================
def test_conviction_weights_sum_to_one():
    ConvictionWeights.validate()
    assert abs(sum(ConvictionWeights.as_dict().values()) - 1.0) < 1e-9


def test_signal_names_match_weights():
    assert set(SIGNAL_NAMES) == set(ConvictionWeights.as_dict().keys())


# ====================== SignalSet ======================
def test_present_signals_excludes_nan():
    ss = _signal_set(fundamental=70.0, technical=NAN, quality=55.0)
    present = ss.present_signals()
    assert "fundamental" in present and "quality" in present
    assert "technical" not in present
    assert ss.coverage() == pytest.approx(2 / len(SIGNAL_NAMES))


# ====================== confidence ======================
def test_confidence_full_vs_partial():
    full = _signal_set(**{n: 60.0 for n in SIGNAL_NAMES})
    assert compute_confidence(full) == pytest.approx(1.0)

    only_fund = _signal_set(fundamental=60.0)
    assert compute_confidence(only_fund) == pytest.approx(ConvictionWeights.FUNDAMENTAL)


def test_confidence_factor_bounds():
    floor = ConvictionConfig.CONFIDENCE_FACTOR_FLOOR
    assert confidence_factor(0.0) == pytest.approx(floor)
    assert confidence_factor(1.0) == pytest.approx(1.0)
    assert floor < confidence_factor(0.5) < 1.0


# ====================== conviction engine ======================
def test_missing_signals_not_zero_filled():
    """A stock with only one strong signal must NOT be dragged toward 0 — the
    fillna(0) bias fix. Its base score equals that signal; the final score is
    only gently shrunk by confidence."""
    engine = ConvictionEngine()
    full = engine.score(_signal_set(**{n: 80.0 for n in SIGNAL_NAMES}))
    sparse = engine.score(_signal_set(fundamental=80.0))

    assert sparse.base_score == pytest.approx(80.0)      # re-normalized, not 80*0.3
    assert sparse.conviction > 65.0                      # shrunk, but not collapsed
    assert full.conviction == pytest.approx(80.0, abs=0.5)


def test_macro_tilt_is_bounded():
    engine = ConvictionEngine()
    ss = _signal_set(**{n: 60.0 for n in SIGNAL_NAMES})
    huge = engine.score(ss, macro_tilt=999.0)
    assert huge.macro_adjustment == pytest.approx(ConvictionConfig.MACRO_TILT_CAP)
    neg = engine.score(ss, macro_tilt=-999.0)
    assert neg.macro_adjustment == pytest.approx(-ConvictionConfig.MACRO_TILT_CAP)


def test_action_thresholds_and_holding_awareness():
    engine = ConvictionEngine()
    strong = engine.score(_signal_set(**{n: 95.0 for n in SIGNAL_NAMES}))
    assert strong.action == "STRONG_BUY"

    weak = _signal_set(**{n: 20.0 for n in SIGNAL_NAMES})
    held = engine.score(weak, held=True)
    not_held = engine.score(weak, held=False)
    assert held.action == "SELL"
    assert not_held.action == "AVOID"     # never "SELL" a stock you don't own


def test_no_signals_is_neutral():
    result = ConvictionEngine().score(_signal_set())
    assert result.conviction == pytest.approx(50.0)
    assert result.confidence == 0.0


def test_low_confidence_caps_strong_buy():
    """A 95-score stock backed only by fundamental data is downgraded to ADD."""
    result = ConvictionEngine().score(_signal_set(fundamental=99.0))
    assert result.confidence < ConvictionConfig.STRONG_BUY_MIN_CONFIDENCE
    assert result.action == "ADD"


# ====================== macro regime ======================
def _history(usdngn, brent):
    n = len(usdngn)
    return pd.DataFrame({"date": [f"2026-01-{i+1:02d}" for i in range(n)],
                         "mpr": [27.5] * n, "inflation": [23.0] * n,
                         "usdngn": usdngn, "brent": brent})


def test_naira_weakening_tilts_exporters_up():
    latest = {"date": "2026-03-01", "mpr": 27.5, "inflation": 23.0,
              "usdngn": 1700.0, "brent": 80.0}
    hist = _history(usdngn=[1500, 1550, 1600, 1700], brent=[80, 80, 80, 80])
    state = MacroRegime(latest, hist).classify()
    assert state.naira_trend == "WEAKENING"
    assert state.sector_tilts.get("AGRICULTURE", 0) > 0     # exporters benefit
    assert state.sector_tilts.get("CONSUMER", 0) < 0        # importers penalised


def test_oil_rising_tilts_oil_gas_up():
    latest = {"date": "2026-03-01", "mpr": 27.5, "inflation": 23.0,
              "usdngn": 1500.0, "brent": 95.0}
    hist = _history(usdngn=[1500] * 4, brent=[70, 78, 86, 95])
    state = MacroRegime(latest, hist).classify()
    assert state.oil_trend == "RISING"
    assert state.sector_tilts.get("OIL_GAS", 0) > 0


def test_flat_environment_has_no_tilts():
    latest = {"date": "2026-03-01", "mpr": 27.5, "inflation": 23.0,
              "usdngn": 1500.0, "brent": 80.0}
    hist = _history(usdngn=[1500] * 4, brent=[80] * 4)
    state = MacroRegime(latest, hist).classify()
    assert state.naira_trend == "STABLE" and state.oil_trend == "FLAT"
    assert all(v == 0 for v in state.sector_tilts.values())


# ====================== portfolio construction ======================
def _decision(ticker, conviction, action, sleeve="long_term", price=100.0,
              sector="OTHER"):
    score = ConvictionScore(ticker=ticker, conviction=conviction, action=action,
                            confidence=1.0)
    return Decision(ticker=ticker, sector=sector, price=price, score=score,
                    sleeve=sleeve)


def test_portfolio_deploys_capital_into_top_names():
    decisions = [
        _decision("AAA", 85, "STRONG_BUY", sector="BANKING"),
        _decision("BBB", 70, "ADD", sector="CONSUMER"),
        _decision("CCC", 40, "HOLD", sector="OTHER"),     # not a buy target
    ]
    orders = PortfolioConstructor().build(decisions, capital=1_000_000, holdings={})
    buys = [o for o in orders if o.side == "BUY"]
    assert buys, "expected at least one BUY order"
    assert all(o.ticker != "CCC" for o in buys)           # HOLD never bought
    deployed = sum(o.naira for o in buys)
    # Never deploy more than capital minus the cash reserve.
    assert deployed <= 1_000_000 * (1 - PortfolioConfig.CASH_RESERVE) + 1.0


def test_portfolio_sells_flagged_holding():
    decisions = [_decision("XXX", 20, "SELL", price=50.0)]
    holdings = {"XXX": {"shares": 1000, "avg_cost": 40.0}}
    orders = PortfolioConstructor().build(decisions, capital=0.0, holdings=holdings)
    sells = [o for o in orders if o.side == "SELL"]
    assert len(sells) == 1
    assert sells[0].ticker == "XXX" and sells[0].shares == 1000


def test_portfolio_respects_max_position_cap():
    # One dominant name — its buy must still be capped at MAX_POSITION of book.
    decisions = [_decision("AAA", 99, "STRONG_BUY", price=10.0)]
    orders = PortfolioConstructor().build(decisions, capital=10_000_000, holdings={})
    buys = [o for o in orders if o.side == "BUY"]
    total = 10_000_000
    for o in buys:
        assert o.naira <= PortfolioConfig.MAX_POSITION * total + 1.0


# ====================== calibration ======================
def test_spearman_basic():
    a = pd.Series(range(20))
    assert _spearman(a, a) == pytest.approx(1.0)
    assert _spearman(a, a[::-1].reset_index(drop=True)) == pytest.approx(-1.0)
    assert _spearman(pd.Series([1, 2]), pd.Series([1, 2])) is None  # too few


def test_evaluate_pair_detects_predictive_factor():
    """A factor that perfectly orders the next-period return scores Spearman 1."""
    syms = [f"S{i}" for i in range(20)]
    prior = pd.DataFrame({"symbol": syms, "price": [100.0] * 20,
                          "eps_growth_ttm": list(range(20))})
    # later price rises in lock-step with prior eps_growth -> perfect predictor
    later = pd.DataFrame({"symbol": syms,
                          "price": [100.0 * (1 + i / 100) for i in range(20)]})
    result = evaluate_pair(prior, later)
    assert result.get("eps_growth") == pytest.approx(1.0)

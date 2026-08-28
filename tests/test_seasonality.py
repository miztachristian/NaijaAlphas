"""
Tests for seasonality analysis.

Uses synthetic monthly-return series with a known pattern (May always +5%,
November always -3%) to verify the SeasonalAnalyzer's computations.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.seasonality import SeasonalAnalyzer, _parse_month
from config.settings import SeasonalityConfig


def _make_monthly_series(
    start_year: int = 2020,
    n_years: int = 5,
    strong_month: int = 5,
    strong_return: float = 0.05,
    weak_month: int = 11,
    weak_return: float = -0.03,
    base_return: float = 0.005,
    seed: int = 1,
) -> pd.Series:
    """Construct synthetic monthly returns indexed at month-end."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(start_year, start_year + n_years):
        for m in range(1, 13):
            day = 28
            if m == strong_month:
                r = strong_return
            elif m == weak_month:
                r = weak_return
            else:
                r = base_return + float(rng.normal(0, 0.005))
            rows.append((datetime(y, m, day), r))
    idx = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


class _FakeLoader:
    """Minimal stand-in for NGXDataLoader that returns canned monthly returns."""

    def __init__(self, series_by_ticker):
        self.series_by_ticker = series_by_ticker

    def get_returns(self, ticker, period="monthly"):
        return self.series_by_ticker.get(ticker)


# ──────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────

def test_avg_returns_recover_planted_pattern():
    series = _make_monthly_series()
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    m = sa.analyze("TEST")
    assert m is not None
    assert m.monthly_avg_returns[5] == pytest.approx(0.05, abs=1e-9)
    assert m.monthly_avg_returns[11] == pytest.approx(-0.03, abs=1e-9)


def test_win_rate_is_one_for_uniformly_strong_month():
    series = _make_monthly_series()
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    m = sa.analyze("TEST")
    assert m.monthly_win_rates[5] == pytest.approx(1.0)
    assert m.monthly_win_rates[11] == pytest.approx(0.0)


def test_sample_sizes_match_year_count():
    series = _make_monthly_series(n_years=5)
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    m = sa.analyze("TEST")
    for month in range(1, 13):
        assert m.monthly_sample_sizes[month] == 5


def test_best_month_is_planted_strong_month():
    series = _make_monthly_series(strong_month=5, weak_month=11)
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    m = sa.analyze("TEST")
    assert m.best_months[0][0] == 5
    assert m.worst_months[0][0] == 11


# ──────────────────────────────────────────────────────────────────
# Minimum-sample guard
# ──────────────────────────────────────────────────────────────────

def test_insufficient_samples_yield_zero_score():
    """When a stock has <MIN_YEARS samples per month, score must be 0."""
    short_series = _make_monthly_series(n_years=SeasonalityConfig.MIN_YEARS - 1)
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": short_series}))
    m = sa.analyze("TEST")
    assert m.current_month_score == 0.0
    assert m.next_month_score == 0.0
    assert m.best_months == []
    assert m.worst_months == []


# ──────────────────────────────────────────────────────────────────
# Current-month sign
# ──────────────────────────────────────────────────────────────────

@patch("analysis.seasonality.datetime")
def test_current_month_score_positive_in_strong_month(mock_dt):
    mock_dt.now.return_value = datetime(2026, 5, 15)
    mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

    series = _make_monthly_series(strong_month=5, weak_month=11)
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    m = sa.analyze("TEST")
    assert m.current_month == 5
    assert m.current_month_score > 0.5


@patch("analysis.seasonality.datetime")
def test_current_month_score_negative_in_weak_month(mock_dt):
    mock_dt.now.return_value = datetime(2026, 11, 15)
    mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

    series = _make_monthly_series(strong_month=5, weak_month=11)
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    m = sa.analyze("TEST")
    assert m.current_month == 11
    assert m.current_month_score < -0.5


# ──────────────────────────────────────────────────────────────────
# Scoring layer
# ──────────────────────────────────────────────────────────────────

def test_score_seasonal_neutral_on_missing_metrics():
    sa = SeasonalAnalyzer(loader=_FakeLoader({}))
    scores = sa.score_seasonal(None)
    assert scores["seasonal_current_score"] == pytest.approx(0.5)
    assert scores["seasonal_consistency_score"] == pytest.approx(0.5)


def test_score_seasonal_in_unit_interval():
    series = _make_monthly_series()
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    m = sa.analyze("TEST")
    scores = sa.score_seasonal(m)
    assert 0.0 <= scores["seasonal_current_score"] <= 1.0
    assert 0.0 <= scores["seasonal_consistency_score"] <= 1.0


# ──────────────────────────────────────────────────────────────────
# Month parsing
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (5, 5),
    ("5", 5),
    ("MAY", 5),
    ("May", 5),
    ("may", 5),
    ("JUN", 6),
    ("June", 6),
    (12, 12),
])
def test_parse_month_accepts_variants(value, expected):
    assert _parse_month(value) == expected


def test_parse_month_rejects_invalid():
    with pytest.raises(ValueError):
        _parse_month("Smarch")
    with pytest.raises(ValueError):
        _parse_month(0)
    with pytest.raises(ValueError):
        _parse_month(13)


# ──────────────────────────────────────────────────────────────────
# Table generation
# ──────────────────────────────────────────────────────────────────

def test_seasonal_table_has_average_row_and_year_rows():
    series = _make_monthly_series(start_year=2020, n_years=5)
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    table = sa.generate_seasonal_table("TEST")
    assert table is not None
    assert "Average" in table.index
    assert 2024 in table.index
    assert len(table.columns) == 12
    # May average should equal +5.0 (after *100 conversion)
    assert table.loc["Average", "May"] == pytest.approx(5.0, abs=1e-9)


# ──────────────────────────────────────────────────────────────────
# Label helper
# ──────────────────────────────────────────────────────────────────

@patch("analysis.seasonality.datetime")
def test_label_current_month_describes_strong_month(mock_dt):
    mock_dt.now.return_value = datetime(2026, 5, 15)
    mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

    series = _make_monthly_series(strong_month=5)
    sa = SeasonalAnalyzer(loader=_FakeLoader({"TEST": series}))
    m = sa.analyze("TEST")
    label = sa.label_current_month(m)
    assert "Strong" in label
    assert "May" in label

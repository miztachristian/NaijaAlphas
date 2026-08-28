"""
Tests for the NGX disclosure pipeline.

Covers the pure logic of the disclosure feed and the three integration
tiers — category normalisation, disclosure event signals, earnings-forecast
parsing and financial-statement parsing — using synthetic inputs (no network).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.fetch_disclosures import _categorise
from ingest.parse_forecasts import _to_number, _parse_pnl, _validate as validate_forecast
from ingest.parse_statements import (
    _grab, _pct_change, _validate as validate_statement, PNL_LABELS, POS_LABELS,
)
from analysis.disclosure_analyzer import DisclosureAnalyzer
from analysis.growth import GrowthAnalyzer
from analysis.fundamental import FundamentalAnalyzer
from core.models import FundamentalMetrics
from config.settings import DisclosureScoring

AS_OF = datetime(2026, 5, 22, tzinfo=timezone.utc)


class TestCategorise:
    """Type_of_Submission is filed inconsistently; _categorise normalises it."""

    def test_financial_statement_variants(self):
        for raw in ("Financial Statements", "FINANCIAL STATEMENTS",
                    "Financial Statement", "Corporate Actions Financial Statements"):
            assert _categorise(raw) == "financial_statement"

    def test_earnings_forecast_variants(self):
        for raw in ("EarningForcast", "Earnings Forecast", "EARNINGFORCAST"):
            assert _categorise(raw) == "earnings_forecast"

    def test_directors_dealings_variants(self):
        for raw in ("DirectorsDealings", "Directors Dealings", "DIRECTORSDEALINGS"):
            assert _categorise(raw) == "directors_dealings"

    def test_meetings(self):
        assert _categorise("Board Meeting (BM)") == "board_meeting"
        assert _categorise("Annual General Meeting (AGM)") == "agm"
        assert _categorise("Extra-Ordinary General Meeting (EGM)") == "egm"

    def test_unknown_is_other(self):
        assert _categorise("") == "other"
        assert _categorise("Some Novel Filing Type") == "other"


class TestDisclosureAnalyzer:
    """Event signals derived from synthetic disclosure rows."""

    @staticmethod
    def _rows():
        return [
            {"ticker": "TESTCO", "category": "financial_statement",
             "type": "Financial Statements", "title": "TESTCO Q1 RESULTS",
             "created": "2026-05-17T00:00:00Z"},
            {"ticker": "TESTCO", "category": "board_meeting",
             "type": "Board Meeting (BM)", "title": "TESTCO BOARD MEETING",
             "created": "2026-05-12T00:00:00Z"},
            {"ticker": "TESTCO", "category": "directors_dealings",
             "type": "DirectorsDealings", "title": "TESTCO DIRECTORSDEALINGS",
             "created": "2026-04-25T00:00:00Z"},
            {"ticker": "TESTCO", "category": "corporate_action",
             "type": "Corporate Actions",
             "title": "TESTCO DECLARATION OF DIVIDEND PAYMENT",
             "created": "2026-05-02T00:00:00Z"},
            {"ticker": "LATECO", "category": "corporate_action",
             "type": "Corporate Actions",
             "title": "LATECO NOTICE OF DELAY IN FILING 2025 ACCOUNTS",
             "created": "2026-03-01T00:00:00Z"},
        ]

    def _analyzer(self):
        return DisclosureAnalyzer(rows=self._rows(), as_of=AS_OF)

    def test_earnings_catalyst_from_fresh_results(self):
        signals = self._analyzer().analyze("TESTCO")
        assert signals["earnings_catalyst"] is True
        assert "Results filed" in signals["catalyst_reason"]

    def test_insider_activity(self):
        signals = self._analyzer().analyze("TESTCO")
        assert signals["insider_activity"] is True
        assert signals["insider_count_90d"] == 1
        assert 0 < signals["insider_score"] <= 1

    def test_dividend_recent(self):
        signals = self._analyzer().analyze("TESTCO")
        assert signals["dividend_recent"] is True
        assert signals["last_dividend_date"] == "2026-05-02"

    def test_days_since_results(self):
        signals = self._analyzer().analyze("TESTCO")
        assert signals["days_since_results"] == 5

    def test_late_filer_from_delay_notice(self):
        signals = self._analyzer().analyze("LATECO")
        assert signals["late_filer"] is True
        assert "Delay notice" in signals["late_filer_reason"]

    def test_unknown_ticker_is_neutral(self):
        signals = self._analyzer().analyze("NOSUCHCO")
        assert signals["earnings_catalyst"] is False
        assert signals["late_filer"] is False
        assert signals["disclosure_count_90d"] == 0

    def test_case_insensitive_ticker(self):
        assert self._analyzer().analyze("testco")["dividend_recent"] is True


class TestForecastParsing:
    """Earnings-forecast PDF text parsing and sanity validation."""

    def test_to_number_plain_and_grouped(self):
        assert _to_number("1,108,722") == 1108722.0
        assert _to_number("788870") == 788870.0

    def test_to_number_parentheses_is_negative(self):
        assert _to_number("(319,852)") == -319852.0

    def test_parse_pnl_extracts_three_metrics(self):
        text = ("Insurance Revenue 6,187,484\n"
                "Insurance Service Expenses (2,333,593)\n"
                "Profit Before Tax 1,108,722\n"
                "Profit After Tax 788,870")
        pnl = _parse_pnl(text)
        assert pnl["forecast_revenue"] == 6187484
        assert pnl["forecast_pbt"] == 1108722
        assert pnl["forecast_pat"] == 788870

    def test_validate_flags_revenue_below_pbt(self):
        record = {"forecast_revenue": 30, "forecast_pbt": 1_500_000,
                  "forecast_pat": 1_000_000, "parse_status": "ok"}
        validate_forecast(record)
        assert record["parse_status"] == "suspect"

    def test_validate_flags_implausible_magnitude(self):
        record = {"forecast_revenue": None, "forecast_pbt": 9_125_351_139,
                  "forecast_pat": 6_033_550_642, "parse_status": "partial"}
        validate_forecast(record)
        assert record["parse_status"] == "suspect"

    def test_validate_passes_clean_record(self):
        record = {"forecast_revenue": 32_886_674, "forecast_pbt": 6_375_404,
                  "forecast_pat": 4_207_766, "parse_status": "ok"}
        validate_forecast(record)
        assert record["parse_status"] == "ok"


class TestStatementParsing:
    """Financial-statement PDF parsing — note-ref skipping and validation."""

    def test_grab_skips_note_reference_number(self):
        # "Revenue 7 330,639,543 ..." — the bare "7" is a note ref, not a value.
        line = "Revenue 7 330,639,543 207,504,191 208,458,388 153,225,834"
        grabbed = _grab(line, PNL_LABELS)
        assert grabbed["revenue"] == 330639543
        assert grabbed["revenue_prior"] == 207504191

    def test_grab_profit_before_tax(self):
        line = "Profit before income tax 17,758,932 1,070,554"
        grabbed = _grab(line, PNL_LABELS)
        assert grabbed["pbt"] == 17758932

    def test_grab_position_total_equity_not_liabilities(self):
        text = ("Total equity and liabilities 965,925,567\n"
                "Total equity 128,980,167")
        grabbed = _grab(text, POS_LABELS)
        assert grabbed["total_equity"] == 128980167

    def test_pct_change(self):
        assert _pct_change(120.0, 100.0) == 20.0
        assert _pct_change(80.0, 100.0) == -20.0
        assert _pct_change(100.0, None) is None
        assert _pct_change(100.0, 0) is None

    def test_validate_flags_equity_exceeding_assets(self):
        record = {"revenue": 1000, "pbt": 200, "pat": 150,
                  "total_assets": 5000, "total_equity": 9000,
                  "parse_status": "ok"}
        validate_statement(record)
        assert record["parse_status"] == "suspect"

    def test_validate_passes_clean_record(self):
        record = {"revenue": 100_861_201, "pbt": 69_244_488, "pat": 49_257_369,
                  "total_assets": 400_000_000, "total_equity": 250_000_000,
                  "parse_status": "ok"}
        validate_statement(record)
        assert record["parse_status"] == "ok"


@pytest.fixture
def restore_disclosure_config():
    """Snapshot and restore DisclosureScoring (tests mutate it globally)."""
    keys = ("ENABLED", "USE_NGX_FUNDAMENTALS", "ADJUSTMENT_CAP",
            "LATE_FILER_PENALTY", "STALE_RESULTS_PENALTY", "STALE_DAYS",
            "DIVIDEND_BONUS")
    saved = {k: getattr(DisclosureScoring, k) for k in keys}
    yield
    for k, v in saved.items():
        setattr(DisclosureScoring, k, v)


class TestDisclosureAdjustment:
    """The capped, sign-unambiguous adjustment folded into growth_score."""

    @staticmethod
    def _adj(signals):
        return GrowthAnalyzer()._calculate_disclosure_adjustment(signals)

    def test_no_signals_is_zero(self):
        adjustment, reasons = self._adj({})
        assert adjustment == 0.0
        assert reasons == []

    def test_late_filer_penalty(self):
        adjustment, reasons = self._adj({"late_filer": True})
        assert adjustment == -3.0
        assert any("late filer" in r for r in reasons)

    def test_dividend_bonus(self):
        adjustment, _ = self._adj({"dividend_recent": True})
        assert adjustment == 1.0

    def test_late_and_stale_clamp_at_cap(self):
        adjustment, _ = self._adj({"late_filer": True, "days_since_results": 300})
        assert adjustment == -5.0  # -3 + -2, exactly the cap

    def test_late_stale_dividend_net(self):
        adjustment, _ = self._adj({"late_filer": True, "days_since_results": 300,
                                   "dividend_recent": True})
        assert adjustment == -4.0  # -3 - 2 + 1

    def test_fresh_results_not_stale(self):
        adjustment, _ = self._adj({"days_since_results": 30})
        assert adjustment == 0.0

    def test_disabled_returns_zero(self, restore_disclosure_config):
        DisclosureScoring.ENABLED = False
        adjustment, _ = self._adj({"late_filer": True})
        assert adjustment == 0.0


class TestNgxFundamentalScoring:
    """score_fundamentals prefers NGX-official figures over the snapshot."""

    def test_ngx_roe_preferred_when_present(self, restore_disclosure_config):
        DisclosureScoring.USE_NGX_FUNDAMENTALS = True
        fa = FundamentalAnalyzer()
        m = FundamentalMetrics(ticker="TESTCO")
        m.roe = 4.0                        # weak snapshot value
        m.ngx_figures = {"ngx_roe": 30.0}  # strong NGX-official value
        scores = fa.score_fundamentals(m)
        assert scores["roe_score"] == fa._score_roe(30.0)
        assert scores["roe_score"] > fa._score_roe(4.0)

    def test_snapshot_used_when_no_ngx_figure(self, restore_disclosure_config):
        fa = FundamentalAnalyzer()
        m = FundamentalMetrics(ticker="TESTCO")
        m.roe = 22.0
        scores = fa.score_fundamentals(m)
        assert scores["roe_score"] == fa._score_roe(22.0)

    def test_toggle_off_ignores_ngx(self, restore_disclosure_config):
        DisclosureScoring.USE_NGX_FUNDAMENTALS = False
        fa = FundamentalAnalyzer()
        m = FundamentalMetrics(ticker="TESTCO")
        m.roe = 4.0
        m.ngx_figures = {"ngx_roe": 30.0}
        scores = fa.score_fundamentals(m)
        assert scores["roe_score"] == fa._score_roe(4.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

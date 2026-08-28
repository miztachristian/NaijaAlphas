"""Unit tests for the new daily_ingest pipeline steps (step bodies mocked)."""
import sys

import pytest

import daily_ingest as di
from ingest.run_notebooks import NotebookResult


def test_step_notebooks_all_ok(monkeypatch):
    monkeypatch.setattr(
        "ingest.run_notebooks.run_notebooks",
        lambda only=None: [NotebookResult("a.ipynb", "ok"),
                           NotebookResult("b.ipynb", "ok")],
    )
    r = di.step_notebooks()
    assert r.name == "notebooks"
    assert r.status == "ok"
    assert "2/2 ok" in r.message


def test_step_notebooks_one_failed_sets_failed(monkeypatch):
    monkeypatch.setattr(
        "ingest.run_notebooks.run_notebooks",
        lambda only=None: [NotebookResult("a.ipynb", "ok"),
                           NotebookResult("b.ipynb", "failed", "boom")],
    )
    r = di.step_notebooks()
    assert r.status == "failed"
    assert "b.ipynb" in r.message


def test_step_seasonality_invokes_run_analysis_with_month(monkeypatch):
    calls = {}

    def fake_run_module(module, args=None, timeout=600):
        calls["module"] = module
        calls["args"] = args or []
        return True, "ok"

    monkeypatch.setattr(di, "_run_module", fake_run_module)
    r = di.step_seasonality()
    assert r.name == "seasonality"
    assert r.status == "ok"
    assert calls["module"] == "run_analysis"
    assert calls["args"][0] == "--seasonal-month"
    # month is passed as a string number, e.g. "6" for June
    assert calls["args"][1].isdigit()


def test_main_runs_notebooks_then_seasonality_then_notify(monkeypatch):
    order = []

    monkeypatch.setattr(di, "step_decisions",
                        lambda date, capital: (di.StepResult("decisions", "ok"), object()))
    monkeypatch.setattr(di, "step_notebooks",
                        lambda: order.append("notebooks") or di.StepResult("notebooks", "ok"))
    monkeypatch.setattr(di, "step_seasonality",
                        lambda: order.append("seasonality") or di.StepResult("seasonality", "ok"))
    monkeypatch.setattr(di, "step_notify",
                        lambda result: order.append("notify") or di.StepResult("notify", "ok"))
    monkeypatch.setattr(di, "_write_report", lambda steps, date: None)
    monkeypatch.setattr(sys, "argv",
                        ["daily_ingest.py", "--skip-ingest", "--date", "2026-06-03"])

    with pytest.raises(SystemExit) as exc:
        di.main()
    assert exc.value.code == 0
    assert order == ["notebooks", "seasonality", "notify"]

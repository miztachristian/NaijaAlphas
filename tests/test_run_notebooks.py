"""Unit tests for ingest.run_notebooks (subprocess mocked — no real execution)."""
from ingest import run_notebooks as rn


def test_curated_list_is_nine_notebooks_in_order():
    assert rn.DAILY_NOTEBOOKS == [
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
    # personal-portfolio notebooks are intentionally excluded
    assert not any("portfolio_tracker" in n or "sister" in n for n in rn.DAILY_NOTEBOOKS)


def test_missing_notebook_returns_missing():
    res = rn.execute_notebook("does_not_exist_xyz.ipynb")
    assert res.status == "missing"
    assert res.name == "does_not_exist_xyz.ipynb"


def test_execute_notebook_success(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "done"
        stderr = ""
    monkeypatch.setattr(rn.subprocess, "run", lambda *a, **k: _Proc())
    # uses a real curated notebook so the existence check passes
    res = rn.execute_notebook("analysis_notebook.ipynb")
    assert res.status == "ok"


def test_execute_notebook_failure_captures_stderr_tail(monkeypatch):
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "line1\nline2\nCellExecutionError: boom"
    monkeypatch.setattr(rn.subprocess, "run", lambda *a, **k: _Proc())
    res = rn.execute_notebook("analysis_notebook.ipynb")
    assert res.status == "failed"
    assert "boom" in res.message


def test_execute_notebook_timeout(monkeypatch):
    import subprocess
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nbconvert", timeout=600)
    monkeypatch.setattr(rn.subprocess, "run", _raise)
    res = rn.execute_notebook("analysis_notebook.ipynb")
    assert res.status == "failed"
    assert "timed out" in res.message


def test_run_notebooks_only_subset(monkeypatch):
    monkeypatch.setattr(
        rn, "execute_notebook",
        lambda name, timeout=rn.NOTEBOOK_TIMEOUT: rn.NotebookResult(name, "ok"),
    )
    results = rn.run_notebooks(only=["a.ipynb", "b.ipynb"])
    assert [r.name for r in results] == ["a.ipynb", "b.ipynb"]

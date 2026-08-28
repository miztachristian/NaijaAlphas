"""
Seasonality Analysis Module
===========================
Computes per-stock monthly seasonal patterns from historical price returns.

Patterns are computed from the monthly returns already exposed by
NGXDataLoader.get_returns(period="monthly"). No external data source needed.
"""

import calendar
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import SeasonalityConfig
from core.data_loader import NGXDataLoader
from core.models import SeasonalMetrics


MONTH_NAMES = {i: calendar.month_name[i] for i in range(1, 13)}
MONTH_ABBR = {i: calendar.month_abbr[i] for i in range(1, 13)}


def _parse_month(value) -> int:
    """Accept int (1-12), full name ('May'), or abbreviation ('MAY')."""
    if isinstance(value, (int, np.integer)):
        m = int(value)
    else:
        s = str(value).strip().lower()
        if s.isdigit():
            m = int(s)
        else:
            for i in range(1, 13):
                if calendar.month_name[i].lower() == s or calendar.month_abbr[i].lower() == s:
                    m = i
                    break
            else:
                raise ValueError(f"Unrecognized month: {value!r}")
    if not 1 <= m <= 12:
        raise ValueError(f"Month must be in 1..12, got {m}")
    return m


class SeasonalAnalyzer:
    """
    Compute monthly seasonal patterns for NGX stocks.

    Mirrors the analyzer pattern used by FundamentalAnalyzer / TechnicalAnalyzer:
      - analyze(ticker)        -> SeasonalMetrics
      - score_seasonal(metrics) -> dict of 0..1 scores
    """

    def __init__(self, loader: Optional[NGXDataLoader] = None):
        self.loader = loader or NGXDataLoader()

    # -------- Core computation --------

    def _monthly_returns_by_year(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return a wide DataFrame indexed by year, columns 1..12 of monthly returns."""
        monthly = self.loader.get_returns(ticker, period="monthly")
        if monthly is None or monthly.empty:
            return None

        monthly = monthly.dropna()
        if monthly.empty:
            return None

        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        })
        # Last calendar day per month -> a single value per (year, month)
        df = df.groupby(["year", "month"], as_index=False)["ret"].last()
        wide = df.pivot(index="year", columns="month", values="ret")
        # Ensure all 12 month columns exist
        for m in range(1, 13):
            if m not in wide.columns:
                wide[m] = np.nan
        return wide[sorted(wide.columns)]

    def analyze(self, ticker: str) -> Optional[SeasonalMetrics]:
        wide = self._monthly_returns_by_year(ticker)
        if wide is None or wide.empty:
            return None

        avg, median, win, n, std = {}, {}, {}, {}, {}
        for m in range(1, 13):
            col = wide[m].dropna()
            if len(col) == 0:
                avg[m] = float("nan")
                median[m] = float("nan")
                win[m] = float("nan")
                n[m] = 0
                std[m] = float("nan")
                continue
            avg[m] = float(col.mean())
            median[m] = float(col.median())
            win[m] = float((col > 0).sum() / len(col))
            n[m] = int(len(col))
            std[m] = float(col.std(ddof=0)) if len(col) > 1 else 0.0

        # Cross-section reference (for normalizing one month vs the others)
        valid_avgs = [v for m, v in avg.items() if n[m] >= SeasonalityConfig.MIN_YEARS and not np.isnan(v)]
        ref_mean = float(np.mean(valid_avgs)) if valid_avgs else 0.0
        ref_std = float(np.std(valid_avgs, ddof=0)) if len(valid_avgs) > 1 else 0.01

        now = datetime.now()
        current_m = now.month
        next_m = 1 if current_m == 12 else current_m + 1

        current_score = self._month_score(current_m, avg, win, n, ref_mean, ref_std)
        next_score = self._month_score(next_m, avg, win, n, ref_mean, ref_std)

        # Best / worst months (only among those with sufficient samples)
        scoreable = [(m, avg[m]) for m in range(1, 13)
                     if n[m] >= SeasonalityConfig.MIN_YEARS and not np.isnan(avg[m])]
        scoreable.sort(key=lambda x: x[1], reverse=True)
        best_months = scoreable[:3]
        worst_months = list(reversed(scoreable[-3:]))

        # Per-year-per-month dictionary for matrix display
        yearly_monthly: Dict[int, Dict[int, float]] = {}
        for y in wide.index:
            row = wide.loc[y]
            yearly_monthly[int(y)] = {
                int(m): (float(v) if not pd.isna(v) else float("nan"))
                for m, v in row.items()
            }

        return SeasonalMetrics(
            ticker=ticker,
            monthly_avg_returns=avg,
            monthly_median_returns=median,
            monthly_win_rates=win,
            monthly_sample_sizes=n,
            monthly_stdev=std,
            yearly_monthly=yearly_monthly,
            current_month=current_m,
            current_month_score=current_score,
            next_month_score=next_score,
            best_months=best_months,
            worst_months=worst_months,
            years_of_data=int(len(wide.index)),
        )

    @staticmethod
    def _month_score(
        m: int,
        avg: Dict[int, float],
        win: Dict[int, float],
        n: Dict[int, int],
        ref_mean: float,
        ref_std: float,
    ) -> float:
        """
        Score a specific calendar month in [-1, +1].
        Combines z-scored avg return vs the stock's own cross-section
        with a win-rate centering term. Returns 0 if sample size < MIN_YEARS.
        """
        if n.get(m, 0) < SeasonalityConfig.MIN_YEARS:
            return 0.0
        a = avg.get(m, float("nan"))
        w = win.get(m, float("nan"))
        if np.isnan(a) or np.isnan(w):
            return 0.0
        z = (a - ref_mean) / (ref_std if ref_std > 1e-9 else 1e-9)
        magnitude = float(np.tanh(z))           # squashes to [-1, +1]
        consistency = (w - 0.5) * 2.0           # [-1, +1] around 50%
        score = 0.7 * magnitude + 0.3 * consistency
        return float(max(-1.0, min(1.0, score)))

    # -------- Scoring --------

    def score_seasonal(self, metrics: Optional[SeasonalMetrics]) -> Dict[str, float]:
        """
        Convert SeasonalMetrics into 0..1 sub-scores for the growth pipeline.
        Returns neutral 0.5 when metrics are unavailable.
        """
        if metrics is None:
            return {"seasonal_current_score": 0.5, "seasonal_consistency_score": 0.5}

        current = (metrics.current_month_score + 1.0) / 2.0  # -1..1 -> 0..1

        if metrics.best_months:
            best_win_rates = [
                metrics.monthly_win_rates.get(m, 0.0)
                for m, _ in metrics.best_months
                if metrics.monthly_sample_sizes.get(m, 0) >= SeasonalityConfig.MIN_YEARS
            ]
            consistency = float(np.mean(best_win_rates)) if best_win_rates else 0.5
        else:
            consistency = 0.5

        return {
            "seasonal_current_score": float(max(0.0, min(1.0, current))),
            "seasonal_consistency_score": float(max(0.0, min(1.0, consistency))),
        }

    # -------- Labels / display --------

    def label_current_month(self, metrics: Optional[SeasonalMetrics]) -> str:
        """Human-readable label like 'Strong May (+4.3%, 80% win, n=5)'."""
        if metrics is None or metrics.current_month == 0:
            return "N/A"
        m = metrics.current_month
        n = metrics.monthly_sample_sizes.get(m, 0)
        if n < SeasonalityConfig.MIN_YEARS:
            return f"{MONTH_NAMES[m]} (insufficient data, n={n})"
        avg = metrics.monthly_avg_returns.get(m, float("nan"))
        win = metrics.monthly_win_rates.get(m, float("nan"))
        if avg >= SeasonalityConfig.STRONG_MONTH_THRESHOLD:
            tone = "Strong"
        elif avg <= SeasonalityConfig.WEAK_MONTH_THRESHOLD:
            tone = "Weak"
        else:
            tone = "Mixed"
        return f"{tone} {MONTH_NAMES[m]} ({avg*100:+.1f}%, {win*100:.0f}% win, n={n})"

    def is_warning(self, metrics: Optional[SeasonalMetrics]) -> bool:
        if metrics is None:
            return False
        return metrics.current_month_score < SeasonalityConfig.WARNING_SCORE

    # -------- Tables / cross-section queries --------

    def generate_seasonal_table(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        TradingView-style year x month matrix with an Average row at the top.
        Values are percentages (multiplied by 100 for readability).
        """
        wide = self._monthly_returns_by_year(ticker)
        if wide is None or wide.empty:
            return None
        out = (wide * 100.0).copy()
        out.columns = [MONTH_ABBR[m] for m in out.columns]
        out.index.name = "Year"
        avg_row = out.mean(axis=0, skipna=True).to_frame().T
        avg_row.index = ["Average"]
        out = pd.concat([avg_row, out.sort_index(ascending=False)])
        return out

    def rank_by_month(
        self,
        month,
        tickers: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Rank tickers by their historical avg return in the given calendar month."""
        from config.settings import get_all_tickers, get_ticker_sector

        month_int = _parse_month(month)
        tickers = tickers or get_all_tickers()

        rows: List[Dict] = []
        total = len(tickers)
        for i, t in enumerate(tickers):
            if show_progress:
                print(f"  Ranking {t} ({i+1}/{total})...", end="\r")
            metrics = self.analyze(t)
            if metrics is None:
                continue
            n = metrics.monthly_sample_sizes.get(month_int, 0)
            if n < SeasonalityConfig.MIN_YEARS:
                continue
            rows.append({
                "ticker": t,
                "sector": get_ticker_sector(t),
                "month": MONTH_NAMES[month_int],
                "avg_return_pct": metrics.monthly_avg_returns.get(month_int, np.nan) * 100,
                "median_return_pct": metrics.monthly_median_returns.get(month_int, np.nan) * 100,
                "win_rate_pct": metrics.monthly_win_rates.get(month_int, np.nan) * 100,
                "stdev_pct": metrics.monthly_stdev.get(month_int, np.nan) * 100,
                "samples": n,
            })
        if show_progress:
            print(" " * 60, end="\r")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("avg_return_pct", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def build_universe_matrix(
        self,
        tickers: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """
        Build a (ticker x month) matrix of average monthly returns (%).
        Used by the heatmap CLI and the notebook.
        """
        from config.settings import get_all_tickers

        tickers = tickers or get_all_tickers()
        records: Dict[str, Dict[str, float]] = {}
        total = len(tickers)
        for i, t in enumerate(tickers):
            if show_progress:
                print(f"  Building seasonal matrix {t} ({i+1}/{total})...", end="\r")
            metrics = self.analyze(t)
            if metrics is None:
                continue
            row = {}
            for m in range(1, 13):
                n = metrics.monthly_sample_sizes.get(m, 0)
                if n < SeasonalityConfig.MIN_YEARS:
                    row[MONTH_ABBR[m]] = np.nan
                else:
                    row[MONTH_ABBR[m]] = metrics.monthly_avg_returns.get(m, np.nan) * 100
            records[t] = row
        if show_progress:
            print(" " * 60, end="\r")

        if not records:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(records, orient="index")
        df = df[[MONTH_ABBR[m] for m in range(1, 13)]]
        df.index.name = "ticker"
        return df

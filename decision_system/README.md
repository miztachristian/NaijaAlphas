# decision_system

A unified NGX decision engine. It fuses every existing analysis signal plus a
Nigeria macro regime into **one explainable 0–100 conviction score + action per
stock**, constructs a concrete portfolio, and runs as an automated daily pipeline.

It *orchestrates* the analyzers in `analysis/` — it never re-implements their
scoring math.

## Daily use

```bash
python daily_ingest.py --capital 1000000
```

Runs, each step failure-isolated: macro ingest → news → disclosures → snapshot
(uses today's manual export if present, else builds from the TradingView
scanner API) → decision pipeline → writes `outputs/decisions/<date>/`.

Schedule it with Windows Task Scheduler via `_run_daily.bat` (see that file's
header for the `schtasks` command).

## Outputs (`outputs/decisions/<date>/`)

| File | Contents |
|---|---|
| `decision_table.csv` | Per stock: conviction, action, all 8 sub-scores, confidence, reasons |
| `orders.csv` | Concrete BUY / TRIM / SELL list |
| `macro_summary.md` | Macro regime + active sector tilts |
| `run_manifest.json` | Per-run stats, degraded sources, weight-set version |

## How the conviction score works

1. Eight signals (`signals.py`) — fundamental, technical, quality, seasonality,
   disclosure, report tone, news sentiment, market context — each normalized to
   0–100, or **NaN when there is no data**.
2. Weighted sum (`conviction.py`) with weights from `config.ConvictionWeights`,
   **re-normalized over only the signals that have data** — a missing signal is
   excluded, never zero-filled.
3. A bounded macro regime tilt (`macro_regime.py`, ±`MACRO_TILT_CAP` points).
4. Confidence shrinkage (`confidence.py`) — a thinly-covered stock is pulled
   toward neutral (50) instead of being penalised.
5. A holding-aware action: STRONG_BUY / ADD / HOLD / TRIM / SELL / AVOID.

No ML — with ~30 historical snapshots a model would fit noise. The weighted
scheme is auditable and calibratable.

## Modules

| File | Responsibility |
|---|---|
| `models.py` | Dataclasses (pure data) |
| `signals.py` | Collect + normalize the 8 signals |
| `confidence.py` | Data-coverage confidence + shrinkage factor |
| `macro_regime.py` | Classify macro regime → sector tilts |
| `conviction.py` | Fuse signals → conviction score + action |
| `portfolio.py` | Conviction-weighted sizing → order list |
| `calibration.py` | Validate signals vs realized forward returns |
| `pipeline.py` | Orchestrate a full run |
| `outputs.py` | Write the run artifacts |

Supporting: `ingest/macro.py` (macro series), `ingest/build_snapshot.py`
(API-based snapshot), `daily_ingest.py` (orchestrator).

## Calibration

```bash
python -m decision_system.calibration
```

Measures each factor's Spearman correlation with forward returns across all
stored snapshot pairs and writes `outputs/calibration/<date>_weight_report.md`.
A human reviews it and edits `config.ConvictionWeights` — nothing auto-writes.

## Tuning

All weights, thresholds and caps live in `config/settings.py`:
`ConvictionWeights`, `ConvictionConfig`, `MacroConfig`, `PortfolioConfig`.

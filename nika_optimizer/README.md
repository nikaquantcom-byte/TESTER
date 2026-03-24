# NikaQuant V7 Pro — Python Optimizer

**XAUUSD M10 Parameter Optimization for TradingView Pine Script**

Massive parameter sweep engine that finds the best settings for the
"Trend Duration Forecast + Optimizer V7 Pro" indicator.
Outputs ready-to-paste TradingView settings.

---

## Architecture

```
nika_optimizer/
├── __init__.py          # Package init
├── __main__.py          # CLI entry point
├── run_optimizer.py     # Main orchestrator (7-step pipeline)
├── data_loader.py       # M10 OHLCV loader + MTF generation (1H, 4H)
├── signals.py           # Vectorized signal engine (T3, TSI, ADX, PSAR, RSI, ATR, Regime, Quality Score)
├── backtest_engine.py   # Numba @njit bar-by-bar backtest (trailing, break-even, PSAR exit, hard stop)
├── grid_search.py       # Two-phase multiprocessing sweep (24 cores)
├── walk_forward.py      # Walk-forward validation (VBT Pro or manual splits)
├── visualize.py         # Heatmaps, equity curves, robustness analysis
└── README.md            # This file
```

## Requirements

```
Python 3.12+
numpy
pandas
numba >= 0.64
matplotlib
```

Optional (recommended):
```
vectorbtpro >= 2026.3.1   # For VBT Pro walk-forward splits
```

Install:
```bash
pip install numpy pandas numba matplotlib
# VBT Pro: follow vectorbtpro.com install instructions
```

## Data Preparation

Export **M10 (10-minute) OHLCV bars** from MetaTrader 5:

1. Open MT5 → File → Open Data Folder
2. Use the History Center or a script to export XAUUSD M10 bars as CSV
3. The CSV should have columns: `<DATE>`, `<TIME>`, `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<TICKVOL>`
4. Tab-separated (default MT5 format)

Example filename: `XAUUSD_M10_202401_202603.csv`

## Usage

### Quick test (~1K combos, ~2 minutes)
```bash
python -m nika_optimizer --data "C:/MT5_Data/XAUUSD_M10.csv" --quick
```

### Medium sweep (~50K combos, ~30 minutes)
```bash
python -m nika_optimizer --data "C:/MT5_Data/XAUUSD_M10.csv" --mode medium
```

### Full sweep (~500K+ combos, ~2-6 hours on 24 cores)
```bash
python -m nika_optimizer --data "C:/MT5_Data/XAUUSD_M10.csv" --mode full
```

### All options
```bash
python -m nika_optimizer \
    --data "C:/MT5_Data/XAUUSD_M10.csv" \
    --mode full \
    --cores 24 \
    --output output \
    --sep "\t" \
    --spread 0.30 \
    --initial-equity 10000 \
    --risk-pct 1.0 \
    --wf-folds 8

# Resume from Phase 1 (skip signal sweep):
python -m nika_optimizer \
    --data "C:/MT5_Data/XAUUSD_M10.csv" \
    --skip-phase1 --phase1-file output/phase1_results.pkl

# Skip walk-forward (faster):
python -m nika_optimizer \
    --data "C:/MT5_Data/XAUUSD_M10.csv" --skip-wf
```

## Pipeline Steps

| Step | Phase | Description |
|------|-------|-------------|
| 1 | Data | Load M10 bars, create 1H and 4H MTF data |
| 2 | Phase 1 | Signal parameter sweep (T3 length, sensitivity, ADX, RSI, quality gates, regime) |
| 3 | Phase 2 | Trade management sweep (trailing stop, break-even, PSAR exit, hard stop) — uses best signals from Phase 1 |
| 4 | WF | Walk-forward validation: 6 months IS / 2 months OOS rolling windows |
| 5 | Equity | Final backtest with best params → equity curve + trade log |
| 6 | Charts | Heatmaps, robustness plots, WF summary, equity curve, trade distribution |
| 7 | Output | Best params formatted for TradingView + JSON export |

## Two-Phase Sweep Design

**Phase 1 — Signal Parameters** (the expensive part):
- T3 slow length: 15-100 (step 5)
- T3 sensitivity: 1, 2, 3, 4, 5, 7, 10
- T3 fast length: 3, 5, 8
- ADX length: 10, 14, 20
- ADX threshold: 15, 20, 25, 30
- RSI OB/OS: 70-90 / 10-30
- Min quality score: 40, 50, 60, 70
- Min trend strength: 0.0, 0.3, 0.6
- Regime thresholds (trending + volatile)

Each combo generates vectorized signals → runs through Numba backtest.

**Phase 2 — Trade Management** (fast, reuses signals):
- ATR hard stop: 0 (disabled), 2.0, 2.5, 3.0, 3.5, 4.0
- Trailing stop activation: 1.0-3.0 ATR
- Trailing stop offset: 0.3-1.0 ATR
- Break-even trigger: 1.0-3.0 ATR
- PSAR exit: ON/OFF + maturity settings
- Swing lookback: 5, 10, 15, 20 bars

## Ranking Formula

```
Rank = P^0.45 × S^0.30 × C^0.25
```

- **P (Performance)**: Profit Factor + Expectancy
- **S (Stability)**: Max Drawdown + Sharpe + Recovery Factor
- **C (Confidence)**: Trade count + Win Rate

Quality gates filter out noise:
- Minimum 100 trades
- PF >= 1.10
- Max DD <= 35%
- Net profit > 0

## Output

After completion, the `output/` directory contains:

```
output/
├── best_params_tradingview.txt   # Copy-paste settings for TradingView
├── best_params.json              # Machine-readable best params + metrics
├── phase1_results.pkl            # Full Phase 1 results (resumable)
├── phase2_results.pkl            # Full Phase 2 results
├── equity_curve.png              # Equity curve + drawdown
├── trade_distribution.png        # P&L histogram + holding period
├── walk_forward_summary.png      # WF fold performance
├── heatmap_*.png                 # Parameter heatmaps
└── robustness_*.png              # Parameter robustness plateaus
```

## Applying Results in TradingView

1. Open the `best_params_tradingview.txt` file
2. In TradingView, add the "Trend Duration Forecast + Optimizer V7 Pro" indicator
3. Open indicator Settings → Inputs tab
4. Set Mode = **Custom**
5. Enter each value from the "Quick-Paste" section at the bottom of the output file
6. The walk-forward validation results tell you how robust these settings are:
   - **ROBUST** (OOS/IS >= 0.60): High confidence — settings generalize well
   - **MODERATE** (0.40-0.60): Decent — monitor for degradation
   - **OVERFIT RISK** (< 0.40): Caution — consider using a wider T3 length plateau

## Tips

- **Start with `--quick`** to verify data loads correctly and the pipeline runs
- **Check robustness plots** — flat plateaus are better than sharp peaks
- **Walk-forward OOS/IS ratio > 0.50** means parameters generalize well
- **Phase 1 is resumable** — if it crashes, re-run with `--skip-phase1 --phase1-file output/phase1_results.pkl`
- **Spread matters** — $0.30 is typical for XAUUSD during London/NY sessions; increase to $0.50+ for off-hours testing
- If VBT Pro is installed, walk-forward will use its `Splitter` for cleaner splits

---

*Built by nikaquant — the best visual trading tool in the world for manual TradingView users.*

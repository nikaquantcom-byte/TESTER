# NikaQuant MEGA Optimizer V3 — FIXED

Drop-in replacement for the original `nika_optimizer` run that crashed at Phase 4.

## What was broken

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `run_mega_v3.py` | `WFConfig(n_splits=5, train_pct=0.70)` — wrong field names | New `WFConfig` dataclass with `n_splits`, `train_pct`, `min_trades` |
| 2 | `run_mega_v3.py` | `run_walk_forward(ohlcv, eng, conf, tp, cfg)` — V2 expected list of T3Params | New `run_walk_forward` accepts single `EngineConfig` tuple |
| 3 | `walk_forward_v2.py` | Imported `signals_v2`, `grid_search_v2` — V3 uses `signals_v3` | `walk_forward_v3.py` imports from `signals_v3` + `backtest_engine_v2` |
| 4 | `walk_forward_v2.py` | Result dict keys `oos_results` / `is_results` missing | Standardised return dict |

## How to run

```bash
# From C:\Desktop\10min\ (the folder containing both nika_optimizer and mega_v3_fixed)

# Full 4-phase run
python -m mega_v3_fixed --data nika_optimizer/XAUUSD_M10.csv --top 50

# Skip Phase 1 (reuse existing results)
python -m mega_v3_fixed --data nika_optimizer/XAUUSD_M10.csv --skip-phase1 --phase1-file output_v3/phase1_results.pkl --top 50

# Skip Phase 1 + Phase 3 (go straight to WF from Phase 2)
python -m mega_v3_fixed --data nika_optimizer/XAUUSD_M10.csv --skip-phase1 --phase1-file output_v3/phase1_results.pkl --skip-phase3

# Skip WF entirely (just Phases 1-3)
python -m mega_v3_fixed --data nika_optimizer/XAUUSD_M10.csv --skip-wf

# Custom WF settings
python -m mega_v3_fixed --data nika_optimizer/XAUUSD_M10.csv --wf-splits 5 --wf-train-pct 0.70 --wf-min-trades 10
```

## Pro tip — skip Phase 1 & 3 immediately

You already have `output_v3/phase1_results.pkl` and `output_v3/phase3_results.pkl`.
To jump straight to Phase 4 Walk-Forward using your existing Phase 3 results:

```bash
python -m mega_v3_fixed --data nika_optimizer/XAUUSD_M10.csv \
    --skip-phase1 --phase1-file output_v3/phase1_results.pkl \
    --skip-phase3 \
    --wf-top 5
```

But note: `--skip-phase3` means Phase 2 will re-run. If you also want to skip Phase 2,
you need to load `phase2_results.pkl` — support for `--phase2-file` can be added on request.

## Files

```
mega_v3_fixed/
  __init__.py              — package entry
  __main__.py              — python -m mega_v3_fixed entry point
  run_mega_v3_fixed.py     — main orchestrator (FIXED)
  walk_forward_v3.py       — WF engine (FIXED, V3 compatible)
  README.md                — this file
```

All other modules (`signals_v3`, `backtest_engine_v2`, `grid_search_v3`, `data_loader`)
are imported directly from `nika_optimizer/` — no copies needed.

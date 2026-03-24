# T3 Tournament — Phase 4: Walk-Forward Validation
# VectorBT Pro Splitter for IS/OOS with embargo
# Acceptance: OOS PF > 1.0 in 80%+ of folds, OOS/IS ratio > 0.5

import numpy as np
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .t3_engine import compute_t3_signals
from .t3_confluence import _apply_confluence_fast
from .t3_grid_search import compute_score
from nika_optimizer.backtest_engine_v2 import (
    run_backtest, make_trade_params, NUM_RESULTS,
    R_PROFIT_FACTOR, R_TOTAL_TRADES, R_NET_PROFIT,
    R_MAX_DRAWDOWN_PCT, R_SHARPE,
)
from nika_optimizer.signals_v3 import (
    atr_calc as atr_calc_v3,
    psar_calc, highest_n, lowest_n,
)
from .t3_engine import _ema, adx_di, rsi_calc, tsi_val


# ---------------------------------------------------------------------------
# Walk-forward parameters
# ---------------------------------------------------------------------------

WF_CONFIGS = [
    # (is_bars, oos_bars, embargo_bars)
    (20000, 5000,  250),   # ~4yr IS / ~1yr OOS
    (15000, 5000,  250),   # ~3yr IS / ~1yr OOS
    (10000, 3000,  150),   # ~2yr IS / ~6mo OOS
]
MIN_OOS_PF_PASS   = 1.0
MIN_FOLD_PASS_PCT = 0.80
MIN_OOS_IS_RATIO  = 0.50


# ---------------------------------------------------------------------------
# Single backtest on a bar slice
# ---------------------------------------------------------------------------

def _run_slice(ohlcv, start, end, t3_cfg, conf_cfg, mgmt_cfg):
    sl = slice(start, end)
    close  = ohlcv['close'][sl]
    open_  = ohlcv['open'][sl]
    high   = ohlcv['high'][sl]
    low    = ohlcv['low'][sl]
    volume = ohlcv['volume'][sl]
    n      = len(close)

    atr_v              = atr_calc_v3(high, low, close, 14)
    dip_v, dim_v, adxv = adx_di(high, low, close, 14)
    rsi_v              = rsi_calc(close, 14)
    tsi_v              = tsi_val(close, 5, 25, 14)
    psar_v, psar_bull  = psar_calc(high, low, 0.02, 0.02, 0.2)
    vol_sma            = _ema(volume.astype(np.float64), 20)

    pfu = np.zeros(n, dtype=np.bool_)
    pfd = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if  psar_bull[i] and not psar_bull[i-1]: pfu[i] = True
        if not psar_bull[i] and  psar_bull[i-1]: pfd[i] = True

    ont = np.empty(n, dtype=np.float64)
    ont[:-1] = open_[1:]; ont[-1] = close[-1]
    swh = highest_n(high, 10)
    swl = lowest_n(low,  10)

    buy_raw, sell_raw, _ = compute_t3_signals(
        t3_cfg, open_, high, low, close, volume
    )

    if conf_cfg is not None:
        buy, sell = _apply_confluence_fast(
            buy_raw, sell_raw, conf_cfg,
            adxv, dip_v, dim_v, rsi_v, tsi_v,
            psar_bull, volume.astype(np.float64), vol_sma,
        )
    else:
        buy, sell = buy_raw, sell_raw

    t3_trend = np.zeros(n, dtype=np.int32)
    cur = 0
    for i in range(n):
        if buy[i]:    cur =  1
        elif sell[i]: cur = -1
        t3_trend[i] = cur

    dummy = np.zeros(n, dtype=np.int16)

    if mgmt_cfg is not None:
        tp_params = make_trade_params(
            spread_points  = 0.30,
            use_open_entry = 1.0,
            atr_sl_mult    = mgmt_cfg.atr_sl_mult,
            atr_tp_mult    = mgmt_cfg.atr_tp_mult,
            trail_mult     = mgmt_cfg.trail_mult,
            be_trigger_r   = mgmt_cfg.be_trigger,
            max_bars_held  = mgmt_cfg.maturity,
        )
    else:
        tp_params = make_trade_params(spread_points=0.30, use_open_entry=1.0)

    return run_backtest(
        open_, high, low, close, ont,
        t3_trend, buy, sell, atr_v,
        dummy, dummy, 0, 0,
        pfu, pfd, swh, swl,
        tp_params, 0, -1,
    )


# ---------------------------------------------------------------------------
# Walk-forward for one strategy config
# ---------------------------------------------------------------------------

def walk_forward_one(ohlcv, t3_cfg, conf_cfg, mgmt_cfg, label=''):
    total_bars = len(ohlcv['close'])
    all_results = []

    for (is_bars, oos_bars, emb) in WF_CONFIGS:
        step     = oos_bars + emb
        is_start = 0
        fold     = 0
        oos_pfs  = []
        is_pfs   = []

        while is_start + is_bars + emb + oos_bars <= total_bars:
            is_end   = is_start + is_bars
            oos_start= is_end   + emb
            oos_end  = oos_start+ oos_bars

            res_is  = _run_slice(ohlcv, is_start,  is_end,   t3_cfg, conf_cfg, mgmt_cfg)
            res_oos = _run_slice(ohlcv, oos_start, oos_end,  t3_cfg, conf_cfg, mgmt_cfg)

            is_pf  = float(res_is[R_PROFIT_FACTOR])
            oos_pf = float(res_oos[R_PROFIT_FACTOR])
            oos_pfs.append(oos_pf)
            is_pfs.append(is_pf)
            fold += 1
            is_start += step

        if fold == 0:
            continue

        pass_rate    = sum(1 for x in oos_pfs if x > MIN_OOS_PF_PASS) / fold
        avg_oos_pf   = np.mean(oos_pfs)
        avg_is_pf    = np.mean(is_pfs)
        oos_is_ratio = avg_oos_pf / avg_is_pf if avg_is_pf > 0 else 0.0
        passed       = (pass_rate >= MIN_FOLD_PASS_PCT and oos_is_ratio >= MIN_OOS_IS_RATIO)

        all_results.append({
            'is_bars':      is_bars,
            'oos_bars':     oos_bars,
            'folds':        fold,
            'pass_rate':    round(pass_rate, 3),
            'avg_oos_pf':   round(avg_oos_pf, 3),
            'avg_is_pf':    round(avg_is_pf, 3),
            'oos_is_ratio': round(oos_is_ratio, 3),
            'passed':       passed,
            'oos_pfs':      oos_pfs,
        })

    return all_results


# ---------------------------------------------------------------------------
# Run walk-forward on top N Phase 3 results
# ---------------------------------------------------------------------------

def run_walkforward(ohlcv, top_phase3_results, n_top=50):
    top = top_phase3_results[:n_top]
    total = len(top)

    print(f"\n{'='*80}")
    print(f"  PHASE 4 — WALK-FORWARD VALIDATION")
    print(f"  Validating top {total} strategies across {len(WF_CONFIGS)} WF configs")
    print(f"  Acceptance: OOS PF>{MIN_OOS_PF_PASS} in {MIN_FOLD_PASS_PCT*100:.0f}%+ folds, "
          f"OOS/IS>{MIN_OOS_IS_RATIO}")
    print(f"{'='*80}")

    validated = []
    t0 = time.time()

    for i, entry in enumerate(top):
        if len(entry) == 5:
            mgmt_cfg, conf_cfg, t3_cfg, res_full, score = entry
        elif len(entry) == 4:
            conf_cfg, t3_cfg, res_full, score = entry
            mgmt_cfg = None
        else:
            t3_cfg, res_full, score = entry
            conf_cfg = mgmt_cfg = None

        label = f"#{i+1} score={score:.1f}"
        wf_results = walk_forward_one(ohlcv, t3_cfg, conf_cfg, mgmt_cfg, label=label)

        n_configs   = len(wf_results)
        n_passed    = sum(1 for r in wf_results if r['passed'])
        all_passed  = n_passed == n_configs and n_configs > 0
        any_passed  = n_passed > 0

        status = '✅ PASS' if all_passed else ('⚠️  PARTIAL' if any_passed else '❌ FAIL')
        avg_oos = np.mean([r['avg_oos_pf'] for r in wf_results]) if wf_results else 0
        avg_rat = np.mean([r['oos_is_ratio'] for r in wf_results]) if wf_results else 0

        print(f"  [{i+1:>3}/{total}] {status} | IS score={score:.1f} | "
              f"OOS PF={avg_oos:.2f} | OOS/IS={avg_rat:.2f} | "
              f"Passed {n_passed}/{n_configs} WF configs", flush=True)

        validated.append({
            'rank':        i + 1,
            'score':       score,
            'wf_results':  wf_results,
            'all_passed':  all_passed,
            'any_passed':  any_passed,
            'avg_oos_pf':  round(avg_oos, 3),
            'oos_is_ratio':round(avg_rat, 3),
            'entry':       entry,
        })

    elapsed = time.time() - t0
    survivors = [v for v in validated if v['all_passed']]
    partials  = [v for v in validated if v['any_passed'] and not v['all_passed']]

    print(f"\n  Walk-forward done: {elapsed:.1f}s")
    print(f"  Full passes:    {len(survivors)}/{total}")
    print(f"  Partial passes: {len(partials)}/{total}")
    print(f"  Rejected:       {total - len(survivors) - len(partials)}/{total}")

    return validated, survivors

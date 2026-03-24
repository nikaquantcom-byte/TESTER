# T3 Tournament — Phase 3: Management Sweep
# ATR SL/TP, trailing stop, break-even, maturity cutoff
# Runs on top Phase 2 (confluence) results

import numpy as np
import math
import time
from multiprocessing import Pool, cpu_count
from typing import NamedTuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .t3_engine import compute_t3_signals
from .t3_confluence import _apply_confluence_fast
from .t3_grid_search import compute_score
from nika_optimizer.backtest_engine_v2 import (
    run_backtest, make_trade_params, NUM_RESULTS,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT,
    R_TOTAL_TRADES, R_NET_PROFIT, R_EXPECTANCY,
    R_SHARPE, R_RECOVERY_FACTOR, R_CALMAR,
    R_AVG_BARS_HELD, R_MAX_CONSEC_LOSS,
)
from nika_optimizer.signals_v3 import (
    atr_calc as atr_calc_v3,
    psar_calc, highest_n, lowest_n,
)


# ---------------------------------------------------------------------------
# Management config
# ---------------------------------------------------------------------------

class MgmtConfig(NamedTuple):
    phase2_idx:     int
    atr_sl_mult:    float   # 0.0 = no hard SL
    atr_tp_mult:    float   # 0.0 = no TP
    trail_mult:     float   # 0.0 = no trailing
    be_trigger:     float   # R-multiple to activate BE (0.0 = OFF)
    maturity:       int     # max bars to hold (0 = OFF)


ATR_SL_MULTS  = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
ATR_TP_MULTS  = [0.0, 1.5, 2.0, 3.0, 4.0, 5.0]
TRAIL_MULTS   = [0.0, 1.0, 1.5, 2.0, 3.0]
BE_TRIGGERS   = [0.0, 0.5, 1.0, 1.5]
MATURITIES    = [0, 20, 50, 100, 200]


def generate_management_combos(n_top_phase2):
    combos = []
    for idx in range(n_top_phase2):
        for sl in ATR_SL_MULTS:
            for tp in ATR_TP_MULTS:
                if tp > 0 and sl == 0: continue  # TP without SL is undefined
                for tr in TRAIL_MULTS:
                    for be in BE_TRIGGERS:
                        if be > 0 and sl == 0: continue  # BE needs reference SL
                        for mat in MATURITY:
                            combos.append(MgmtConfig(
                                phase2_idx=idx,
                                atr_sl_mult=sl,
                                atr_tp_mult=tp,
                                trail_mult=tr,
                                be_trigger=be,
                                maturity=mat,
                            ))
    return combos


_shared_m = {}

def _init_worker_m(data):
    global _shared_m
    _shared_m = data


def _worker_m(cfg):
    d = _shared_m
    try:
        conf_cfg, t3_cfg, _, _ = d['top_results'][cfg.phase2_idx]

        buy_raw, sell_raw, _ = compute_t3_signals(
            t3_cfg, d['open'], d['high'], d['low'], d['close'], d['volume']
        )

        buy, sell = _apply_confluence_fast(
            buy_raw, sell_raw, conf_cfg,
            d['adx'], d['dip'], d['dim'],
            d['rsi'], d['tsi'],
            d['psar_bull'], d['volume'], d['vol_sma'],
        )

        n = len(d['close'])
        t3_trend = np.zeros(n, dtype=np.int32)
        cur = 0
        for i in range(n):
            if buy[i]:    cur =  1
            elif sell[i]: cur = -1
            t3_trend[i] = cur

        dummy = np.zeros(n, dtype=np.int16)

        tp_params = make_trade_params(
            spread_points  = 0.30,
            use_open_entry = 1.0,
            atr_sl_mult    = cfg.atr_sl_mult,
            atr_tp_mult    = cfg.atr_tp_mult,
            trail_mult     = cfg.trail_mult,
            be_trigger_r   = cfg.be_trigger,
            max_bars_held  = cfg.maturity,
        )

        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'], d['open_next'],
            t3_trend, buy, sell, d['atr'],
            dummy, dummy, 0, 0,
            d['psar_flip_up'], d['psar_flip_down'],
            d['swing_high'], d['swing_low'],
            tp_params, 0, -1,
        )
        return (cfg, conf_cfg, t3_cfg, res, compute_score(res))
    except Exception:
        return (cfg, None, None, np.zeros(NUM_RESULTS, dtype=np.float64), -999.0)


def run_management_sweep(ohlcv, top_phase2_results, n_top=100, n_cores=None):
    if n_cores is None:
        n_cores = cpu_count()

    top     = top_phase2_results[:n_top]
    combos  = generate_management_combos(len(top))
    total   = len(combos)

    print(f"\n{'='*80}")
    print(f"  PHASE 3 — MANAGEMENT SWEEP")
    print(f"  Top {n_top} Phase 2 configs × management params = {total:,} combos")
    print(f"  Cores: {n_cores}")
    print(f"{'='*80}")

    close   = ohlcv['close']
    open_   = ohlcv['open']
    high    = ohlcv['high']
    low     = ohlcv['low']
    volume  = ohlcv['volume']
    n       = len(close)

    from .t3_engine import _ema, adx_di, rsi_calc, tsi_val
    from nika_optimizer.signals_v3 import atr_calc as atr_calc_v3

    atr_v                 = atr_calc_v3(high, low, close, 14)
    dip_v, dim_v, adx_v  = adx_di(high, low, close, 14)
    rsi_v                 = rsi_calc(close, 14)
    tsi_v                 = tsi_val(close, 5, 25, 14)
    psar_v, psar_bull     = psar_calc(high, low, 0.02, 0.02, 0.2)
    vol_sma               = _ema(volume.astype(np.float64), 20)

    pfu = np.zeros(n, dtype=np.bool_)
    pfd = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if  psar_bull[i] and not psar_bull[i-1]: pfu[i] = True
        if not psar_bull[i] and  psar_bull[i-1]: pfd[i] = True

    ont = np.empty(n, dtype=np.float64)
    ont[:-1] = open_[1:]; ont[-1] = close[-1]
    swh = highest_n(high, 10)
    swl = lowest_n(low,  10)

    shared = {
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
        'atr': atr_v, 'adx': adx_v, 'dip': dip_v, 'dim': dim_v,
        'rsi': rsi_v, 'tsi': tsi_v,
        'psar_bull': psar_bull,
        'psar_flip_up': pfu, 'psar_flip_down': pfd,
        'vol_sma': vol_sma,
        'swing_high': swh, 'swing_low': swl,
        'open_next': ont,
        'top_results': top,
    }

    t0 = time.time()
    results = []
    pi = max(100, total // 200)
    last_p = t0

    with Pool(n_cores, initializer=_init_worker_m, initargs=(shared,)) as pool:
        chunk = max(1, total // (n_cores * 10))
        for i, r in enumerate(pool.imap_unordered(_worker_m, combos, chunksize=chunk)):
            results.append(r)
            now = time.time()
            if (i+1) % pi == 0 or (now - last_p) > 15:
                last_p = now
                pct  = (i+1)/total*100
                rate = (i+1)/(now-t0)
                best = max(x[4] for x in results)
                bar  = '\u2588'*int(pct//5) + '\u2591'*(20-int(pct//5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] "
                      f"{rate:.0f}/s | Best: {best:.1f}", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[4], reverse=True)
    print(f"\n  Phase 3 done: {elapsed:.1f}s | Best: {results[0][4]:.1f}")
    return results

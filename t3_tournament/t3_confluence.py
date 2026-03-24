# T3 Tournament — Phase 2: Confluence Filter Sweep
# Takes top T3 configs from Phase 1 and sweeps confluence filters on top
# Always includes OFF baseline for every filter (data decides)

import numpy as np
import time
from multiprocessing import Pool, cpu_count
from typing import NamedTuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .t3_engine import (
    _ema, _t3, atr_calc, rsi_calc, tsi_val, adx_di,
    compute_t3_signals,
    STRUCT_OF_IND,
)
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
from .t3_grid_search import compute_score


# ---------------------------------------------------------------------------
# Confluence config
# ---------------------------------------------------------------------------

class ConfluenceConfig(NamedTuple):
    t3_idx:      int    # index into top T3 results list
    adx_thresh:  float  # 0 = OFF
    rsi_zone:    int    # 0=OFF 1=30/70 2=40/60
    tsi_dir:     int    # 0=OFF 1=ON
    psar_agree:  int    # 0=OFF 1=ON
    di_diff:     int    # 0=OFF 1=ON
    vol_filter:  int    # 0=OFF 1=ON
    n_of_m:      int    # how many active filters must agree


ADX_THRESHOLDS = [0.0, 20.0, 25.0, 30.0]
RSI_ZONES      = [0, 1, 2]
TSI_DIRS       = [0, 1]
PSAR_AGREES    = [0, 1]
DI_DIFFS       = [0, 1]
VOL_FILTERS    = [0, 1]
N_OF_M_VALUES  = [1, 2, 3]


def generate_confluence_combos(n_top_t3):
    combos = []
    for idx in range(n_top_t3):
        for adx in ADX_THRESHOLDS:
            for rsi in RSI_ZONES:
                for tsi in TSI_DIRS:
                    for psar in PSAR_AGREES:
                        for di in DI_DIFFS:
                            for vol in VOL_FILTERS:
                                n_active = sum([
                                    adx > 0, rsi > 0, tsi > 0,
                                    psar > 0, di > 0, vol > 0
                                ])
                                for nom in N_OF_M_VALUES:
                                    if nom > max(n_active, 1):
                                        continue
                                    combos.append(ConfluenceConfig(
                                        t3_idx=idx,
                                        adx_thresh=adx,
                                        rsi_zone=rsi,
                                        tsi_dir=tsi,
                                        psar_agree=psar,
                                        di_diff=di,
                                        vol_filter=vol,
                                        n_of_m=nom,
                                    ))
    return combos


# ---------------------------------------------------------------------------
# Confluence filter (pure Python — fast enough, called per-signal-bar only)
# ---------------------------------------------------------------------------

def _apply_confluence_fast(buy_raw, sell_raw, cfg,
                            adx_v, dip_v, dim_v, rsi_v, tsi_v,
                            psar_bull, vol_v, vol_sma):
    n    = len(buy_raw)
    buy  = buy_raw.copy()
    sell = sell_raw.copy()
    for i in range(n):
        if not buy[i] and not sell[i]:
            continue
        vb = 0; vs = 0; na = 0
        if cfg.adx_thresh > 0:
            na += 1
            if adx_v[i] >= cfg.adx_thresh: vb += 1; vs += 1
        if cfg.rsi_zone > 0:
            na += 1
            lo = 30.0 if cfg.rsi_zone == 1 else 40.0
            hi = 70.0 if cfg.rsi_zone == 1 else 60.0
            if rsi_v[i] < hi: vb += 1
            if rsi_v[i] > lo: vs += 1
        if cfg.tsi_dir > 0:
            na += 1
            if tsi_v[i] > 0.0: vb += 1
            if tsi_v[i] < 0.0: vs += 1
        if cfg.psar_agree > 0:
            na += 1
            if psar_bull[i]:     vb += 1
            if not psar_bull[i]: vs += 1
        if cfg.di_diff > 0:
            na += 1
            if dip_v[i] > dim_v[i]: vb += 1
            if dim_v[i] > dip_v[i]: vs += 1
        if cfg.vol_filter > 0:
            na += 1
            if vol_v[i] > vol_sma[i]: vb += 1; vs += 1
        if na > 0:
            if buy[i]  and vb < cfg.n_of_m: buy[i]  = False
            if sell[i] and vs < cfg.n_of_m: sell[i] = False
    return buy, sell


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_shared_c = {}

def _init_worker_c(data):
    global _shared_c
    _shared_c = data


def _worker_c(cfg):
    d = _shared_c
    try:
        t3_cfg, t3_res, _ = d['top_results'][cfg.t3_idx]
        buy_raw, sell_raw, _ = compute_t3_signals(
            t3_cfg, d['open'], d['high'], d['low'], d['close'], d['volume']
        )
        n = len(d['close'])

        buy, sell = _apply_confluence_fast(
            buy_raw, sell_raw, cfg,
            d['adx'], d['dip'], d['dim'],
            d['rsi'], d['tsi'],
            d['psar_bull'], d['volume'], d['vol_sma'],
        )

        t3_trend = np.zeros(n, dtype=np.int32)
        cur = 0
        for i in range(n):
            if buy[i]:    cur =  1
            elif sell[i]: cur = -1
            t3_trend[i] = cur

        dummy = np.zeros(n, dtype=np.int16)
        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'], d['open_next'],
            t3_trend, buy, sell, d['atr'],
            dummy, dummy, 0, 0,
            d['psar_flip_up'], d['psar_flip_down'],
            d['swing_high'], d['swing_low'],
            d['tp'], 0, -1,
        )
        return (cfg, t3_cfg, res, compute_score(res))
    except Exception:
        return (cfg, None, np.zeros(NUM_RESULTS, dtype=np.float64), -999.0)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_confluence_sweep(ohlcv, top_t3_results, n_top=200, n_cores=None):
    if n_cores is None:
        n_cores = cpu_count()

    top    = top_t3_results[:n_top]
    combos = generate_confluence_combos(len(top))
    total  = len(combos)

    print(f"\n{'='*80}")
    print(f"  PHASE 2 — CONFLUENCE SWEEP")
    print(f"  Top {n_top} T3 configs × confluence filters = {total:,} combos")
    print(f"  Cores: {n_cores}")
    print(f"{'='*80}")

    close  = ohlcv['close']
    open_  = ohlcv['open']
    high   = ohlcv['high']
    low    = ohlcv['low']
    volume = ohlcv['volume']
    n      = len(close)

    atr_v              = atr_calc_v3(high, low, close, 14)
    dip_v, dim_v, adx_v = adx_di(high, low, close, 14)
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

    tp = make_trade_params(spread_points=0.30, use_open_entry=1.0)

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
        'tp': tp,
    }

    t0 = time.time()
    results = []
    pi = max(100, total // 200)
    last_p = t0

    with Pool(n_cores, initializer=_init_worker_c, initargs=(shared,)) as pool:
        chunk = max(1, total // (n_cores * 10))
        for i, r in enumerate(pool.imap_unordered(_worker_c, combos, chunksize=chunk)):
            results.append(r)
            now = time.time()
            if (i+1) % pi == 0 or (now - last_p) > 15:
                last_p = now
                pct  = (i+1) / total * 100
                rate = (i+1) / (now - t0)
                best = max(x[3] for x in results)
                bar  = '\u2588'*int(pct//5) + '\u2591'*(20-int(pct//5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] "
                      f"{rate:.0f}/s | Best: {best:.1f}", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Phase 2 done: {elapsed:.1f}s | Best: {results[0][3]:.1f}")
    return results

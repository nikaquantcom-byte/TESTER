"""
Module 3 — Universal Grid Search V3
Phase 1: ALL trigger types compete (MA types, TSI, ADX, PSAR, RSI, LinReg, Volume)
Phase 2: Confluence sweep on winners
Phase 3: Trade management sweep
Phase 4: Walk-forward validation

FIX v3.1: Shared indicators (ATR, TSI, ADX, PSAR, RSI) precomputed ONCE per run.
          Passed into generate_universal_signals() via shared= param.
          Per-combo speedup: ~117ms → ~3ms.
FIX v3.2: Numba warm-up in main process before Pool spawn. Fixes hang on Windows.
"""

import numpy as np
import math
import time
import pickle
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple

from .signals_v3 import (
    EngineConfig, ConfluenceConfig, SignalResult,
    generate_universal_signals, pack_indicator_bits, apply_confluence,
    precompute_shared_indicators, SharedIndicators,
    NUM_MA_TYPES, MA_NAMES, NUM_TRIGGER_TYPES, TRIG_NAMES,
    TRIG_MA, TRIG_TSI, TRIG_ADX, TRIG_PSAR, TRIG_RSI, TRIG_VOL,
    TRIG_LINREG, TRIG_LINREG_MA,
    MA_SMA, MA_EMA, MA_WMA, MA_VWMA, MA_RMA, MA_DEMA, MA_TEMA,
    MA_ZLEMA, MA_HMA, MA_DONCHIAN, MA_T3,
    MODE_FLIP, MODE_SLOPE, MODE_PRICE_CROSS, MODE_CROSSOVER, MODE_ZERO_CROSS,
    NUM_IND, IND_NAMES,
    _ema, _rma, _sma, _wma, _dema, _tema, _zlema, _hma, _donchian, _t3,
    heiken_ashi, atr_calc, rsi_calc, tsi_calc, adx_calc, psar_calc,
    highest_n, lowest_n, compute_signal_mode, compute_crossover_signal,
    _mtf_slope_signals, nika_linreg,
)
from .backtest_engine_v2 import (
    run_backtest, make_trade_params, NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_NET_PROFIT, R_EXPECTANCY, R_SHARPE, R_CALMAR, R_FINAL_EQUITY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Grids
# ─────────────────────────────────────────────────────────────────────────────

MA_LENGTHS = list(range(5, 501, 5))
T_FACTORS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SENSITIVITIES = list(range(1, 31))
MA_MODES = [MODE_FLIP, MODE_SLOPE, MODE_PRICE_CROSS, MODE_CROSSOVER]
OSCILLATOR_MODES = [MODE_ZERO_CROSS, MODE_FLIP]
HA_OPTIONS = [0, 1]
FAST_RATIOS = [0.10, 0.50]
LINREG_LENGTHS = [5, 8, 13, 21, 34, 55]
LINREG_MA_PERIODS = [3, 5, 8, 13]
TSI_PARAMS = [(5, 25, 14), (5, 13, 8), (8, 30, 14), (3, 20, 10)]
ADX_LENS = [7, 10, 14, 20, 25]
RSI_PERIODS = [7, 10, 14, 21]
PSAR_PARAMS = [(0.01, 0.01, 0.1), (0.02, 0.02, 0.2), (0.03, 0.03, 0.3)]
VOL_PARAMS = [(10, 1.5), (20, 1.5), (20, 2.0), (50, 2.0)]
BE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
TRAIL_ACT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999.0]
TRAIL_OFF = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
HARD_STOP = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
PSAR_EXIT = [0.0, 1.0]
SPREAD_GRID = [0.10, 0.20, 0.30, 0.50, 1.00]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(results):
    trades = results[R_TOTAL_TRADES]
    if trades < 5: return -999.0
    pf = results[R_PROFIT_FACTOR]
    pf_s = min(max((pf - 1.0) / 2.0 * 100.0, 0), 100.0)
    exp = results[R_EXPECTANCY]
    exp_s = min(max(exp, 0) / 2.0 * 100.0, 100.0)
    perf = max(0.6 * pf_s + 0.4 * exp_s, 0.01)
    dd = abs(results[R_MAX_DRAWDOWN_PCT])
    dd_s = max(0, 100.0 - dd * 2.5)
    sharpe = min(max(results[R_SHARPE], 0), 3.0)
    sh_s = sharpe / 3.0 * 100.0
    stab = max(0.5 * dd_s + 0.5 * sh_s, 0.01)
    tc_s = min(math.log(max(trades, 10)) / math.log(300) * 100.0, 100.0)
    wr = results[R_WIN_RATE]
    wr_s = min(wr / 55.0 * 100.0, 100.0)
    conf = max(0.6 * tc_s + 0.4 * wr_s, 0.01)
    return (perf ** 0.45) * (stab ** 0.30) * (conf ** 0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Combo generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_phase1_combos(quick=False):
    combos = []
    if quick:
        lengths = [20, 50, 80, 100, 150, 200]
        tfactors = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        senses = [1, 3, 5, 10, 15, 20]
    else:
        lengths = MA_LENGTHS; tfactors = T_FACTORS; senses = SENSITIVITIES

    for ma_type in range(NUM_MA_TYPES):
        tfs = tfactors if ma_type == MA_T3 else [0.7]
        for length in lengths:
            for tf in tfs:
                for sens in senses:
                    if sens >= length: continue
                    for mode in [MODE_FLIP, MODE_SLOPE, MODE_PRICE_CROSS]:
                        for ha in HA_OPTIONS:
                            combos.append(EngineConfig(
                                trigger_type=TRIG_MA, ma_type=ma_type,
                                ma_length=length, tfactor=tf, sensitivity=sens,
                                signal_mode=mode, use_heiken_ashi=ha,
                            ))
                    for ha in HA_OPTIONS:
                        fast_len = max(2, int(length * 0.3))
                        if fast_len < length:
                            combos.append(EngineConfig(
                                trigger_type=TRIG_MA, ma_type=ma_type,
                                ma_length=length, ma_fast_length=fast_len,
                                tfactor=tf, sensitivity=sens,
                                signal_mode=MODE_CROSSOVER, use_heiken_ashi=ha,
                            ))

    for tsi_s, tsi_l, tsi_sig in TSI_PARAMS:
        for ha in HA_OPTIONS:
            combos.append(EngineConfig(trigger_type=TRIG_TSI, tsi_short=tsi_s, tsi_long=tsi_l, tsi_signal=tsi_sig, use_heiken_ashi=ha))

    for adx_l in ADX_LENS:
        for ha in HA_OPTIONS:
            combos.append(EngineConfig(trigger_type=TRIG_ADX, adx_len=adx_l, use_heiken_ashi=ha))

    for ps, pi, pm in PSAR_PARAMS:
        for ha in HA_OPTIONS:
            combos.append(EngineConfig(trigger_type=TRIG_PSAR, psar_start=ps, psar_inc=pi, psar_max=pm, use_heiken_ashi=ha))

    for rp in RSI_PERIODS:
        for ha in HA_OPTIONS:
            combos.append(EngineConfig(trigger_type=TRIG_RSI, rsi_period=rp, use_heiken_ashi=ha))

    for vl, vm in VOL_PARAMS:
        for ha in HA_OPTIONS:
            combos.append(EngineConfig(trigger_type=TRIG_VOL, vol_sma_len=vl, vol_mult=vm, use_heiken_ashi=ha))

    for lr_len in LINREG_LENGTHS:
        for ha in HA_OPTIONS:
            combos.append(EngineConfig(trigger_type=TRIG_LINREG, linreg_length=lr_len, use_heiken_ashi=ha))
            for lr_ma in range(NUM_MA_TYPES):
                for lr_per in LINREG_MA_PERIODS:
                    combos.append(EngineConfig(
                        trigger_type=TRIG_LINREG_MA, linreg_length=lr_len,
                        linreg_ma_type=lr_ma, linreg_ma_period=lr_per,
                        use_heiken_ashi=ha,
                    ))
    return combos


# ─────────────────────────────────────────────────────────────────────────────
# Numba Warm-Up — compile all @njit functions ONCE in main process
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_numba():
    """Force-compile all Numba functions with tiny arrays. Call before Pool."""
    t = np.zeros(100, dtype=np.float64)
    h = np.ones(100, dtype=np.float64) * 2.0
    l = np.ones(100, dtype=np.float64) * 0.5
    _ema(t, 5); _rma(t, 5); _sma(t, 5); _wma(t, 5)
    _dema(t, 5); _tema(t, 5); _zlema(t, 5); _hma(t, 5)
    _donchian(t, 5); _t3(t, 5, 0.7)
    heiken_ashi(t, h, l, t)
    atr_calc(h, l, t, 5)
    rsi_calc(t + 1.0, 5)
    tsi_calc(t + 1.0, 5, 13, 8)
    adx_calc(h, l, t + 1.0, 5)
    psar_calc(h, l, 0.02, 0.02, 0.2)
    highest_n(t, 5); lowest_n(t, 5)
    compute_signal_mode(t, t, 3, 0)
    compute_crossover_signal(t, t + 0.1)
    _mtf_slope_signals(t, 3, 100)
    nika_linreg(t + 1.0, t + 1.0, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Workers — FIX: shared indicators passed via _shared dict, not recomputed
# ─────────────────────────────────────────────────────────────────────────────

_shared = {}

def _init_worker(data):
    global _shared
    _shared = data

def _worker_phase1(args):
    engine, tp, start_idx, end_idx = args
    d = _shared
    try:
        shared = d.get('shared_indicators')
        sig = generate_universal_signals(
            d['open'], d['high'], d['low'], d['close'], d['volume'],
            engine, d.get('htf_1h'), d.get('htf_4h'),
            shared=shared,
        )
        n = len(d['close'])
        t3_trend = np.zeros(n, dtype=np.int32)
        current = 0
        for i in range(n):
            if sig.buy[i]: current = 1
            elif sig.sell[i]: current = -1
            t3_trend[i] = current
        ibb = pack_indicator_bits(sig.ind_buy)
        ibs = pack_indicator_bits(sig.ind_sell)
        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'], sig.open_next,
            t3_trend, sig.buy, sig.sell, sig.atr,
            ibb, ibs, 0, 0,
            sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
            tp, start_idx, end_idx,
        )
        return (engine, res, compute_score(res))
    except Exception as e:
        return (engine, np.zeros(NUM_RESULTS, dtype=np.float64), -999.0)


def _worker_phase2(args):
    engine, conf, tp, start_idx, end_idx = args
    d = _shared
    try:
        shared = d.get('shared_indicators')
        sig = generate_universal_signals(
            d['open'], d['high'], d['low'], d['close'], d['volume'],
            engine, d.get('htf_1h'), d.get('htf_4h'),
            shared=shared,
        )
        buy_f, sell_f = apply_confluence(sig.buy, sig.sell, sig.ind_buy, sig.ind_sell, conf)
        n = len(d['close'])
        t3_trend = np.zeros(n, dtype=np.int32)
        current = 0
        for i in range(n):
            if buy_f[i]: current = 1
            elif sell_f[i]: current = -1
            t3_trend[i] = current
        ibb = pack_indicator_bits(sig.ind_buy)
        ibs = pack_indicator_bits(sig.ind_sell)
        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'], sig.open_next,
            t3_trend, buy_f, sell_f, sig.atr,
            ibb, ibs, 0, 0,
            sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
            tp, start_idx, end_idx,
        )
        return (engine, conf, res, compute_score(res))
    except Exception:
        return (engine, conf, np.zeros(NUM_RESULTS, dtype=np.float64), -999.0)


# ─────────────────────────────────────────────────────────────────────────────
# Phase runners — FIX v3.2: Numba warm-up before Pool starts
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(ohlcv, n_cores=None, quick=False, start_idx=0, end_idx=-1):
    if n_cores is None: n_cores = cpu_count()
    combos = generate_phase1_combos(quick=quick)
    total = len(combos)

    by_type = {}
    for c in combos:
        name = TRIG_NAMES[c.trigger_type]
        if c.trigger_type == TRIG_MA: name = f"MA:{MA_NAMES[c.ma_type]}"
        by_type[name] = by_type.get(name, 0) + 1

    print(f"\n[Phase 1] Universal Signal Tournament — {total:,} combos × {n_cores} cores")
    for name, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {name:>15}: {count:,}")

    base_engine = EngineConfig()
    print("  Precomputing shared indicators (ATR, TSI, ADX, PSAR, RSI)...", flush=True)
    shared = precompute_shared_indicators(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'],
        base_engine
    )
    ohlcv_with_shared = dict(ohlcv)
    ohlcv_with_shared['shared_indicators'] = shared
    print("  Done. Shared indicators will be reused across all combos.", flush=True)

    tp = make_trade_params(spread_points=0.30, use_open_entry=1.0)
    args = [(eng, tp, start_idx, end_idx) for eng in combos]

    # FIX v3.2: Warm up Numba + run 1 test combo BEFORE spawning Pool
    print("  Warming up Numba JIT (one-time compile)...", flush=True)
    _warmup_numba()
    _init_worker(ohlcv_with_shared)
    _worker_phase1((combos[0], tp, start_idx, end_idx))
    print("  Numba ready. All functions compiled.", flush=True)

    t0 = time.time()
    results = []
    pi = max(50, total // 100); last_p = t0
    print(f"  Starting...", flush=True)

    with Pool(n_cores, initializer=_init_worker, initargs=(ohlcv_with_shared,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_phase1, args, chunksize=max(1, total // (n_cores * 8)))):
            results.append(r)
            now = time.time()
            if (i + 1) % pi == 0 or (now - last_p > 10):
                last_p = now
                pct = (i+1)/total*100; rate = (i+1)/(now-t0)
                best = max(r[2] for r in results)
                bar = '\u2588' * int(pct // 5) + '\u2591' * (20 - int(pct // 5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] {rate:.0f}/s | ETA {(total-i-1)/max(rate,0.01):.0f}s | Best: {best:.1f}", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n[Phase 1] Done: {elapsed:.1f}s | Best: {results[0][2]:.1f}")

    print(f"\n  {'Trigger':>15} | {'Score':>6} | {'PF':>6} | {'Trades':>6} | {'WR%':>5} | Config")
    print(f"  {'-'*80}")
    seen_types = set()
    for eng, res, score in results:
        tname = TRIG_NAMES[eng.trigger_type]
        if eng.trigger_type == TRIG_MA: tname = f"MA:{MA_NAMES[eng.ma_type]}"
        if tname in seen_types: continue
        seen_types.add(tname)
        print(f"  {tname:>15} | {score:6.1f} | {res[R_PROFIT_FACTOR]:6.2f} | {int(res[R_TOTAL_TRADES]):>6} | {res[R_WIN_RATE]:5.1f} | "
              f"L={eng.ma_length} TF={eng.tfactor:.2f} S={eng.sensitivity} HA={eng.use_heiken_ashi}")
    return results


def run_phase2(ohlcv, top_engines, n_cores=None, start_idx=0, end_idx=-1):
    if n_cores is None: n_cores = cpu_count()

    confs = [ConfluenceConfig(indicator_mask=0, min_agree=0, mtf_block=0)]
    for mask in range(1, 256):
        active = bin(mask).count('1')
        for min_ag in range(1, active + 1):
            confs.append(ConfluenceConfig(indicator_mask=mask, min_agree=min_ag, mtf_block=0))
    seen = set(); unique = []
    for c in confs:
        key = (c.indicator_mask, c.min_agree)
        if key not in seen: seen.add(key); unique.append(c)
    confs = unique

    base_engine = EngineConfig()
    print("  Precomputing shared indicators...", flush=True)
    shared = precompute_shared_indicators(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'],
        base_engine
    )
    ohlcv_with_shared = dict(ohlcv)
    ohlcv_with_shared['shared_indicators'] = shared

    tp = make_trade_params(spread_points=0.30, use_open_entry=1.0)
    args = [(eng, conf, tp, start_idx, end_idx) for eng in top_engines for conf in confs]
    total = len(args)
    print(f"\n[Phase 2] Confluence Sweep — {total:,} combos ({len(top_engines)} engines × {len(confs)} configs)")

    # FIX v3.2: Warm up before Pool
    print("  Warming up Numba JIT...", flush=True)
    _warmup_numba()
    _init_worker(ohlcv_with_shared)
    _worker_phase2(args[0])
    print("  Numba ready.", flush=True)

    t0 = time.time()
    results = []
    pi = max(50, total // 100); last_p = t0
    print(f"  Starting...", flush=True)

    with Pool(n_cores, initializer=_init_worker, initargs=(ohlcv_with_shared,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_phase2, args, chunksize=max(1, total // (n_cores * 8)))):
            results.append(r)
            now = time.time()
            if (i + 1) % pi == 0 or (now - last_p > 10):
                last_p = now
                pct = (i+1)/total*100; rate = (i+1)/(now-t0)
                best = max(r[3] for r in results)
                bar = '\u2588' * int(pct // 5) + '\u2591' * (20 - int(pct // 5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] {rate:.0f}/s | Best: {best:.1f}", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[3], reverse=True)
    print(f"\n[Phase 2] Done: {elapsed:.1f}s | Best: {results[0][3]:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_results(data, filepath):
    with open(filepath, 'wb') as f: pickle.dump(data, f)
    print(f"  Saved: {filepath}")

def load_results(filepath):
    with open(filepath, 'rb') as f: return pickle.load(f)

def print_top(results, top_n=20, phase=""):
    print(f"\n{'='*120}")
    print(f"  TOP {top_n} {phase}")
    print(f"{'='*120}")
    for i, r in enumerate(results[:top_n]):
        if len(r) == 3:
            eng, res, score = r
            tname = TRIG_NAMES[eng.trigger_type]
            if eng.trigger_type == TRIG_MA: tname = MA_NAMES[eng.ma_type]
            print(f"  #{i+1} Score={score:.1f} | PF={res[R_PROFIT_FACTOR]:.2f} | WR={res[R_WIN_RATE]:.1f}% | "
                  f"Trades={int(res[R_TOTAL_TRADES])} | DD={res[R_MAX_DRAWDOWN_PCT]:.1f}% | "
                  f"{tname} L={eng.ma_length} TF={eng.tfactor:.2f} S={eng.sensitivity} "
                  f"mode={eng.signal_mode} HA={eng.use_heiken_ashi}")
        elif len(r) == 4:
            eng, conf, res, score = r
            tname = TRIG_NAMES[eng.trigger_type]
            if eng.trigger_type == TRIG_MA: tname = MA_NAMES[eng.ma_type]
            active = [IND_NAMES[j] for j in range(NUM_IND) if conf.indicator_mask & (1 << j)]
            filt = '+'.join(active) if active else 'none'
            print(f"  #{i+1} Score={score:.1f} | PF={res[R_PROFIT_FACTOR]:.2f} | "
                  f"Trades={int(res[R_TOTAL_TRADES])} | {tname} L={eng.ma_length} | "
                  f"Confluence({conf.min_agree}/{len(active)}): {filt}")
    print(f"{'='*120}")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Trade Management Sweep
# ─────────────────────────────────────────────────────────────────────────────

def _worker_phase3(args):
    engine, conf, tp, start_idx, end_idx = args
    d = _shared
    try:
        shared = d.get('shared_indicators')
        sig = generate_universal_signals(
            d['open'], d['high'], d['low'], d['close'], d['volume'],
            engine, d.get('htf_1h'), d.get('htf_4h'),
            shared=shared,
        )
        buy_f, sell_f = apply_confluence(sig.buy, sig.sell, sig.ind_buy, sig.ind_sell, conf)
        n = len(d['close'])
        t3_trend = np.zeros(n, dtype=np.int32)
        current = 0
        for i in range(n):
            if buy_f[i]: current = 1
            elif sell_f[i]: current = -1
            t3_trend[i] = current
        ibb = pack_indicator_bits(sig.ind_buy)
        ibs = pack_indicator_bits(sig.ind_sell)
        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'], sig.open_next,
            t3_trend, buy_f, sell_f, sig.atr,
            ibb, ibs, 0, 0,
            sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
            tp, start_idx, end_idx,
        )
        return (engine, conf, tp, compute_score(res), res)
    except Exception:
        return (engine, conf, tp, -999.0, np.zeros(NUM_RESULTS, dtype=np.float64))


def run_phase3(ohlcv, top_phase2, n_cores=None, start_idx=0, end_idx=-1):
    if n_cores is None: n_cores = cpu_count()

    tp_combos = []
    for be in BE_GRID:
        for ta in TRAIL_ACT:
            for to in TRAIL_OFF:
                for hs in HARD_STOP:
                    for pe in PSAR_EXIT:
                        for sp in SPREAD_GRID:
                            tp = make_trade_params(
                                break_even_atr=be,
                                trail_activate_atr=ta,
                                trail_offset_atr=to,
                                atr_stop_mult=hs,
                                psar_exit_enabled=pe,
                                spread_points=sp,
                                use_open_entry=1.0,
                            )
                            tp_combos.append(tp)

    args = []
    for eng, conf, res, score in top_phase2:
        for tp in tp_combos:
            args.append((eng, conf, tp, start_idx, end_idx))

    total = len(args)
    n_configs = len(top_phase2)
    n_tp = len(tp_combos)
    print(f"\n[Phase 3] Trade Management Sweep — {total:,} combos ({n_configs} configs × {n_tp:,} trade params)")

    base_engine = EngineConfig()
    print("  Precomputing shared indicators...", flush=True)
    shared = precompute_shared_indicators(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'],
        base_engine
    )
    ohlcv_with_shared = dict(ohlcv)
    ohlcv_with_shared['shared_indicators'] = shared

    # FIX v3.2: Warm up before Pool
    print("  Warming up Numba JIT...", flush=True)
    _warmup_numba()
    _init_worker(ohlcv_with_shared)
    _worker_phase3(args[0])
    print("  Numba ready.", flush=True)

    t0 = time.time()
    results = []
    pi = max(50, total // 100); last_p = t0
    print(f"  Starting...", flush=True)

    with Pool(n_cores, initializer=_init_worker, initargs=(ohlcv_with_shared,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_phase3, args, chunksize=max(1, total // (n_cores * 8)))):
            results.append(r)
            now = time.time()
            if (i + 1) % pi == 0 or (now - last_p > 10):
                last_p = now
                pct = (i+1)/total*100; rate = (i+1)/(now-t0)
                best = max(r[3] for r in results)
                bar = '\u2588' * int(pct // 5) + '\u2591' * (20 - int(pct // 5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] {rate:.0f}/s | ETA {(total-i-1)/max(rate,0.01):.0f}s | Best: {best:.1f}", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[3], reverse=True)
    print(f"\n[Phase 3] Done: {elapsed:.1f}s | Best: {results[0][3]:.1f}")

    return results

"""
Module 3 — 5-Phase Grid Search Funnel (MEGA Optimizer V2)
Phase 1: T3 Core Discovery (pure T3 flip, no filters)
Phase 2: Independent Indicator Testing (each ON/OFF)
Phase 3: Confluence Sweep (voter subsets × gate subsets × thresholds)
Phase 4: Trade Management Sweep
Phase 5: Walk-Forward Validation (handled by walk_forward_v2.py)
"""

import numpy as np
import itertools
import math
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any
import time
import pickle
import os

from .signals_v2 import (
    T3Params, IndicatorParams, ConfluenceConfig, SignalArrays,
    generate_signals, apply_confluence, pack_indicator_bits,
    generate_all_confluence_configs,
    NUM_INDICATORS, IND_NAMES,
)
from .backtest_engine_v2 import (
    run_backtest, make_trade_params, NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_SHARPE, R_EXPECTANCY, R_NET_PROFIT, R_RECOVERY_FACTOR, R_CALMAR,
    R_FINAL_EQUITY, R_AVG_BARS_HELD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Grids
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1: T3 Core — FIX #6: UNIFORM grids, FULL ranges, NO assumptions
T3_SLOW = list(range(5, 501, 5))                          # 5 to 500, step 5 = 100 values
T_FACTOR = [round(x * 0.05, 2) for x in range(31)]        # 0.00 to 1.50, step 0.05 = 31 values
SENSITIVITY = list(range(1, 31))                           # 1 to 30, uniform = 30 values
FAST_RATIO = [0.10, 0.50]                                  # data proved fast length irrelevant

# Phase 2: Indicator internal params
IND_GRIDS = {
    'ADX': {
        'adx_len': [7,10,14,20,25,30],
        'adx_threshold': [10,15,20,25,30,35,40],
    },
    'TSI': {
        'tsi_long': [13,20,25,30,40],
        'tsi_short': [3,5,8,13],
        'tsi_signal': [5,8,13],
    },
    'RSI': {
        'rsi_period': [7,10,14,21],
        'rsi_overbought': [65,70,75,80,85,90],
        'rsi_oversold': [10,15,20,25,30,35],
    },
    'PSAR': {
        'psar_start': [0.01,0.015,0.02,0.025,0.03],
        'psar_inc': [0.01,0.015,0.02,0.025,0.03],
        'psar_max': [0.1,0.15,0.2,0.25,0.3],
    },
    'Volume': {
        'vol_sma_len': [5,10,15,20,30,50],
    },
    'Regime': {
        'regime_lookback': [50,100,200],
        'trend_metric_thresh': [0.3,0.4,0.5,0.6,0.8,1.0],
        'volatile_atr_thresh': [1.0,1.1,1.2,1.3,1.5,2.0],
    },
}

# Phase 4: Trade management
BE_GRID = [0.0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
TRAIL_GRID = [
    (999.0, 0.5),   # OFF
    (1.0, 0.3), (1.5, 0.3), (1.5, 0.5),
    (2.0, 0.5), (2.5, 0.8), (3.0, 1.0),
]
HARD_STOP_GRID = [0.0, 2.5, 3.0, 4.0, 5.0, 7.0]
PSAR_EXIT_GRID = [0.0, 1.0]
PSAR_MAT_GRID = [5.0, 10.0, 15.0]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

# FIX #8: NO quality gates. Rank EVERYTHING. Let data decide.

def compute_composite_score(results):
    """P^0.45 × S^0.30 × C^0.25. NO PF cap. NO gates. Rank everything."""
    trades = results[R_TOTAL_TRADES]
    if trades < 5:  # absolute minimum to compute stats
        return -999.0

    # NO PF CAP — let data decide
    pf = results[R_PROFIT_FACTOR]
    pf_s = min(max((pf - 1.0) / 2.0 * 100.0, 0), 100.0)
    exp = results[R_EXPECTANCY]
    exp_s = min(max(exp, 0) / 2.0 * 100.0, 100.0)
    perf = max(0.6 * pf_s + 0.4 * exp_s, 0.01)

    dd = abs(results[R_MAX_DRAWDOWN_PCT])
    dd_s = max(0, 100.0 - dd * 2.5)
    sharpe = min(max(results[R_SHARPE], 0), 3.0)
    sh_s = sharpe / 3.0 * 100.0
    rf = min(max(results[R_RECOVERY_FACTOR], 0), 5.0)
    rf_s = rf / 5.0 * 100.0
    stab = max(0.4 * dd_s + 0.35 * sh_s + 0.25 * rf_s, 0.01)

    tc_s = min(math.log(max(trades, 10)) / math.log(300) * 100.0, 100.0)
    wr = results[R_WIN_RATE]
    wr_s = min(wr / 55.0 * 100.0, 100.0)
    conf = max(0.6 * tc_s + 0.4 * wr_s, 0.01)

    return (perf ** 0.45) * (stab ** 0.30) * (conf ** 0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing shared data
# ─────────────────────────────────────────────────────────────────────────────

_shared = {}

def _init_worker(data):
    global _shared
    _shared = data


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: T3 Core Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _worker_phase1(args):
    """Worker: one T3 config → signals + backtest (no confluence, no management)."""
    t3p, tp_none, tp_be, start_idx, end_idx = args
    d = _shared

    try:
        sig = generate_signals(
            d['open'], d['high'], d['low'], d['close'], d['volume'],
            t3p, htf_close_1h=d.get('htf_1h'), htf_close_4h=d.get('htf_4h'),
        )

        # Pack bitmasks (all 8 indicators)
        ibb = pack_indicator_bits(sig.ind_buy)
        ibs = pack_indicator_bits(sig.ind_sell)

        # Backtest: NO confluence (mask=0, min_agree=0)
        res_none = run_backtest(
            d['open'], d['high'], d['low'], d['close'], sig.open_next,
            sig.t3_trend, sig.raw_flip_up, sig.raw_flip_down, sig.atr,
            ibb, ibs, 0, 0,  # no confluence
            sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
            tp_none, start_idx, end_idx,
        )

        # Backtest: with break-even 1.5 ATR
        res_be = run_backtest(
            d['open'], d['high'], d['low'], d['close'], sig.open_next,
            sig.t3_trend, sig.raw_flip_up, sig.raw_flip_down, sig.atr,
            ibb, ibs, 0, 0,
            sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
            tp_be, start_idx, end_idx,
        )

        score_none = compute_composite_score(res_none)
        score_be = compute_composite_score(res_be)

        return (t3p, res_none, score_none, res_be, score_be)
    except Exception:
        z = np.zeros(NUM_RESULTS, dtype=np.float64)
        return (t3p, z, -999.0, z, -999.0)


def run_phase1(
    ohlcv_data: Dict,
    n_cores: int = None,
    start_idx: int = 0,
    end_idx: int = -1,
    progress_every: int = 500,
) -> List:
    """Phase 1: Sweep T3 core parameters. No confluence, no management."""
    if n_cores is None: n_cores = cpu_count()

    tp_none = make_trade_params(spread_points=0.30)
    tp_be = make_trade_params(break_even_atr=1.5, spread_points=0.30)

    # Generate combos
    combos = []
    for sl in T3_SLOW:
        for tf in T_FACTOR:
            for sens in SENSITIVITY:
                if sens >= sl: continue
                for fr in FAST_RATIO:
                    fl = max(2, round(sl * fr))
                    if fl >= sl: continue
                    combos.append(T3Params(slow_len=sl, fast_len=fl, tfactor=tf, sensitivity=sens))

    total = len(combos)
    print(f"\n[Phase 1] T3 Core Discovery — {total:,} combos × {n_cores} cores")
    print(f"  T3 slow: {len(T3_SLOW)} | T-Factor: {len(T_FACTOR)} | Sens: {len(SENSITIVITY)} | Fast ratio: {len(FAST_RATIO)}")

    args = [(t3p, tp_none, tp_be, start_idx, end_idx) for t3p in combos]

    t0 = time.time()
    results = []
    import sys
    # Print every 50 combos or every 10 seconds
    last_print = t0
    print_interval = max(50, total // 100)  # ~1% increments

    print(f"  Starting... (updates every ~{print_interval} combos)", flush=True)

    with Pool(n_cores, initializer=_init_worker, initargs=(ohlcv_data,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_phase1, args, chunksize=max(1, total // (n_cores * 8)))):
            results.append(r)
            now = time.time()
            if (i + 1) % print_interval == 0 or (now - last_print > 10):
                last_print = now
                elapsed = now - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / max(rate, 0.01)
                pct = (i + 1) / total * 100
                best = max(r[2] for r in results)
                best_be = max(r[4] for r in results)
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] {rate:.0f}/s | ETA {eta:.0f}s | Best: {max(best, best_be):.1f}", flush=True)

    elapsed = time.time() - t0

    # Sort by best of (raw, BE)
    results.sort(key=lambda x: max(x[2], x[4]), reverse=True)

    passing = sum(1 for r in results if max(r[2], r[4]) > 0)
    best_score = max(max(r[2], r[4]) for r in results[:1]) if results else 0
    print(f"\n[Phase 1] Done: {elapsed:.1f}s | {passing:,}/{total:,} passed | Best: {best_score:.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Independent Indicator Testing
# ─────────────────────────────────────────────────────────────────────────────

def _worker_phase2(args):
    """Worker: one T3 config + one indicator ON with specific params."""
    t3p, indp, conf, tp, start_idx, end_idx = args
    d = _shared

    try:
        sig = generate_signals(
            d['open'], d['high'], d['low'], d['close'], d['volume'],
            t3p, indp, d.get('htf_1h'), d.get('htf_4h'),
        )

        vb = pack_voter_bits(sig, 'buy')
        vs = pack_voter_bits(sig, 'sell')
        gb = pack_gate_bits(sig)

        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'],
            sig.t3_trend, sig.raw_flip_up, sig.raw_flip_down, sig.atr,
            vb, vs, gb,
            conf.voter_mask, conf.gate_mask, conf.min_voters_agree,
            sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
            tp, start_idx, end_idx,
        )

        return (t3p, indp, conf, res, compute_composite_score(res))
    except Exception:
        return (t3p, indp, conf, np.zeros(NUM_RESULTS, dtype=np.float64), -999.0)


def run_phase2(
    ohlcv_data: Dict,
    top_t3_configs: List[T3Params],
    baseline_results: Dict,  # {t3p: baseline_pf}
    n_cores: int = None,
    start_idx: int = 0,
    end_idx: int = -1,
) -> Dict:
    """Phase 2: Test each indicator independently. Returns marginal value per indicator."""
    if n_cores is None: n_cores = cpu_count()

    tp = make_trade_params(spread_points=0.30)

    # Build test configs for each indicator independently
    # Voters: TSI=bit0, RSI=bit1, PSAR=bit2, MTF1H=bit3, MTF4H=bit4
    voter_tests = {
        'TSI': (1 << 0, 0),   # voter_mask=1, gate_mask=0
        'RSI': (1 << 1, 0),
        'PSAR': (1 << 2, 0),
        'MTF_1H': (1 << 3, 0),
        'MTF_4H': (1 << 4, 0),
    }
    # Gates: ADX=bit0, Volume=bit1, Regime=bit2
    gate_tests = {
        'ADX': (0, 1 << 0),
        'Volume': (0, 1 << 1),
        'Regime': (0, 1 << 2),
    }

    all_tests = {}
    all_tests.update(voter_tests)
    all_tests.update(gate_tests)

    args = []
    for t3p in top_t3_configs:
        for ind_name, (vm, gm) in all_tests.items():
            # For voters: min_agree=1 (must agree); for gates: gate must pass
            min_agree = 1 if vm > 0 else 0
            conf = ConfluenceConfig(voter_mask=vm, gate_mask=gm, min_voters_agree=min_agree, mtf_block=0)
            indp = IndicatorParams()  # defaults for now
            args.append((t3p, indp, conf, tp, start_idx, end_idx))

    total = len(args)
    print(f"\n[Phase 2] Independent Indicator Testing — {total:,} combos")
    print(f"  {len(top_t3_configs)} T3 configs × {len(all_tests)} indicators")

    t0 = time.time()
    results = []
    pi = max(50, total // 50)
    last_p = t0
    print(f"  Starting...", flush=True)

    with Pool(n_cores, initializer=_init_worker, initargs=(ohlcv_data,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_phase2, args, chunksize=max(1, total // (n_cores * 8)))):
            results.append(r)
            now = time.time()
            if (i + 1) % pi == 0 or (now - last_p > 10):
                last_p = now
                pct = (i+1)/total*100; rate = (i+1)/(now-t0)
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] {rate:.0f}/s | ETA {(total-i-1)/max(rate,0.01):.0f}s", flush=True)

    elapsed = time.time() - t0

    # Compute marginal value per indicator
    print(f"\n[Phase 2] Indicator Marginal Value Analysis:")
    indicator_deltas = {name: [] for name in all_tests}

    for t3p, indp, conf, res, score in results:
        # Find which indicator this test was for
        for ind_name, (vm, gm) in all_tests.items():
            if conf.voter_mask == vm and conf.gate_mask == gm:
                base_pf = baseline_results.get(t3p, 1.0)
                delta_pf = res[R_PROFIT_FACTOR] - base_pf
                indicator_deltas[ind_name].append(delta_pf)
                break

    for name, deltas in indicator_deltas.items():
        if deltas:
            med = np.median(deltas)
            mean = np.mean(deltas)
            pct_positive = sum(1 for d in deltas if d > 0) / len(deltas) * 100
            print(f"  {name:>10}: median ΔPF={med:+.4f} | mean={mean:+.4f} | positive {pct_positive:.0f}% of the time")

    print(f"[Phase 2] Done: {elapsed:.1f}s")

    return indicator_deltas


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Confluence Sweep
# ─────────────────────────────────────────────────────────────────────────────

def _worker_phase3(args):
    """Worker: one T3 config + one confluence config."""
    t3p, conf, tp, start_idx, end_idx = args
    d = _shared

    try:
        sig = generate_signals(
            d['open'], d['high'], d['low'], d['close'], d['volume'],
            t3p, htf_close_1h=d.get('htf_1h'), htf_close_4h=d.get('htf_4h'),
        )

        vb = pack_voter_bits(sig, 'buy')
        vs = pack_voter_bits(sig, 'sell')
        gb = pack_gate_bits(sig)

        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'],
            sig.t3_trend, sig.raw_flip_up, sig.raw_flip_down, sig.atr,
            vb, vs, gb,
            conf.voter_mask, conf.gate_mask, conf.min_voters_agree,
            sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
            tp, start_idx, end_idx,
        )

        return (t3p, conf, res, compute_composite_score(res))
    except Exception:
        return (t3p, conf, np.zeros(NUM_RESULTS, dtype=np.float64), -999.0)


def run_phase3(
    ohlcv_data: Dict,
    top_t3_configs: List[T3Params],
    confluence_configs: List[ConfluenceConfig] = None,
    n_cores: int = None,
    start_idx: int = 0,
    end_idx: int = -1,
    progress_every: int = 500,
) -> List:
    """Phase 3: Sweep confluence configurations."""
    if n_cores is None: n_cores = cpu_count()

    if confluence_configs is None:
        confluence_configs = generate_all_confluence_configs(include_no_filter=True)

    tp = make_trade_params(spread_points=0.30)

    args = [(t3p, conf, tp, start_idx, end_idx)
            for t3p in top_t3_configs
            for conf in confluence_configs]

    total = len(args)
    print(f"\n[Phase 3] Confluence Sweep — {total:,} combos")
    print(f"  {len(top_t3_configs)} T3 configs × {len(confluence_configs)} confluence configs")

    t0 = time.time()
    results = []
    pi = max(50, total // 100)
    last_p = t0
    print(f"  Starting...", flush=True)

    with Pool(n_cores, initializer=_init_worker, initargs=(ohlcv_data,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_phase3, args, chunksize=max(1, total // (n_cores * 8)))):
            results.append(r)
            now = time.time()
            if (i + 1) % pi == 0 or (now - last_p > 10):
                last_p = now
                pct = (i+1)/total*100; rate = (i+1)/(now-t0)
                best = max(r[3] for r in results)
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] {rate:.0f}/s | ETA {(total-i-1)/max(rate,0.01):.0f}s | Best: {best:.1f}", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[3], reverse=True)

    passing = sum(1 for r in results if r[3] > 0)
    print(f"\n[Phase 3] Done: {elapsed:.1f}s | {passing:,}/{total:,} passed")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Trade Management Sweep
# ─────────────────────────────────────────────────────────────────────────────

def generate_trade_combos():
    """Generate all trade management combinations."""
    combos = []
    for be in BE_GRID:
        for (trail_act, trail_off) in TRAIL_GRID:
            for hs in HARD_STOP_GRID:
                for psar_exit in PSAR_EXIT_GRID:
                    # Skip PSAR maturity variants when PSAR exit is OFF
                    psar_mats = PSAR_MAT_GRID if psar_exit > 0.5 else [10.0]
                    for pm in psar_mats:
                        combos.append(make_trade_params(
                            break_even_atr=be,
                            atr_stop_mult=hs,
                            trail_activate_atr=trail_act,
                            trail_offset_atr=trail_off,
                            psar_exit_enabled=psar_exit,
                            psar_maturity_min=pm,
                            spread_points=0.30,
                        ))
    return combos


def _worker_phase4(args):
    """Worker: precomputed signals + one trade management config."""
    tp, start_idx, end_idx = args
    d = _shared

    try:
        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'],
            d['t3_trend'], d['raw_flip_up'], d['raw_flip_down'], d['atr'],
            d['vb'], d['vs'], d['gb'],
            d['voter_mask'], d['gate_mask'], d['min_voters'],
            d['psar_flip_up'], d['psar_flip_down'], d['swing_high'], d['swing_low'],
            tp, start_idx, end_idx,
        )
        return (tp, res, compute_composite_score(res))
    except Exception:
        return (tp, np.zeros(NUM_RESULTS, dtype=np.float64), -999.0)


def run_phase4(
    ohlcv_data: Dict,
    signal_data: Dict,  # precomputed signal arrays + confluence config
    n_cores: int = None,
    start_idx: int = 0,
    end_idx: int = -1,
) -> List:
    """Phase 4: Sweep trade management on precomputed signals."""
    if n_cores is None: n_cores = cpu_count()

    combos = generate_trade_combos()
    total = len(combos)
    print(f"\n[Phase 4] Trade Management Sweep — {total:,} combos")

    # Merge ohlcv and signal data for workers
    shared = {**ohlcv_data, **signal_data}
    args = [(tp, start_idx, end_idx) for tp in combos]

    t0 = time.time()
    results = []

    with Pool(n_cores, initializer=_init_worker, initargs=(shared,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_phase4, args, chunksize=max(1, total // (n_cores * 4)))):
            results.append(r)

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"[Phase 4] Done: {elapsed:.1f}s | Best: {results[0][2]:.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_results(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved: {filepath}")

def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def print_phase1_top(results, top_n=20):
    print(f"\n{'='*100}")
    print(f"  TOP {top_n} — Phase 1 (T3 Core)")
    print(f"{'='*100}")
    for i, (t3p, res_none, score_none, res_be, score_be) in enumerate(results[:top_n]):
        best_res = res_be if score_be > score_none else res_none
        best_score = max(score_none, score_be)
        is_be = "BE" if score_be > score_none else "raw"
        print(f"  #{i+1} [{is_be}] Score={best_score:.1f} | "
              f"PF={best_res[R_PROFIT_FACTOR]:.2f} | WR={best_res[R_WIN_RATE]:.1f}% | "
              f"Trades={int(best_res[R_TOTAL_TRADES])} | DD={best_res[R_MAX_DRAWDOWN_PCT]:.1f}% | "
              f"Exp={best_res[R_EXPECTANCY]:.3f} | "
              f"T3 L{t3p.slow_len}/F{t3p.fast_len} TF={t3p.tfactor:.3f} S={t3p.sensitivity}")
    print(f"{'='*100}")

def print_phase3_top(results, top_n=20):
    print(f"\n{'='*100}")
    print(f"  TOP {top_n} — Phase 3 (Confluence)")
    print(f"{'='*100}")
    for i, (t3p, conf, res, score) in enumerate(results[:top_n]):
        active_v = bin(conf.voter_mask).count('1')
        active_g = bin(conf.gate_mask).count('1')
        voters = [VOTER_NAMES[j] for j in range(NUM_VOTERS) if conf.voter_mask & (1 << j)]
        gates = [GATE_NAMES[j] for j in range(NUM_GATES) if conf.gate_mask & (1 << j)]
        v_str = '+'.join(voters) if voters else 'none'
        g_str = '+'.join(gates) if gates else 'none'
        print(f"  #{i+1} Score={score:.1f} | "
              f"PF={res[R_PROFIT_FACTOR]:.2f} | Trades={int(res[R_TOTAL_TRADES])} | "
              f"T3 L{t3p.slow_len} TF={t3p.tfactor:.2f} S={t3p.sensitivity} | "
              f"Voters({conf.min_voters_agree}/{active_v}): {v_str} | "
              f"Gates: {g_str}")
    print(f"{'='*100}")

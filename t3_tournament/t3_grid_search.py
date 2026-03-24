# T3 Tournament Grid Search
# Multiprocessing sweep across all T3 combo structures
# Pattern mirrors grid_search_v3.py exactly

import numpy as np
import math
import time
import pickle
import pandas as pd
from multiprocessing import Pool, cpu_count

from .t3_engine import (
    T3Config, generate_all_combos,
    compute_t3_signals,
    STRUCT_NAMES, SOURCE_NAMES, MODE_NAMES, IND_INPUT_NAMES,
    STRUCT_SINGLE, STRUCT_CROSS, STRUCT_TRIPLE, STRUCT_OF_IND,
    _ema, _rma, _sma, _t3, heiken_ashi, atr_calc, rsi_calc,
    tsi_val, adx_di, compute_signal_mode, compute_crossover,
    compute_crossover_slope_confirm, compute_crossover_accel_confirm,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nika_optimizer.backtest_engine_v2 import (
    run_backtest, make_trade_params, NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT,
    R_TOTAL_TRADES, R_NET_PROFIT, R_EXPECTANCY, R_SHARPE,
    R_RECOVERY_FACTOR, R_CALMAR, R_AVG_BARS_HELD,
    R_MAX_CONSEC_LOSS,
)
from nika_optimizer.signals_v3 import (
    atr_calc as atr_calc_v3,
    psar_calc, highest_n, lowest_n,
    pack_indicator_bits,
)


# ---------------------------------------------------------------------------
# Scoring — identical formula to grid_search_v3
# ---------------------------------------------------------------------------

def compute_score(res):
    trades = res[R_TOTAL_TRADES]
    if trades < 5: return -999.0
    pf    = res[R_PROFIT_FACTOR]
    pf_s  = min(max((pf - 1.0) / 2.0 * 100.0, 0), 100.0)
    exp   = res[R_EXPECTANCY]
    exp_s = min(max(exp, 0) / 2.0 * 100.0, 100.0)
    perf  = max(0.6 * pf_s + 0.4 * exp_s, 0.01)
    dd    = abs(res[R_MAX_DRAWDOWN_PCT])
    dd_s  = max(0, 100.0 - dd * 2.5)
    sharpe = min(max(res[R_SHARPE], 0), 3.0)
    sh_s  = sharpe / 3.0 * 100.0
    stab  = max(0.5 * dd_s + 0.5 * sh_s, 0.01)
    tc_s  = min(math.log(max(trades, 10)) / math.log(300) * 100.0, 100.0)
    wr    = res[R_WIN_RATE]
    wr_s  = min(wr / 55.0 * 100.0, 100.0)
    conf  = max(0.6 * tc_s + 0.4 * wr_s, 0.01)
    return (perf ** 0.45) * (stab ** 0.30) * (conf ** 0.25)


# ---------------------------------------------------------------------------
# Worker shared state
# ---------------------------------------------------------------------------

_shared = {}

def _init_worker(data):
    global _shared
    _shared = data


def _worker(args):
    cfg, tp = args
    d = _shared
    try:
        buy, sell, main_line = compute_t3_signals(
            cfg, d['open'], d['high'], d['low'], d['close'], d['volume']
        )
        n   = len(d['close'])
        atr = d['atr']
        pfu = d['psar_flip_up']
        pfd = d['psar_flip_down']
        swh = d['swing_high']
        swl = d['swing_low']
        ont = d['open_next']

        t3_trend = np.zeros(n, dtype=np.int32)
        cur = 0
        for i in range(n):
            if buy[i]:  cur =  1
            elif sell[i]: cur = -1
            t3_trend[i] = cur

        dummy_bits = np.zeros(n, dtype=np.int16)
        res = run_backtest(
            d['open'], d['high'], d['low'], d['close'], ont,
            t3_trend, buy, sell, atr,
            dummy_bits, dummy_bits, 0, 0,
            pfu, pfd, swh, swl,
            tp, 0, -1,
        )
        return (cfg, res, compute_score(res))
    except Exception:
        return (cfg, np.zeros(NUM_RESULTS, dtype=np.float64), -999.0)


# ---------------------------------------------------------------------------
# Numba warm-up
# ---------------------------------------------------------------------------

def _warmup():
    z = np.ones(200, dtype=np.float64)
    h = z * 1.1; l = z * 0.9
    _ema(z, 5); _rma(z, 5); _sma(z, 5); _t3(z, 5, 0.7)
    heiken_ashi(z, h, l, z)
    atr_calc(h, l, z, 5)
    rsi_calc(z, 5)
    tsi_val(z, 5, 13, 8)
    adx_di(h, l, z, 5)
    compute_signal_mode(z, z, 3, 0)
    compute_crossover(z, z * 1.01)
    compute_crossover_slope_confirm(z * 1.01, z, 3)
    compute_crossover_accel_confirm(z * 1.01, z, 3)
    psar_calc(h, l, 0.02, 0.02, 0.2)
    highest_n(z, 5); lowest_n(z, 5)


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------

def run_t3_tournament(ohlcv, n_cores=None, quick=False):
    if n_cores is None:
        n_cores = cpu_count()

    print("\n" + "=" * 80)
    print("  T3 TOURNAMENT — Generating combos...")
    print("=" * 80)

    combos = generate_all_combos()
    total  = len(combos)
    print(f"  Running {total:,} combos on {n_cores} cores")

    if quick:
        import random; random.seed(42)
        combos = random.sample(combos, min(5000, total))
        total  = len(combos)
        print(f"  Quick mode: sampled {total:,} combos")

    # Precompute shared arrays
    close   = ohlcv['close']
    open_   = ohlcv['open']
    high    = ohlcv['high']
    low     = ohlcv['low']
    volume  = ohlcv['volume']
    n       = len(close)

    atr_v = atr_calc_v3(high, low, close, 14)
    psar_v, psar_bull = psar_calc(high, low, 0.02, 0.02, 0.2)
    pfu = np.zeros(n, dtype=np.bool_)
    pfd = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if  psar_bull[i] and not psar_bull[i-1]: pfu[i] = True
        if not psar_bull[i] and psar_bull[i-1]:  pfd[i] = True
    ont = np.empty(n, dtype=np.float64)
    ont[:-1] = open_[1:]; ont[-1] = close[-1]
    swh = highest_n(high, 10)
    swl = lowest_n(low, 10)

    shared_data = {
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
        'atr': atr_v, 'psar_flip_up': pfu, 'psar_flip_down': pfd,
        'swing_high': swh, 'swing_low': swl, 'open_next': ont,
    }

    tp = make_trade_params(spread_points=0.30, use_open_entry=1.0)
    args = [(cfg, tp) for cfg in combos]

    print("  Warming up Numba JIT...", flush=True)
    _warmup()
    _init_worker(shared_data)
    _worker(args[0])
    print("  Numba ready.", flush=True)

    t0 = time.time()
    results = []
    pi = max(100, total // 200)
    last_p = t0

    with Pool(n_cores, initializer=_init_worker, initargs=(shared_data,)) as pool:
        chunk = max(1, total // (n_cores * 10))
        for i, r in enumerate(pool.imap_unordered(_worker, args, chunksize=chunk)):
            results.append(r)
            now = time.time()
            if (i + 1) % pi == 0 or (now - last_p) > 15:
                last_p = now
                pct  = (i+1) / total * 100
                rate = (i+1) / (now - t0)
                best = max(x[2] for x in results)
                bar  = '\u2588' * int(pct // 5) + '\u2591' * (20 - int(pct // 5))
                print(f"  {bar} {pct:5.1f}% [{i+1:,}/{total:,}] "
                      f"{rate:.0f}/s | ETA {(total-i-1)/max(rate,0.01):.0f}s | Best: {best:.1f}",
                      flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Done: {elapsed:.1f}s ({total/elapsed:.0f} combos/s) | Best score: {results[0][2]:.1f}")
    return results


# ---------------------------------------------------------------------------
# Results → DataFrame + CSV
# ---------------------------------------------------------------------------

def results_to_df(results):
    """Convert list of (T3Config, results_array, score) to a flat DataFrame."""
    rows = []
    for cfg, res, score in results:
        struct = STRUCT_NAMES[cfg.structure]
        src    = SOURCE_NAMES[cfg.src_id] if cfg.structure != STRUCT_OF_IND else IND_INPUT_NAMES[cfg.ind_input]
        mode   = MODE_NAMES[cfg.signal_mode]

        row = {
            # Score
            'score':          round(score, 4),
            # Structure
            'structure':      struct,
            'source':         src,
            'signal_mode':    mode,
            'cross_type':     cfg.cross_type,
            # T3 params
            'slow_len':       cfg.slow_len,
            'slow_vf':        cfg.slow_vf,
            'fast_len':       cfg.fast_len,
            'fast_vf':        cfg.fast_vf,
            'mid_len':        cfg.mid_len,
            'mid_vf':         cfg.mid_vf,
            'sensitivity':    cfg.sensitivity,
            # Indicator input (STRUCT_OF_IND only)
            'ind_input':      IND_INPUT_NAMES[cfg.ind_input] if cfg.structure == STRUCT_OF_IND else '',
            'ind_period':     cfg.ind_period if cfg.structure == STRUCT_OF_IND else 0,
            # Backtest metrics
            'profit_factor':  round(float(res[R_PROFIT_FACTOR]),  4),
            'win_rate':       round(float(res[R_WIN_RATE]),        2),
            'trades':         int(res[R_TOTAL_TRADES]),
            'net_profit':     round(float(res[R_NET_PROFIT]),      4),
            'expectancy':     round(float(res[R_EXPECTANCY]),      4),
            'max_dd_pct':     round(float(res[R_MAX_DRAWDOWN_PCT]),2),
            'sharpe':         round(float(res[R_SHARPE]),          4),
            'recovery_factor':round(float(res[R_RECOVERY_FACTOR]),4),
            'calmar':         round(float(res[R_CALMAR]),          4),
            'avg_bars_held':  round(float(res[R_AVG_BARS_HELD]),   1),
            'max_consec_loss':int(res[R_MAX_CONSEC_LOSS]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_csv(results, path):
    """Save full results to CSV — every combo, all metrics, all params."""
    df = results_to_df(results)
    df.to_csv(path, index=False)
    print(f"  CSV saved: {path} ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Save / Load / Print
# ---------------------------------------------------------------------------

def save_results(data, path):
    with open(path, 'wb') as f: pickle.dump(data, f)
    print(f"  PKL saved: {path}")

def load_results(path):
    with open(path, 'rb') as f: return pickle.load(f)

def print_top(results, top_n=30):
    print(f"\n{'='*120}")
    print(f"  TOP {top_n} T3 TOURNAMENT RESULTS")
    print(f"{'='*120}")
    hdr = (f"  {'#':>3} | {'Score':>7} | {'PF':>6} | {'WR%':>5} | {'Trades':>7} | "
           f"{'DD%':>6} | {'Sharpe':>6} | {'Structure':>12} | {'Source':>10} | {'Mode':>10} | Config")
    print(hdr)
    print(f"  {'-'*120}")
    for i, (cfg, res, score) in enumerate(results[:top_n]):
        struct = STRUCT_NAMES[cfg.structure]
        src    = SOURCE_NAMES[cfg.src_id] if cfg.structure != STRUCT_OF_IND else IND_INPUT_NAMES[cfg.ind_input]
        mode   = MODE_NAMES[cfg.signal_mode]
        if cfg.structure in (STRUCT_SINGLE, STRUCT_OF_IND):
            detail = f"L={cfg.slow_len} VF={cfg.slow_vf:.1f} S={cfg.sensitivity}"
        elif cfg.structure == STRUCT_CROSS:
            detail = (f"slow={cfg.slow_len}(vf={cfg.slow_vf:.1f}) "
                      f"fast={cfg.fast_len}(vf={cfg.fast_vf:.1f}) "
                      f"ct={cfg.cross_type} S={cfg.sensitivity}")
        else:
            detail = (f"fast={cfg.fast_len} mid={cfg.mid_len} slow={cfg.slow_len} "
                      f"vf=({cfg.fast_vf:.1f}/{cfg.mid_vf:.1f}/{cfg.slow_vf:.1f}) "
                      f"S={cfg.sensitivity}")
        print(f"  {i+1:>3} | {score:7.1f} | {res[R_PROFIT_FACTOR]:6.2f} | "
              f"{res[R_WIN_RATE]:5.1f} | {int(res[R_TOTAL_TRADES]):>7} | "
              f"{res[R_MAX_DRAWDOWN_PCT]:6.1f} | {res[R_SHARPE]:6.3f} | "
              f"{struct:>12} | {src:>10} | {mode:>10} | {detail}")
    print(f"{'='*120}")

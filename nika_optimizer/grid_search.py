"""
Module 4 — Grid Search + Multiprocessing
Massive parameter sweep across all 24 cores.
Two-phase approach:
  Phase 1: Signal param grid (T3 length, sensitivity, ADX, RSI, etc.)
           → vectorized signal generation, then Numba backtest
  Phase 2: Trade management param grid (trailing, break-even, PSAR, hard stop)
           → re-use best signals, sweep trade params through Numba

This avoids redundant signal computation — the expensive part.
"""

import numpy as np
import itertools
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import time
import pickle
import os

from .signals import SignalParams, generate_signals, SignalOutput
from .backtest_engine import (
    run_backtest, make_trade_params, NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_SHARPE, R_EXPECTANCY, R_RECOVERY_FACTOR, R_NET_PROFIT, R_CALMAR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Grid Definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalGrid:
    """Parameter ranges for signal generation sweep."""
    # T3 — wide sweep
    t3_slow_lens: List[int] = field(default_factory=lambda: list(range(15, 101, 5)))
    t3_fast_lens: List[int] = field(default_factory=lambda: [3, 5, 8])
    t3_alpha: List[float] = field(default_factory=lambda: [0.7])  # proven sweet spot
    t3_sensitivity: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7])
    
    # ADX
    adx_len: List[int] = field(default_factory=lambda: [10, 14, 20])
    adx_threshold: List[float] = field(default_factory=lambda: [15.0, 20.0, 25.0, 30.0])
    
    # RSI
    rsi_overbought: List[int] = field(default_factory=lambda: [70, 80, 90])
    rsi_oversold: List[int] = field(default_factory=lambda: [10, 20, 30])
    
    # Quality gate
    min_quality_score: List[float] = field(default_factory=lambda: [40.0, 50.0, 60.0, 70.0])
    
    # Trend strength filter
    min_trend_strength: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.6])
    
    # Regime
    trend_metric_thresh: List[float] = field(default_factory=lambda: [0.40, 0.60, 0.80])
    volatile_atr_thresh: List[float] = field(default_factory=lambda: [1.20, 1.30, 1.50])


@dataclass
class TradeGrid:
    """Parameter ranges for trade management sweep."""
    # Hard ATR stop (0 = disabled)
    atr_stop_mult: List[float] = field(default_factory=lambda: [0.0, 2.0, 2.5, 3.0, 3.5, 4.0])
    
    # Trailing stop
    trail_activate_atr: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])
    trail_offset_atr: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.8, 1.0])
    
    # Break-even
    breakeven_atr: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])
    
    # PSAR exit
    psar_exit_enabled: List[float] = field(default_factory=lambda: [0.0, 1.0])
    psar_maturity_pct: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    psar_maturity_min: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0])
    
    # Swing lookback
    swing_lookback: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0])


def count_combos(grid) -> int:
    """Count total parameter combinations in a grid."""
    total = 1
    for field_name in grid.__dataclass_fields__:
        total *= len(getattr(grid, field_name))
    return total


def generate_signal_combos(grid: SignalGrid) -> List[SignalParams]:
    """Generate all SignalParams combinations from the grid."""
    combos = []
    for vals in itertools.product(
        grid.t3_slow_lens,
        grid.t3_fast_lens,
        grid.t3_alpha,
        grid.t3_sensitivity,
        grid.adx_len,
        grid.adx_threshold,
        grid.rsi_overbought,
        grid.rsi_oversold,
        grid.min_quality_score,
        grid.min_trend_strength,
        grid.trend_metric_thresh,
        grid.volatile_atr_thresh,
    ):
        # Skip invalid: fast length must be < slow length
        if vals[1] >= vals[0]:
            continue
        # Skip invalid: sensitivity must be < slow length
        if vals[3] >= vals[0]:
            continue
            
        combos.append(SignalParams(
            t3_slow_len=vals[0],
            t3_fast_len=vals[1],
            t3_alpha=vals[2],
            t3_sensitivity=vals[3],
            adx_len=vals[4],
            adx_threshold=vals[5],
            rsi_overbought=vals[6],
            rsi_oversold=vals[7],
            min_quality_score=vals[8],
            min_trend_strength=vals[9],
            trend_metric_thresh=vals[10],
            volatile_atr_thresh=vals[11],
        ))
    return combos


def generate_trade_combos(grid: TradeGrid) -> List[np.ndarray]:
    """Generate all trade_params arrays from the grid."""
    combos = []
    for vals in itertools.product(
        grid.atr_stop_mult,
        grid.trail_activate_atr,
        grid.trail_offset_atr,
        grid.breakeven_atr,
        grid.psar_exit_enabled,
        grid.psar_maturity_pct,
        grid.psar_maturity_min,
        grid.swing_lookback,
    ):
        # Skip invalid: break-even should be <= trail activation
        if vals[3] > vals[1] + 0.5:
            continue
        # Skip PSAR maturity params when PSAR exit disabled
        if vals[4] < 0.5 and (vals[5] != grid.psar_maturity_pct[0] or vals[6] != grid.psar_maturity_min[0]):
            continue
            
        combos.append(make_trade_params(
            atr_stop_mult=vals[0],
            trail_activate_atr=vals[1],
            trail_offset_atr=vals[2],
            breakeven_atr=vals[3],
            psar_exit_enabled=vals[4],
            psar_maturity_pct=vals[5],
            psar_maturity_min=vals[6],
            swing_lookback=vals[7],
        ))
    return combos


# ─────────────────────────────────────────────────────────────────────────────
# Worker Functions for Multiprocessing
# ─────────────────────────────────────────────────────────────────────────────

# Global shared data (set by initializer)
_shared_data = {}


def _init_worker(data_dict):
    """Initialize worker process with shared OHLCV data."""
    global _shared_data
    _shared_data = data_dict


def _worker_signal_and_backtest(args):
    """
    Worker: compute signals for one SignalParams, then run backtest with default trade params.
    Returns: (signal_params_tuple, results_array)
    """
    signal_params, trade_params, start_idx, end_idx = args
    
    d = _shared_data
    open_ = d['open']
    high = d['high']
    low = d['low']
    close = d['close']
    volume = d['volume']
    htf_close_1h = d.get('htf_close_1h')
    htf_close_4h = d.get('htf_close_4h')
    swing_lookback = int(d.get('swing_lookback', 10))
    
    try:
        sig = generate_signals(
            open_, high, low, close, volume,
            signal_params,
            htf_close_1h=htf_close_1h,
            htf_close_4h=htf_close_4h,
            swing_lookback=swing_lookback,
        )
        
        results = run_backtest(
            open_, high, low, close,
            sig.t3_trend, sig.atr,
            sig.psar_flip_up, sig.psar_flip_down,
            sig.swing_high, sig.swing_low,
            sig.signal_buy, sig.signal_sell,
            trade_params,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        
        return (signal_params, results)
    except Exception as e:
        # Return empty results on error
        return (signal_params, np.zeros(NUM_RESULTS, dtype=np.float64))


def _worker_trade_params_only(args):
    """
    Worker: run backtest with pre-computed signals and one trade_params combo.
    Signals are stored in shared data.
    Returns: (trade_params_array, results_array)
    """
    trade_params, start_idx, end_idx = args
    
    d = _shared_data
    
    try:
        results = run_backtest(
            d['open'], d['high'], d['low'], d['close'],
            d['t3_trend'], d['atr'],
            d['psar_flip_up'], d['psar_flip_down'],
            d['swing_high'], d['swing_low'],
            d['signal_buy'], d['signal_sell'],
            trade_params,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        return (trade_params, results)
    except Exception:
        return (trade_params, np.zeros(NUM_RESULTS, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# Ranking & Filtering
# ─────────────────────────────────────────────────────────────────────────────

# Quality gates (matching your 3-score framework)
MIN_TRADES = 100
MIN_PF = 1.10
MAX_DD = -35.0  # negative value
MIN_WIN_RATE = 0.0  # no minimum — some strategies are low WR high RR


def passes_gates(results: np.ndarray) -> bool:
    """Check if a result passes minimum quality gates."""
    return (
        results[R_TOTAL_TRADES] >= MIN_TRADES and
        results[R_PROFIT_FACTOR] >= MIN_PF and
        results[R_MAX_DRAWDOWN_PCT] >= MAX_DD and  # DD is negative
        results[R_NET_PROFIT] > 0
    )


def compute_rank_score(results: np.ndarray) -> float:
    """
    Composite ranking score matching your P^0.45 * S^0.30 * C^0.25 framework.
    P = Performance (PF, expectancy, net profit)
    S = Stability (max DD, Sharpe, recovery factor)
    C = Confidence (trade count, win rate consistency)
    """
    if not passes_gates(results):
        return -999.0
    
    # Performance score (0-100)
    pf = min(results[R_PROFIT_FACTOR], 5.0)
    pf_score = min((pf - 1.0) / 3.0 * 100.0, 100.0)  # PF 1.0→4.0 maps to 0→100
    
    exp = results[R_EXPECTANCY]
    exp_score = min(max(exp, 0) / 2.0 * 100.0, 100.0)  # expectancy 0→2 ATR maps to 0→100
    
    perf = 0.6 * pf_score + 0.4 * exp_score
    
    # Stability score (0-100)
    dd = abs(results[R_MAX_DRAWDOWN_PCT])
    dd_score = max(0, 100.0 - dd * 4.0)  # 0% DD = 100, 25% DD = 0
    
    sharpe = min(max(results[R_SHARPE], 0), 3.0)
    sharpe_score = sharpe / 3.0 * 100.0
    
    rf = min(max(results[R_RECOVERY_FACTOR], 0), 5.0)
    rf_score = rf / 5.0 * 100.0
    
    stab = 0.4 * dd_score + 0.35 * sharpe_score + 0.25 * rf_score
    
    # Confidence score (0-100)
    trades = results[R_TOTAL_TRADES]
    trade_score = min(trades / 500.0 * 100.0, 100.0)
    
    wr = results[R_WIN_RATE]
    wr_score = min(wr / 60.0 * 100.0, 100.0)  # 60% WR = 100
    
    conf = 0.6 * trade_score + 0.4 * wr_score
    
    # Final rank: P^0.45 * S^0.30 * C^0.25 * 100
    perf = max(perf, 0.01)
    stab = max(stab, 0.01)
    conf = max(conf, 0.01)
    
    rank = (perf ** 0.45) * (stab ** 0.30) * (conf ** 0.25)
    return rank


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Signal Parameter Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_signal_sweep(
    ohlcv_data: Dict[str, np.ndarray],
    signal_grid: SignalGrid,
    default_trade_params: np.ndarray = None,
    start_idx: int = 0,
    end_idx: int = -1,
    n_cores: int = None,
    progress_every: int = 500,
) -> List[Tuple[SignalParams, np.ndarray, float]]:
    """
    Phase 1: Sweep all signal parameter combinations.
    
    Args:
        ohlcv_data: dict with 'open', 'high', 'low', 'close', 'volume', 
                    optionally 'htf_close_1h', 'htf_close_4h'
        signal_grid: parameter ranges
        default_trade_params: trade params to use (or default if None)
        start_idx, end_idx: data range (for walk-forward windows)
        n_cores: CPU cores to use (default: all)
    
    Returns: List of (SignalParams, results_array, rank_score), sorted by rank desc.
    """
    if n_cores is None:
        n_cores = cpu_count()
    
    if default_trade_params is None:
        default_trade_params = make_trade_params()
    
    combos = generate_signal_combos(signal_grid)
    total = len(combos)
    print(f"\n[grid_search] Phase 1: Signal sweep — {total:,} combinations × {n_cores} cores")
    
    # Prepare worker arguments
    args = [(sp, default_trade_params, start_idx, end_idx) for sp in combos]
    
    t0 = time.time()
    results_list = []
    
    with Pool(
        processes=n_cores,
        initializer=_init_worker,
        initargs=(ohlcv_data,),
    ) as pool:
        for i, (sp, res) in enumerate(pool.imap_unordered(_worker_signal_and_backtest, args, chunksize=max(1, total // (n_cores * 4)))):
            rank = compute_rank_score(res)
            results_list.append((sp, res, rank))
            
            if (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / max(rate, 0.01)
                best_rank = max(r[2] for r in results_list)
                passing = sum(1 for r in results_list if r[2] > 0)
                print(f"  [{i+1:,}/{total:,}] {rate:.0f} combos/s | ETA {eta:.0f}s | "
                      f"Best rank: {best_rank:.1f} | Passing: {passing:,}")
    
    elapsed = time.time() - t0
    
    # Sort by rank descending
    results_list.sort(key=lambda x: x[2], reverse=True)
    
    passing = sum(1 for r in results_list if r[2] > 0)
    print(f"\n[grid_search] Phase 1 complete: {elapsed:.1f}s | "
          f"{passing:,}/{total:,} passed gates | "
          f"Best rank: {results_list[0][2]:.1f}")
    
    return results_list


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Trade Management Sweep (using best signals)
# ─────────────────────────────────────────────────────────────────────────────

def run_trade_sweep(
    ohlcv_data: Dict[str, np.ndarray],
    best_signal_params: SignalParams,
    trade_grid: TradeGrid,
    htf_close_1h: np.ndarray = None,
    htf_close_4h: np.ndarray = None,
    swing_lookback: int = 10,
    start_idx: int = 0,
    end_idx: int = -1,
    n_cores: int = None,
    progress_every: int = 200,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Phase 2: Fix signal params to the best from Phase 1, sweep trade management.
    
    Pre-computes signals once, then distributes trade param combos across cores.
    """
    if n_cores is None:
        n_cores = cpu_count()
    
    # Pre-compute signals once
    print(f"\n[grid_search] Phase 2: Computing signals for best signal params...")
    sig = generate_signals(
        ohlcv_data['open'], ohlcv_data['high'], ohlcv_data['low'],
        ohlcv_data['close'], ohlcv_data['volume'],
        best_signal_params,
        htf_close_1h=htf_close_1h,
        htf_close_4h=htf_close_4h,
        swing_lookback=swing_lookback,
    )
    
    # Pack signals into shared data
    signal_data = {
        'open': ohlcv_data['open'],
        'high': ohlcv_data['high'],
        'low': ohlcv_data['low'],
        'close': ohlcv_data['close'],
        't3_trend': sig.t3_trend,
        'atr': sig.atr,
        'psar_flip_up': sig.psar_flip_up,
        'psar_flip_down': sig.psar_flip_down,
        'swing_high': sig.swing_high,
        'swing_low': sig.swing_low,
        'signal_buy': sig.signal_buy,
        'signal_sell': sig.signal_sell,
    }
    
    combos = generate_trade_combos(trade_grid)
    total = len(combos)
    print(f"[grid_search] Phase 2: Trade sweep — {total:,} combinations × {n_cores} cores")
    
    args = [(tp, start_idx, end_idx) for tp in combos]
    
    t0 = time.time()
    results_list = []
    
    with Pool(
        processes=n_cores,
        initializer=_init_worker,
        initargs=(signal_data,),
    ) as pool:
        for i, (tp, res) in enumerate(pool.imap_unordered(_worker_trade_params_only, args, chunksize=max(1, total // (n_cores * 4)))):
            rank = compute_rank_score(res)
            results_list.append((tp, res, rank))
            
            if (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                best_rank = max(r[2] for r in results_list)
                print(f"  [{i+1:,}/{total:,}] {rate:.0f} combos/s | Best rank: {best_rank:.1f}")
    
    elapsed = time.time() - t0
    results_list.sort(key=lambda x: x[2], reverse=True)
    
    passing = sum(1 for r in results_list if r[2] > 0)
    print(f"\n[grid_search] Phase 2 complete: {elapsed:.1f}s | "
          f"{passing:,}/{total:,} passed gates | "
          f"Best rank: {results_list[0][2]:.1f}")
    
    return results_list


# ─────────────────────────────────────────────────────────────────────────────
# Save/Load Results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results_list, filepath: str):
    """Save sweep results to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(results_list, f)
    print(f"[grid_search] Saved {len(results_list):,} results to {filepath}")


def load_results(filepath: str):
    """Load sweep results from disk."""
    with open(filepath, 'rb') as f:
        results_list = pickle.load(f)
    print(f"[grid_search] Loaded {len(results_list):,} results from {filepath}")
    return results_list


def print_top_results(results_list, top_n: int = 20, phase: str = ""):
    """Pretty-print the top N results."""
    print(f"\n{'='*100}")
    print(f"  TOP {top_n} RESULTS {phase}")
    print(f"{'='*100}")
    
    for i, (params, results, rank) in enumerate(results_list[:top_n]):
        trades = int(results[R_TOTAL_TRADES])
        wr = results[R_WIN_RATE]
        pf = results[R_PROFIT_FACTOR]
        dd = results[R_MAX_DRAWDOWN_PCT]
        sharpe = results[R_SHARPE]
        exp = results[R_EXPECTANCY]
        net = results[R_NET_PROFIT]
        calmar = results[R_CALMAR]
        
        print(f"\n  #{i+1} | Rank: {rank:.1f}")
        print(f"    Trades: {trades} | WR: {wr:.1f}% | PF: {pf:.2f} | "
              f"DD: {dd:.1f}% | Sharpe: {sharpe:.2f} | Exp: {exp:.3f} ATR | Calmar: {calmar:.2f}")
        
        if isinstance(params, SignalParams):
            print(f"    T3: L{params.t3_slow_len}/F{params.t3_fast_len}/S{params.t3_sensitivity} | "
                  f"ADX: {params.adx_len}/{params.adx_threshold} | "
                  f"RSI: {params.rsi_oversold}-{params.rsi_overbought} | "
                  f"Quality: {params.min_quality_score} | "
                  f"Strength: {params.min_trend_strength} | "
                  f"Regime: T{params.trend_metric_thresh}/V{params.volatile_atr_thresh}")
        elif isinstance(params, np.ndarray):
            from .backtest_engine import (P_ATR_STOP_MULT, P_TRAIL_ACTIVATE_ATR,
                P_TRAIL_OFFSET_ATR, P_BREAKEVEN_ATR, P_PSAR_EXIT_ENABLED,
                P_PSAR_MATURITY_PCT, P_SWING_LOOKBACK)
            print(f"    HardStop: {params[P_ATR_STOP_MULT]:.1f} | "
                  f"Trail: act={params[P_TRAIL_ACTIVATE_ATR]:.1f} off={params[P_TRAIL_OFFSET_ATR]:.1f} | "
                  f"BE: {params[P_BREAKEVEN_ATR]:.1f} | "
                  f"PSAR: {'ON' if params[P_PSAR_EXIT_ENABLED] > 0.5 else 'OFF'} "
                  f"mat={params[P_PSAR_MATURITY_PCT]:.1f} | "
                  f"Swing: {int(params[P_SWING_LOOKBACK])}")
    
    print(f"\n{'='*100}")

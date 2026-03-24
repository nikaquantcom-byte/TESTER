"""
Module 5 — Walk-Forward Validation
Uses VectorBT Pro's cv_split for anchored walk-forward analysis.
Prevents overfitting by training on in-sample, validating on out-of-sample.

Walk-forward approach:
  1. Split data into rolling windows (e.g., 6 months IS, 2 months OOS)
  2. For each fold: optimize on IS, test best params on OOS
  3. Aggregate OOS results to get realistic performance estimate
  4. Compare IS vs OOS degradation — flag overfitting
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

from .signals import SignalParams, generate_signals
from .backtest_engine import (
    run_backtest, run_backtest_with_equity, make_trade_params,
    NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_SHARPE, R_EXPECTANCY, R_NET_PROFIT, R_RECOVERY_FACTOR,
)
from .grid_search import (
    SignalGrid, TradeGrid, compute_rank_score, passes_gates,
    generate_signal_combos, generate_trade_combos,
)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Window sizes in bars (M10 = 6 bars/hour, 144 bars/day)
    is_window_bars: int = 144 * 126    # ~6 months in-sample
    oos_window_bars: int = 144 * 42    # ~2 months out-of-sample
    step_bars: int = 144 * 42          # Step forward by OOS size (no overlap)
    
    # Minimum OOS performance relative to IS
    min_oos_is_ratio: float = 0.50     # OOS rank must be >= 50% of IS rank
    
    # Anchored: IS window grows over time (starts from beginning of data)
    anchored: bool = False
    
    # Number of top IS combos to test on OOS (more = slower but more robust)
    top_n_for_oos: int = 5


def create_wf_splits(
    n_bars: int,
    config: WalkForwardConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    Create walk-forward splits.
    Returns list of (is_start, is_end, oos_start, oos_end) tuples.
    """
    splits = []
    
    if config.anchored:
        # Anchored: IS always starts at 0, grows each fold
        oos_start = config.is_window_bars
        while oos_start + config.oos_window_bars <= n_bars:
            is_start = 0
            is_end = oos_start
            oos_end = oos_start + config.oos_window_bars
            splits.append((is_start, is_end, oos_start, oos_end))
            oos_start += config.step_bars
    else:
        # Rolling: IS window moves with OOS
        pos = 0
        while pos + config.is_window_bars + config.oos_window_bars <= n_bars:
            is_start = pos
            is_end = pos + config.is_window_bars
            oos_start = is_end
            oos_end = is_end + config.oos_window_bars
            splits.append((is_start, is_end, oos_start, oos_end))
            pos += config.step_bars
    
    return splits


def _quick_signal_backtest(
    ohlcv_data: Dict[str, np.ndarray],
    signal_params: SignalParams,
    trade_params: np.ndarray,
    start_idx: int,
    end_idx: int,
    htf_close_1h: np.ndarray = None,
    htf_close_4h: np.ndarray = None,
    swing_lookback: int = 10,
) -> np.ndarray:
    """Run signal generation + backtest for a single param set on a data window."""
    sig = generate_signals(
        ohlcv_data['open'], ohlcv_data['high'], ohlcv_data['low'],
        ohlcv_data['close'], ohlcv_data['volume'],
        signal_params,
        htf_close_1h=htf_close_1h,
        htf_close_4h=htf_close_4h,
        swing_lookback=swing_lookback,
    )
    
    results = run_backtest(
        ohlcv_data['open'], ohlcv_data['high'], ohlcv_data['low'],
        ohlcv_data['close'],
        sig.t3_trend, sig.atr,
        sig.psar_flip_up, sig.psar_flip_down,
        sig.swing_high, sig.swing_low,
        sig.signal_buy, sig.signal_sell,
        trade_params,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    
    return results


def run_walk_forward(
    ohlcv_data: Dict[str, np.ndarray],
    candidate_signal_params: List[SignalParams],
    trade_params: np.ndarray,
    config: WalkForwardConfig = None,
    htf_close_1h: np.ndarray = None,
    htf_close_4h: np.ndarray = None,
    swing_lookback: int = 10,
) -> Dict:
    """
    Full walk-forward validation.
    
    For each fold:
      1. Test all candidate signal params on IS window
      2. Pick top N by rank
      3. Test those top N on OOS window
      4. Record IS and OOS performance
    
    Args:
        ohlcv_data: dict with numpy arrays
        candidate_signal_params: list of SignalParams to test (typically top 50-100 from grid search)
        trade_params: fixed trade params (from Phase 2 best)
        config: walk-forward configuration
    
    Returns: Dict with fold-by-fold results and aggregates
    """
    if config is None:
        config = WalkForwardConfig()
    
    n_bars = len(ohlcv_data['close'])
    splits = create_wf_splits(n_bars, config)
    n_folds = len(splits)
    
    print(f"\n[walk_forward] {n_folds} folds | IS: {config.is_window_bars:,} bars | "
          f"OOS: {config.oos_window_bars:,} bars | Step: {config.step_bars:,} bars")
    print(f"[walk_forward] Testing {len(candidate_signal_params):,} candidate param sets per fold")
    
    fold_results = []
    oos_equity_segments = []
    
    t0 = time.time()
    
    for fold_idx, (is_start, is_end, oos_start, oos_end) in enumerate(splits):
        fold_t0 = time.time()
        
        # ── In-Sample: Test all candidates ──
        is_rankings = []
        for sp in candidate_signal_params:
            is_res = _quick_signal_backtest(
                ohlcv_data, sp, trade_params,
                is_start, is_end,
                htf_close_1h, htf_close_4h, swing_lookback,
            )
            is_rank = compute_rank_score(is_res)
            is_rankings.append((sp, is_res, is_rank))
        
        # Sort by IS rank
        is_rankings.sort(key=lambda x: x[2], reverse=True)
        
        # ── Out-of-Sample: Test top N from IS ──
        top_n = min(config.top_n_for_oos, len(is_rankings))
        oos_rankings = []
        
        for j in range(top_n):
            sp, is_res, is_rank = is_rankings[j]
            
            oos_res = _quick_signal_backtest(
                ohlcv_data, sp, trade_params,
                oos_start, oos_end,
                htf_close_1h, htf_close_4h, swing_lookback,
            )
            oos_rank = compute_rank_score(oos_res)
            
            oos_rankings.append({
                'signal_params': sp,
                'is_results': is_res,
                'is_rank': is_rank,
                'oos_results': oos_res,
                'oos_rank': oos_rank,
                'oos_is_ratio': oos_rank / max(is_rank, 0.01),
            })
        
        # Best OOS performer for this fold
        oos_rankings.sort(key=lambda x: x['oos_rank'], reverse=True)
        best_oos = oos_rankings[0]
        
        fold_elapsed = time.time() - fold_t0
        
        fold_info = {
            'fold': fold_idx + 1,
            'is_range': (is_start, is_end),
            'oos_range': (oos_start, oos_end),
            'best_is_rank': is_rankings[0][2],
            'best_oos_rank': best_oos['oos_rank'],
            'best_oos_params': best_oos['signal_params'],
            'best_oos_results': best_oos['oos_results'],
            'oos_is_ratio': best_oos['oos_is_ratio'],
            'all_oos': oos_rankings,
            'elapsed': fold_elapsed,
        }
        
        fold_results.append(fold_info)
        
        # Print fold summary
        oos_pf = best_oos['oos_results'][R_PROFIT_FACTOR]
        oos_wr = best_oos['oos_results'][R_WIN_RATE]
        oos_trades = int(best_oos['oos_results'][R_TOTAL_TRADES])
        is_pf = best_oos['is_results'][R_PROFIT_FACTOR]
        
        overfitting = "⚠ OVERFIT" if best_oos['oos_is_ratio'] < config.min_oos_is_ratio else "✓ OK"
        
        print(f"  Fold {fold_idx+1}/{n_folds}: "
              f"IS PF={is_pf:.2f} → OOS PF={oos_pf:.2f} | "
              f"OOS WR={oos_wr:.1f}% ({oos_trades} trades) | "
              f"OOS/IS ratio={best_oos['oos_is_ratio']:.2f} {overfitting} | "
              f"{fold_elapsed:.1f}s")
    
    elapsed = time.time() - t0
    
    # ── Aggregate OOS metrics ──
    oos_pfs = [f['best_oos_results'][R_PROFIT_FACTOR] for f in fold_results]
    oos_wrs = [f['best_oos_results'][R_WIN_RATE] for f in fold_results]
    oos_dds = [f['best_oos_results'][R_MAX_DRAWDOWN_PCT] for f in fold_results]
    oos_trades = [f['best_oos_results'][R_TOTAL_TRADES] for f in fold_results]
    oos_is_ratios = [f['oos_is_ratio'] for f in fold_results]
    
    # Count how many folds were profitable
    profitable_folds = sum(1 for f in fold_results if f['best_oos_results'][R_NET_PROFIT] > 0)
    
    # Find most consistently selected params
    param_frequency = {}
    for f in fold_results:
        sp = f['best_oos_params']
        key = (sp.t3_slow_len, sp.t3_sensitivity, sp.adx_threshold, sp.min_quality_score)
        param_frequency[key] = param_frequency.get(key, 0) + 1
    
    most_common_params = sorted(param_frequency.items(), key=lambda x: x[1], reverse=True)
    
    summary = {
        'n_folds': n_folds,
        'fold_results': fold_results,
        'elapsed': elapsed,
        'profitable_folds': profitable_folds,
        'profitable_pct': profitable_folds / max(n_folds, 1) * 100.0,
        'avg_oos_pf': np.mean(oos_pfs),
        'median_oos_pf': np.median(oos_pfs),
        'avg_oos_wr': np.mean(oos_wrs),
        'avg_oos_dd': np.mean(oos_dds),
        'total_oos_trades': sum(oos_trades),
        'avg_oos_is_ratio': np.mean(oos_is_ratios),
        'overfitting_folds': sum(1 for r in oos_is_ratios if r < config.min_oos_is_ratio),
        'most_common_params': most_common_params[:5],
    }
    
    # ── Print Summary ──
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'='*80}")
    print(f"  Folds: {n_folds} | Profitable: {profitable_folds}/{n_folds} "
          f"({summary['profitable_pct']:.0f}%)")
    print(f"  Avg OOS PF:  {summary['avg_oos_pf']:.2f} (median: {summary['median_oos_pf']:.2f})")
    print(f"  Avg OOS WR:  {summary['avg_oos_wr']:.1f}%")
    print(f"  Avg OOS DD:  {summary['avg_oos_dd']:.1f}%")
    print(f"  Avg OOS/IS:  {summary['avg_oos_is_ratio']:.2f} "
          f"({'ROBUST' if summary['avg_oos_is_ratio'] >= 0.60 else 'CAUTION' if summary['avg_oos_is_ratio'] >= 0.40 else 'OVERFIT'})")
    print(f"  Total OOS trades: {summary['total_oos_trades']:,}")
    print(f"  Overfitting folds: {summary['overfitting_folds']}/{n_folds}")
    print(f"  Elapsed: {elapsed:.1f}s")
    
    if most_common_params:
        print(f"\n  Most frequently selected params (across folds):")
        for params_key, count in most_common_params[:3]:
            print(f"    T3 L{params_key[0]} S{params_key[1]} | ADX>{params_key[2]} | "
                  f"Quality>{params_key[3]} → selected {count}/{n_folds} folds")
    
    print(f"{'='*80}")
    
    return summary


def run_walk_forward_vbt(
    ohlcv_df: pd.DataFrame,
    candidate_signal_params: List[SignalParams],
    trade_params: np.ndarray,
    n_splits: int = 8,
    is_ratio: float = 0.75,
    htf_1h_df: pd.DataFrame = None,
    htf_4h_df: pd.DataFrame = None,
    swing_lookback: int = 10,
) -> Dict:
    """
    Walk-forward using VectorBT Pro's cv_split if available.
    Falls back to manual splitting if VBT Pro is not importable.
    
    This is the preferred method when VBT Pro is installed.
    """
    try:
        import vectorbtpro as vbt
        
        print(f"\n[walk_forward] Using VectorBT Pro cv_split...")
        
        # Create split using VBT Pro
        splitter = vbt.Splitter.from_rolling(
            ohlcv_df.index,
            length=len(ohlcv_df) // (n_splits + 1) * 2,
            split=is_ratio,
            set_labels=["IS", "OOS"],
        )
        
        print(f"[walk_forward] VBT Pro created {splitter.n_splits} splits")
        
        # Convert to our format
        ohlcv_data = {
            'open': ohlcv_df['open'].values.astype(np.float64),
            'high': ohlcv_df['high'].values.astype(np.float64),
            'low': ohlcv_df['low'].values.astype(np.float64),
            'close': ohlcv_df['close'].values.astype(np.float64),
            'volume': ohlcv_df['volume'].values.astype(np.float64),
        }
        
        htf_1h_close = htf_1h_df['close'].values.astype(np.float64) if htf_1h_df is not None else None
        htf_4h_close = htf_4h_df['close'].values.astype(np.float64) if htf_4h_df is not None else None
        
        fold_results = []
        
        for split_idx in range(splitter.n_splits):
            is_mask = splitter.get_mask(split_idx, "IS")
            oos_mask = splitter.get_mask(split_idx, "OOS")
            
            is_indices = np.where(is_mask)[0]
            oos_indices = np.where(oos_mask)[0]
            
            if len(is_indices) == 0 or len(oos_indices) == 0:
                continue
            
            is_start, is_end = is_indices[0], is_indices[-1] + 1
            oos_start, oos_end = oos_indices[0], oos_indices[-1] + 1
            
            # IS optimization
            is_rankings = []
            for sp in candidate_signal_params:
                is_res = _quick_signal_backtest(
                    ohlcv_data, sp, trade_params,
                    is_start, is_end,
                    htf_1h_close, htf_4h_close, swing_lookback,
                )
                is_rank = compute_rank_score(is_res)
                is_rankings.append((sp, is_res, is_rank))
            
            is_rankings.sort(key=lambda x: x[2], reverse=True)
            
            # OOS test with best IS params
            best_sp = is_rankings[0][0]
            oos_res = _quick_signal_backtest(
                ohlcv_data, best_sp, trade_params,
                oos_start, oos_end,
                htf_1h_close, htf_4h_close, swing_lookback,
            )
            
            fold_results.append({
                'fold': split_idx + 1,
                'is_range': (is_start, is_end),
                'oos_range': (oos_start, oos_end),
                'best_is_rank': is_rankings[0][2],
                'best_oos_results': oos_res,
                'best_oos_params': best_sp,
            })
            
            print(f"  Split {split_idx+1}: IS PF={is_rankings[0][1][R_PROFIT_FACTOR]:.2f} → "
                  f"OOS PF={oos_res[R_PROFIT_FACTOR]:.2f}")
        
        return {'fold_results': fold_results, 'method': 'vbt_pro'}
    
    except ImportError:
        print("[walk_forward] VectorBT Pro not available — using manual splits")
        ohlcv_data = {
            'open': ohlcv_df['open'].values.astype(np.float64),
            'high': ohlcv_df['high'].values.astype(np.float64),
            'low': ohlcv_df['low'].values.astype(np.float64),
            'close': ohlcv_df['close'].values.astype(np.float64),
            'volume': ohlcv_df['volume'].values.astype(np.float64),
        }
        
        htf_1h_close = htf_1h_df['close'].values.astype(np.float64) if htf_1h_df is not None else None
        htf_4h_close = htf_4h_df['close'].values.astype(np.float64) if htf_4h_df is not None else None
        
        config = WalkForwardConfig()
        return run_walk_forward(
            ohlcv_data, candidate_signal_params, trade_params,
            config=config,
            htf_close_1h=htf_1h_close,
            htf_close_4h=htf_4h_close,
            swing_lookback=swing_lookback,
        )

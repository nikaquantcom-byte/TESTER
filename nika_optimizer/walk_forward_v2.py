"""
Module 4 — Walk-Forward Validation V2 (MEGA Optimizer)
9mo IS / 3mo OOS, 1-day embargo, frozen 2-month holdout, plateau scoring.
"""

import numpy as np
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass

from .signals_v2 import T3Params, IndicatorParams, ConfluenceConfig, generate_signals, pack_indicator_bits
from .backtest_engine_v2 import (
    run_backtest, make_trade_params, NUM_RESULTS,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_NET_PROFIT, R_EXPECTANCY, R_SHARPE,
)
from .grid_search_v2 import compute_composite_score


BARS_PER_DAY = 144  # M10 bars

@dataclass
class WFConfig:
    is_days: int = 180          # 9 months ≈ 180 trading days
    oos_days: int = 60          # 3 months ≈ 60 trading days
    step_days: int = 60         # step by OOS size
    embargo_days: int = 1       # 1-day gap between IS and OOS
    holdout_days: int = 40      # ~2 months frozen holdout at end
    top_n_for_oos: int = 5      # top IS configs tested on OOS
    min_oos_pf: float = 0.90    # minimum OOS PF to not be flagged


def create_splits(n_bars, config):
    """Create walk-forward splits with embargo."""
    holdout_start = n_bars - config.holdout_days * BARS_PER_DAY
    embargo = config.embargo_days * BARS_PER_DAY
    is_bars = config.is_days * BARS_PER_DAY
    oos_bars = config.oos_days * BARS_PER_DAY
    step = config.step_days * BARS_PER_DAY

    splits = []
    pos = 0
    while pos + is_bars + embargo + oos_bars <= holdout_start:
        is_start = pos
        is_end = pos + is_bars
        oos_start = is_end + embargo
        oos_end = oos_start + oos_bars
        if oos_end <= holdout_start:
            splits.append((is_start, is_end, oos_start, oos_end))
        pos += step

    return splits, holdout_start


def _run_one_backtest(ohlcv, t3p, indp, conf, tp, start, end, htf_1h=None, htf_4h=None):
    """Run signals + backtest for one config on one data window."""
    sig = generate_signals(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'],
        t3p, indp, htf_1h, htf_4h,
    )
    ibb = pack_indicator_bits(sig.ind_buy)
    ibs = pack_indicator_bits(sig.ind_sell)

    return run_backtest(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], sig.open_next,
        sig.t3_trend, sig.raw_flip_up, sig.raw_flip_down, sig.atr,
        ibb, ibs, conf.indicator_mask, conf.min_agree,
        sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
        tp, start, end,
    )


def run_walk_forward(
    ohlcv: Dict,
    candidates: List[Tuple],  # list of (t3p, conf, tp) tuples
    config: WFConfig = None,
    htf_1h=None, htf_4h=None,
) -> Dict:
    """Full walk-forward validation with embargo and frozen holdout."""
    if config is None: config = WFConfig()

    n_bars = len(ohlcv['close'])
    splits, holdout_start = create_splits(n_bars, config)
    n_folds = len(splits)

    print(f"\n[Walk-Forward] {n_folds} folds | IS: {config.is_days}d | OOS: {config.oos_days}d | "
          f"Embargo: {config.embargo_days}d | Holdout: last {config.holdout_days}d")
    print(f"  Testing {len(candidates)} candidates per fold")

    fold_results = []
    t0 = time.time()

    for fold_idx, (is_start, is_end, oos_start, oos_end) in enumerate(splits):
        fold_t0 = time.time()

        # IS: test all candidates
        is_rankings = []
        for t3p, conf, tp in candidates:
            indp = IndicatorParams()
            is_res = _run_one_backtest(ohlcv, t3p, indp, conf, tp, is_start, is_end, htf_1h, htf_4h)
            is_score = compute_composite_score(is_res)
            is_rankings.append((t3p, conf, tp, is_res, is_score))

        is_rankings.sort(key=lambda x: x[4], reverse=True)

        # OOS: test top N from IS
        top_n = min(config.top_n_for_oos, len(is_rankings))
        oos_results = []
        for j in range(top_n):
            t3p, conf, tp, is_res, is_score = is_rankings[j]
            indp = IndicatorParams()
            oos_res = _run_one_backtest(ohlcv, t3p, indp, conf, tp, oos_start, oos_end, htf_1h, htf_4h)
            oos_score = compute_composite_score(oos_res)
            oos_results.append({
                't3p': t3p, 'conf': conf, 'tp': tp,
                'is_res': is_res, 'is_score': is_score,
                'oos_res': oos_res, 'oos_score': oos_score,
                'oos_is_ratio': oos_score / max(is_score, 0.01),
            })

        oos_results.sort(key=lambda x: x['oos_score'], reverse=True)
        best = oos_results[0]

        fold_info = {
            'fold': fold_idx + 1,
            'is_range': (is_start, is_end),
            'oos_range': (oos_start, oos_end),
            'best': best,
            'all_oos': oos_results,
        }
        fold_results.append(fold_info)

        oos_pf = best['oos_res'][R_PROFIT_FACTOR]
        is_pf = best['is_res'][R_PROFIT_FACTOR]
        oos_trades = int(best['oos_res'][R_TOTAL_TRADES])
        flag = "⚠ OVERFIT" if best['oos_is_ratio'] < 0.4 else "✓"

        print(f"  Fold {fold_idx+1}/{n_folds}: IS PF={is_pf:.2f} → OOS PF={oos_pf:.2f} | "
              f"OOS trades={oos_trades} | ratio={best['oos_is_ratio']:.2f} {flag} | "
              f"{time.time()-fold_t0:.1f}s")

    # ── Frozen holdout ──
    print(f"\n[Walk-Forward] Running frozen holdout (last {config.holdout_days} days)...")
    holdout_results = []
    # Use the most frequently selected config across folds
    from collections import Counter
    config_counts = Counter()
    for f in fold_results:
        b = f['best']
        key = (b['t3p'], b['conf'])
        config_counts[key] += 1

    top_holdout_configs = config_counts.most_common(5)
    for (t3p, conf), count in top_holdout_configs:
        tp = make_trade_params(spread_points=0.30)  # default management
        indp = IndicatorParams()
        ho_res = _run_one_backtest(ohlcv, t3p, indp, conf, tp, holdout_start, n_bars, htf_1h, htf_4h)
        holdout_results.append({
            't3p': t3p, 'conf': conf, 'res': ho_res,
            'fold_count': count,
        })
        ho_pf = ho_res[R_PROFIT_FACTOR]
        ho_trades = int(ho_res[R_TOTAL_TRADES])
        print(f"  Holdout: T3 L{t3p.slow_len} TF={t3p.tfactor:.2f} S={t3p.sensitivity} | "
              f"PF={ho_pf:.2f} | trades={ho_trades} | selected {count}/{n_folds} folds")

    # ── Summary ──
    elapsed = time.time() - t0
    oos_pfs = [f['best']['oos_res'][R_PROFIT_FACTOR] for f in fold_results]
    oos_wrs = [f['best']['oos_res'][R_WIN_RATE] for f in fold_results]
    oos_ratios = [f['best']['oos_is_ratio'] for f in fold_results]
    profitable = sum(1 for f in fold_results if f['best']['oos_res'][R_NET_PROFIT] > 0)

    summary = {
        'n_folds': n_folds,
        'fold_results': fold_results,
        'holdout_results': holdout_results,
        'holdout_start': holdout_start,
        'profitable_folds': profitable,
        'profitable_pct': profitable / max(n_folds, 1) * 100.0,
        'avg_oos_pf': np.mean(oos_pfs) if oos_pfs else 0,
        'median_oos_pf': np.median(oos_pfs) if oos_pfs else 0,
        'avg_oos_wr': np.mean(oos_wrs) if oos_wrs else 0,
        'avg_oos_is_ratio': np.mean(oos_ratios) if oos_ratios else 0,
        'elapsed': elapsed,
    }

    robustness = "ROBUST" if summary['avg_oos_is_ratio'] >= 0.60 else \
                 "MODERATE" if summary['avg_oos_is_ratio'] >= 0.40 else "OVERFIT RISK"

    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'='*80}")
    print(f"  Folds: {n_folds} | Profitable: {profitable}/{n_folds} ({summary['profitable_pct']:.0f}%)")
    print(f"  Avg OOS PF: {summary['avg_oos_pf']:.2f} (median: {summary['median_oos_pf']:.2f})")
    print(f"  Avg OOS/IS ratio: {summary['avg_oos_is_ratio']:.2f} → {robustness}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*80}")

    return summary


def compute_plateau_score(
    target_results: np.ndarray,
    neighbor_results: List[np.ndarray],
) -> float:
    """
    Check if a config sits on a stable plateau.
    Returns: median(neighbor PFs) - 0.5 * std(neighbor PFs)
    Higher = more robust plateau.
    """
    if not neighbor_results:
        return 0.0
    pfs = [r[R_PROFIT_FACTOR] for r in neighbor_results]
    return np.median(pfs) - 0.5 * np.std(pfs)

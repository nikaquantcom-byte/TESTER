"""
Module 4 — Walk-Forward Validation V3 (MEGA Optimizer V3 compatible)

FIXES vs walk_forward_v2.py:
  1. WFConfig uses n_splits / train_pct / min_trades (matches run_mega_v3 call)
  2. run_walk_forward accepts single (eng, conf, tp) EngineConfig tuples — NOT T3Params
  3. Uses signals_v3.generate_universal_signals + precompute_shared_indicators
  4. Uses backtest_engine_v2.run_backtest (unchanged)
  5. Embargo between IS and OOS windows
  6. Frozen holdout (last 10% of data — never touched during optimisation)

WFConfig fields
---------------
  n_splits     : int   — number of rolling folds (default 5)
  train_pct    : float — fraction of each window used as IS (default 0.70)
  min_trades   : int   — minimum OOS trades for fold to be considered valid (default 10)
  embargo_bars : int   — bars gap between IS end and OOS start (default 144 = 1 day M10)
  holdout_pct  : float — fraction of total bars frozen as holdout (default 0.10)
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from .signals_v3 import (
    EngineConfig, ConfluenceConfig,
    generate_universal_signals, precompute_shared_indicators,
    pack_indicator_bits, apply_confluence,
    NUM_IND,
)
from .backtest_engine_v2 import (
    run_backtest, make_trade_params, NUM_RESULTS,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT,
    R_TOTAL_TRADES, R_NET_PROFIT, R_EXPECTANCY, R_SHARPE,
)


BARS_PER_DAY_M10 = 144  # M10: 6 bars/hr * 24 hrs


@dataclass
class WFConfig:
    """Walk-forward configuration — compatible with run_mega_v3_fixed call."""
    n_splits: int = 5           # rolling folds
    train_pct: float = 0.70     # IS fraction of each window
    min_trades: int = 10        # minimum OOS trades for a fold to count
    embargo_bars: int = 144     # 1 day gap between IS and OOS (M10)
    holdout_pct: float = 0.10   # frozen holdout at end of data


def _make_splits(n_bars: int, cfg: WFConfig) -> Tuple[List[Tuple], int]:
    """
    Create rolling IS/OOS splits with embargo and frozen holdout.

    Returns
    -------
    splits       : list of (is_start, is_end, oos_start, oos_end)
    holdout_start: first bar of the frozen holdout period
    """
    holdout_start = int(n_bars * (1.0 - cfg.holdout_pct))
    available = holdout_start  # only non-holdout bars are used for WF

    # Window size: IS + embargo + OOS fill the window
    # We slice available into n_splits overlapping windows
    # Each window = window_size bars, stepped by OOS size
    # OOS = (1 - train_pct) * window_size
    # IS  = train_pct * window_size

    # Solve: n_splits windows fit in available with step = oos_size
    # window_size + (n_splits - 1) * oos_size <= available
    # window_size = is_size + embargo + oos_size
    # Let oos_size = x  =>  is_size = train_pct/(1-train_pct) * x
    # window = is_size + embargo + x
    # is_size + embargo + x + (n_splits - 1) * x <= available
    # is_size + embargo + n_splits * x <= available

    oos_fraction = 1.0 - cfg.train_pct
    is_fraction = cfg.train_pct
    ratio = is_fraction / oos_fraction  # is_size = ratio * oos_size

    # oos_size * (ratio + 1 + (n_splits - 1)) + embargo <= available
    # oos_size * (ratio + n_splits) <= available - embargo
    oos_size = int((available - cfg.embargo_bars) / (ratio + cfg.n_splits))
    is_size = int(oos_size * ratio)

    if is_size < 1 or oos_size < 1:
        raise ValueError(
            f"Not enough bars for {cfg.n_splits} folds. "
            f"Need at least {cfg.n_splits * 2 * BARS_PER_DAY_M10} bars."
        )

    splits = []
    for i in range(cfg.n_splits):
        is_start = i * oos_size
        is_end = is_start + is_size
        oos_start = is_end + cfg.embargo_bars
        oos_end = oos_start + oos_size
        if oos_end > holdout_start:
            break
        splits.append((is_start, is_end, oos_start, oos_end))

    return splits, holdout_start


def _run_config_on_window(
    ohlcv: Dict,
    eng: EngineConfig,
    conf: ConfluenceConfig,
    tp: np.ndarray,
    start: int,
    end: int,
    shared=None,
) -> np.ndarray:
    """
    Generate signals + run backtest for one config on one data window.
    Uses precomputed shared indicators if provided (faster in fold loops).
    """
    if shared is None:
        shared = precompute_shared_indicators(
            ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'], eng
        )

    sig = generate_universal_signals(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'],
        eng,
        ohlcv.get('htf_1h'), ohlcv.get('htf_4h'),
        shared=shared,
    )

    # Build t3_trend array (running direction tracker)
    n = len(ohlcv['close'])
    t3_trend = np.zeros(n, dtype=np.int32)
    current = 0
    for i in range(n):
        if sig.buy[i]:  current = 1
        elif sig.sell[i]: current = -1
        t3_trend[i] = current

    ibb = pack_indicator_bits(sig.ind_buy)
    ibs = pack_indicator_bits(sig.ind_sell)

    if conf.indicator_mask > 0:
        buy_f, sell_f = apply_confluence(sig.buy, sig.sell, sig.ind_buy, sig.ind_sell, conf)
    else:
        buy_f, sell_f = sig.buy, sig.sell

    return run_backtest(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], sig.open_next,
        t3_trend, buy_f, sell_f, sig.atr,
        ibb, ibs, conf.indicator_mask, conf.min_agree,
        sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
        tp, start, end,
    )


def _composite_score(res: np.ndarray) -> float:
    """Simple composite score for IS ranking within WF."""
    pf = res[R_PROFIT_FACTOR]
    wr = res[R_WIN_RATE] / 100.0
    trades = res[R_TOTAL_TRADES]
    dd = abs(res[R_MAX_DRAWDOWN_PCT])
    if trades < 5 or pf < 1.0:
        return 0.0
    tc = np.log1p(trades) / np.log1p(100)  # confidence: log-scaled, caps around 100 trades
    perf = pf * (1.0 + wr)
    stab = 1.0 / max(dd, 0.5)
    return perf ** 0.45 * stab ** 0.30 * tc ** 0.25


def run_walk_forward(
    ohlcv: Dict,
    eng: EngineConfig,
    conf: ConfluenceConfig,
    tp: np.ndarray,
    cfg: WFConfig = None,
) -> Dict:
    """
    Full walk-forward validation with embargo and frozen holdout.

    Parameters
    ----------
    ohlcv : dict with keys open/high/low/close/volume (and optionally htf_1h, htf_4h)
    eng   : EngineConfig (from signals_v3)
    conf  : ConfluenceConfig
    tp    : trade params array (from make_trade_params)
    cfg   : WFConfig — if None, uses defaults

    Returns
    -------
    dict with keys:
        is_results   : list of np.ndarray (one per fold, IS backtest results)
        oos_results  : list of np.ndarray (one per fold, OOS backtest results)
        fold_info    : list of dicts with fold ranges and metrics
        holdout_res  : np.ndarray (holdout period results)
        summary      : dict with avg OOS PF, robustness, etc.
    """
    if cfg is None:
        cfg = WFConfig()

    n_bars = len(ohlcv['close'])
    splits, holdout_start = _make_splits(n_bars, cfg)
    n_folds = len(splits)

    if n_folds == 0:
        raise ValueError(f"No valid WF folds created. Check data length ({n_bars} bars) vs n_splits={cfg.n_splits}.")

    print(f"    WF: {n_folds} folds | IS: {splits[0][1]-splits[0][0]} bars | OOS: {splits[0][3]-splits[0][2]} bars | Embargo: {cfg.embargo_bars} bars")

    # Precompute shared indicators ONCE for full dataset
    shared = precompute_shared_indicators(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'], eng
    )

    is_results = []
    oos_results = []
    fold_info = []
    t0 = time.time()

    for fold_idx, (is_start, is_end, oos_start, oos_end) in enumerate(splits):
        # IS backtest
        is_res = _run_config_on_window(ohlcv, eng, conf, tp, is_start, is_end, shared)
        # OOS backtest
        oos_res = _run_config_on_window(ohlcv, eng, conf, tp, oos_start, oos_end, shared)

        is_pf = is_res[R_PROFIT_FACTOR]
        oos_pf = oos_res[R_PROFIT_FACTOR]
        oos_trades = int(oos_res[R_TOTAL_TRADES])
        oos_wr = oos_res[R_WIN_RATE]
        ratio = oos_pf / max(is_pf, 0.01)

        valid = oos_trades >= cfg.min_trades
        flag = "" if valid else " [low trades]"
        status = "✅" if oos_pf >= 1.0 and ratio >= 0.5 else "⚠️" if oos_pf >= 1.0 else "❌"

        print(f"    Fold {fold_idx+1}/{n_folds}: IS PF={is_pf:.2f} → OOS PF={oos_pf:.2f} "
              f"| WR={oos_wr:.1f}% | Trades={oos_trades}{flag} | ratio={ratio:.2f} {status}")

        is_results.append(is_res)
        oos_results.append(oos_res)
        fold_info.append({
            'fold': fold_idx + 1,
            'is_start': is_start, 'is_end': is_end,
            'oos_start': oos_start, 'oos_end': oos_end,
            'is_pf': is_pf, 'oos_pf': oos_pf,
            'oos_wr': oos_wr, 'oos_trades': oos_trades,
            'ratio': ratio, 'valid': valid,
        })

    # Frozen holdout
    holdout_res = _run_config_on_window(ohlcv, eng, conf, tp, holdout_start, n_bars, shared)
    ho_pf = holdout_res[R_PROFIT_FACTOR]
    ho_trades = int(holdout_res[R_TOTAL_TRADES])
    print(f"    Holdout: PF={ho_pf:.2f} | Trades={ho_trades} | bars={n_bars-holdout_start}")

    # Summary
    valid_folds = [f for f in fold_info if f['valid']]
    oos_pfs = [f['oos_pf'] for f in valid_folds]
    oos_ratios = [f['ratio'] for f in valid_folds]
    profitable_folds = sum(1 for f in valid_folds if f['oos_pf'] >= 1.0)

    avg_oos_pf = float(np.mean(oos_pfs)) if oos_pfs else 0.0
    avg_ratio = float(np.mean(oos_ratios)) if oos_ratios else 0.0
    pct_profitable = profitable_folds / max(len(valid_folds), 1) * 100.0

    robustness_label = (
        "ROBUST" if avg_ratio >= 0.60 and pct_profitable >= 80.0
        else "MODERATE" if avg_ratio >= 0.40 and pct_profitable >= 60.0
        else "OVERFIT RISK"
    )

    summary = {
        'n_folds': n_folds,
        'valid_folds': len(valid_folds),
        'profitable_folds': profitable_folds,
        'profitable_pct': pct_profitable,
        'avg_oos_pf': avg_oos_pf,
        'avg_oos_is_ratio': avg_ratio,
        'holdout_pf': ho_pf,
        'holdout_trades': ho_trades,
        'robustness': robustness_label,
        'elapsed': time.time() - t0,
    }

    return {
        'is_results': is_results,
        'oos_results': oos_results,
        'fold_info': fold_info,
        'holdout_res': holdout_res,
        'summary': summary,
    }

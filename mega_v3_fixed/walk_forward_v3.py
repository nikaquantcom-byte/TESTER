"""
Module 4 — Walk-Forward Validation V3 (MEGA Optimizer V3 compatible)

FIXES vs walk_forward_v2.py:
  1. WFConfig uses n_splits / train_pct / min_trades (matches run_mega_v3 call)
  2. run_walk_forward accepts single (eng, conf, tp) EngineConfig tuples — NOT T3Params
  3. Uses nika_optimizer.signals_v3 + nika_optimizer.backtest_engine_v2
  4. Embargo between IS and OOS windows
  5. Frozen holdout (last 10% of data — never touched during optimisation)
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

from nika_optimizer.signals_v3 import (
    EngineConfig, ConfluenceConfig,
    generate_universal_signals, precompute_shared_indicators,
    pack_indicator_bits, apply_confluence,
    NUM_IND,
)
from nika_optimizer.backtest_engine_v2 import (
    run_backtest, make_trade_params, NUM_RESULTS,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT,
    R_TOTAL_TRADES, R_NET_PROFIT, R_EXPECTANCY, R_SHARPE,
)


BARS_PER_DAY_M10 = 144


@dataclass
class WFConfig:
    n_splits: int = 5
    train_pct: float = 0.70
    min_trades: int = 10
    embargo_bars: int = 144
    holdout_pct: float = 0.10


def _make_splits(n_bars: int, cfg: WFConfig) -> Tuple[List[Tuple], int]:
    holdout_start = int(n_bars * (1.0 - cfg.holdout_pct))
    available = holdout_start
    oos_fraction = 1.0 - cfg.train_pct
    ratio = cfg.train_pct / oos_fraction
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
    n = len(ohlcv['close'])
    t3_trend = np.zeros(n, dtype=np.int32)
    current = 0
    for i in range(n):
        if sig.buy[i]: current = 1
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
    pf = res[R_PROFIT_FACTOR]
    wr = res[R_WIN_RATE] / 100.0
    trades = res[R_TOTAL_TRADES]
    dd = abs(res[R_MAX_DRAWDOWN_PCT])
    if trades < 5 or pf < 1.0:
        return 0.0
    tc = np.log1p(trades) / np.log1p(100)
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
    if cfg is None:
        cfg = WFConfig()
    n_bars = len(ohlcv['close'])
    splits, holdout_start = _make_splits(n_bars, cfg)
    n_folds = len(splits)
    if n_folds == 0:
        raise ValueError(f"No valid WF folds. Data: {n_bars} bars, n_splits={cfg.n_splits}.")
    print(f"    WF: {n_folds} folds | IS: {splits[0][1]-splits[0][0]} bars | OOS: {splits[0][3]-splits[0][2]} bars | Embargo: {cfg.embargo_bars} bars")
    shared = precompute_shared_indicators(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'], eng
    )
    is_results = []
    oos_results = []
    fold_info = []
    t0 = time.time()
    for fold_idx, (is_start, is_end, oos_start, oos_end) in enumerate(splits):
        is_res  = _run_config_on_window(ohlcv, eng, conf, tp, is_start, is_end, shared)
        oos_res = _run_config_on_window(ohlcv, eng, conf, tp, oos_start, oos_end, shared)
        is_pf  = is_res[R_PROFIT_FACTOR]
        oos_pf = oos_res[R_PROFIT_FACTOR]
        oos_trades = int(oos_res[R_TOTAL_TRADES])
        oos_wr = oos_res[R_WIN_RATE]
        ratio = oos_pf / max(is_pf, 0.01)
        valid = oos_trades >= cfg.min_trades
        flag = "" if valid else " [low trades]"
        status = "OK" if oos_pf >= 1.0 and ratio >= 0.5 else "WEAK" if oos_pf >= 1.0 else "FAIL"
        print(f"    Fold {fold_idx+1}/{n_folds}: IS PF={is_pf:.2f} -> OOS PF={oos_pf:.2f} | WR={oos_wr:.1f}% | Trades={oos_trades}{flag} | ratio={ratio:.2f} [{status}]")
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
    holdout_res = _run_config_on_window(ohlcv, eng, conf, tp, holdout_start, n_bars, shared)
    ho_pf = holdout_res[R_PROFIT_FACTOR]
    ho_trades = int(holdout_res[R_TOTAL_TRADES])
    print(f"    Holdout: PF={ho_pf:.2f} | Trades={ho_trades} | bars={n_bars - holdout_start}")
    valid_folds = [f for f in fold_info if f['valid']]
    oos_pfs    = [f['oos_pf'] for f in valid_folds]
    oos_ratios = [f['ratio']  for f in valid_folds]
    profitable_folds = sum(1 for f in valid_folds if f['oos_pf'] >= 1.0)
    avg_oos_pf  = float(np.mean(oos_pfs))    if oos_pfs    else 0.0
    avg_ratio   = float(np.mean(oos_ratios)) if oos_ratios else 0.0
    pct_profitable = profitable_folds / max(len(valid_folds), 1) * 100.0
    robustness_label = (
        "ROBUST"       if avg_ratio >= 0.60 and pct_profitable >= 80.0
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
        'is_results':  is_results,
        'oos_results': oos_results,
        'fold_info':   fold_info,
        'holdout_res': holdout_res,
        'summary':     summary,
    }

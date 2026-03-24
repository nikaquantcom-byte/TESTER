#!/usr/bin/env python
# run_overnight.py — Full T3 Pipeline: Phase 1 → 2 → 3 → 4
# Usage:
#   python run_overnight.py --data nika_optimizer/XAUUSD_M10.csv --cores 22
#
# Output folder: output_overnight/
#   t3_p1_results.pkl / t3_p1_results.csv   — Phase 1 all T3 combos
#   t3_p2_results.pkl / t3_p2_results.csv   — Phase 2 + confluence
#   t3_p3_results.pkl / t3_p3_results.csv   — Phase 3 + management
#   t3_p4_validated.pkl                      — Phase 4 walk-forward survivors
#   t3_FINAL.csv                             — Print-ready finalist table

import argparse
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nika_optimizer.data_loader import load_ohlcv
from t3_tournament.t3_grid_search import (
    run_t3_tournament, save_results, load_results,
    print_top, save_csv, results_to_df,
)
from t3_tournament.t3_confluence import run_confluence_sweep
from t3_tournament.t3_management  import run_management_sweep
from t3_tournament.t3_walkforward  import run_walkforward
from nika_optimizer.backtest_engine_v2 import (
    R_PROFIT_FACTOR, R_WIN_RATE, R_TOTAL_TRADES,
    R_NET_PROFIT, R_MAX_DRAWDOWN_PCT, R_SHARPE,
    R_EXPECTANCY, R_RECOVERY_FACTOR,
)
from t3_tournament.t3_engine import (
    STRUCT_NAMES, SOURCE_NAMES, MODE_NAMES, IND_INPUT_NAMES, STRUCT_OF_IND,
)


def parse_args():
    p = argparse.ArgumentParser(description='T3 Full Overnight Pipeline')
    p.add_argument('--data',      required=True,          help='OHLCV CSV path')
    p.add_argument('--sep',       default='\t',           help='CSV separator')
    p.add_argument('--output',    default='output_overnight')
    p.add_argument('--cores',     type=int, default=None)
    p.add_argument('--top-p1',    type=int, default=500,  help='Top T3 configs into Phase 2')
    p.add_argument('--top-p2',    type=int, default=100,  help='Top confluence configs into Phase 3')
    p.add_argument('--top-p3',    type=int, default=50,   help='Top management configs into Phase 4')
    p.add_argument('--quick',     action='store_true',    help='Quick test: 5K combos per phase')
    p.add_argument('--skip-p1',   default=None,           help='Load existing Phase 1 pkl')
    p.add_argument('--skip-p2',   default=None,           help='Load existing Phase 2 pkl')
    p.add_argument('--skip-p3',   default=None,           help='Load existing Phase 3 pkl')
    return p.parse_args()


def banner(title):
    print(f"\n{'#'*80}")
    print(f"  {title}")
    print(f"{'#'*80}")


def save_pkl(data, path):
    with open(path, 'wb') as f: pickle.dump(data, f)
    print(f"  Saved: {path}")

def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)


def phase3_to_df(results, top_n=200):
    rows = []
    for entry in results[:top_n]:
        if len(entry) == 5:
            mgmt_cfg, conf_cfg, t3_cfg, res, score = entry
        elif len(entry) == 4:
            conf_cfg, t3_cfg, res, score = entry
            mgmt_cfg = None
        else:
            t3_cfg, res, score = entry
            conf_cfg = mgmt_cfg = None

        src = SOURCE_NAMES[t3_cfg.src_id] if t3_cfg.structure != STRUCT_OF_IND else IND_INPUT_NAMES[t3_cfg.ind_input]
        row = {
            'score':         round(score, 4),
            'structure':     STRUCT_NAMES[t3_cfg.structure],
            'source':        src,
            'signal_mode':   MODE_NAMES[t3_cfg.signal_mode],
            'slow_len':      t3_cfg.slow_len,
            'slow_vf':       t3_cfg.slow_vf,
            'fast_len':      t3_cfg.fast_len,
            'fast_vf':       t3_cfg.fast_vf,
            'mid_len':       t3_cfg.mid_len,
            'mid_vf':        t3_cfg.mid_vf,
            'sensitivity':   t3_cfg.sensitivity,
            'cross_type':    t3_cfg.cross_type,
            # Confluence
            'adx_thresh':    conf_cfg.adx_thresh    if conf_cfg else 0,
            'rsi_zone':      conf_cfg.rsi_zone       if conf_cfg else 0,
            'tsi_dir':       conf_cfg.tsi_dir        if conf_cfg else 0,
            'psar_agree':    conf_cfg.psar_agree     if conf_cfg else 0,
            'di_diff':       conf_cfg.di_diff        if conf_cfg else 0,
            'vol_filter':    conf_cfg.vol_filter     if conf_cfg else 0,
            'n_of_m':        conf_cfg.n_of_m         if conf_cfg else 1,
            # Management
            'atr_sl_mult':   mgmt_cfg.atr_sl_mult    if mgmt_cfg else 0,
            'atr_tp_mult':   mgmt_cfg.atr_tp_mult    if mgmt_cfg else 0,
            'trail_mult':    mgmt_cfg.trail_mult      if mgmt_cfg else 0,
            'be_trigger':    mgmt_cfg.be_trigger      if mgmt_cfg else 0,
            'maturity':      mgmt_cfg.maturity        if mgmt_cfg else 0,
            # Backtest
            'pf':            round(float(res[R_PROFIT_FACTOR]),  4),
            'wr':            round(float(res[R_WIN_RATE]),        2),
            'trades':        int(res[R_TOTAL_TRADES]),
            'net_profit':    round(float(res[R_NET_PROFIT]),      4),
            'expectancy':    round(float(res[R_EXPECTANCY]),      4),
            'max_dd_pct':    round(float(res[R_MAX_DRAWDOWN_PCT]),2),
            'sharpe':        round(float(res[R_SHARPE]),          4),
            'recovery':      round(float(res[R_RECOVERY_FACTOR]),4),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def final_table(validated_survivors):
    rows = []
    for v in validated_survivors:
        entry = v['entry']
        if len(entry) == 5:
            mgmt_cfg, conf_cfg, t3_cfg, res, score = entry
        elif len(entry) == 4:
            conf_cfg, t3_cfg, res, score = entry
            mgmt_cfg = None
        else:
            t3_cfg, res, score = entry
            conf_cfg = mgmt_cfg = None

        src = SOURCE_NAMES[t3_cfg.src_id] if t3_cfg.structure != STRUCT_OF_IND else IND_INPUT_NAMES[t3_cfg.ind_input]
        rows.append({
            'wf_rank':       v['rank'],
            'score':         round(v['score'], 3),
            'avg_oos_pf':    v['avg_oos_pf'],
            'oos_is_ratio':  v['oos_is_ratio'],
            'structure':     STRUCT_NAMES[t3_cfg.structure],
            'source':        src,
            'signal_mode':   MODE_NAMES[t3_cfg.signal_mode],
            'slow_len':      t3_cfg.slow_len,
            'slow_vf':       t3_cfg.slow_vf,
            'fast_len':      t3_cfg.fast_len,
            'sensitivity':   t3_cfg.sensitivity,
            'adx_thresh':    conf_cfg.adx_thresh   if conf_cfg else 0,
            'rsi_zone':      conf_cfg.rsi_zone      if conf_cfg else 0,
            'tsi_dir':       conf_cfg.tsi_dir       if conf_cfg else 0,
            'psar_agree':    conf_cfg.psar_agree    if conf_cfg else 0,
            'n_of_m':        conf_cfg.n_of_m        if conf_cfg else 1,
            'atr_sl_mult':   mgmt_cfg.atr_sl_mult   if mgmt_cfg else 0,
            'atr_tp_mult':   mgmt_cfg.atr_tp_mult   if mgmt_cfg else 0,
            'trail_mult':    mgmt_cfg.trail_mult     if mgmt_cfg else 0,
            'be_trigger':    mgmt_cfg.be_trigger     if mgmt_cfg else 0,
            'maturity':      mgmt_cfg.maturity       if mgmt_cfg else 0,
            'pf':            round(float(res[R_PROFIT_FACTOR]),  3),
            'wr':            round(float(res[R_WIN_RATE]),        1),
            'trades':        int(res[R_TOTAL_TRADES]),
            'max_dd_pct':    round(float(res[R_MAX_DRAWDOWN_PCT]),2),
            'sharpe':        round(float(res[R_SHARPE]),          3),
        })
    return pd.DataFrame(rows)


def main():
    args   = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    n_cores = args.cores or os.cpu_count()
    t_total = time.time()

    banner("T3 OVERNIGHT PIPELINE")
    print(f"  Data:    {args.data}")
    print(f"  Output:  {args.output}")
    print(f"  Cores:   {n_cores}")
    print(f"  Top P1 → P2: {args.top_p1}")
    print(f"  Top P2 → P3: {args.top_p2}")
    print(f"  Top P3 → P4: {args.top_p3}")

    # Load data
    print(f"\n[Data] Loading {args.data}...")
    df = load_ohlcv(args.data, sep=args.sep)
    ohlcv = {
        'open':   df['open'].values.astype(np.float64),
        'high':   df['high'].values.astype(np.float64),
        'low':    df['low'].values.astype(np.float64),
        'close':  df['close'].values.astype(np.float64),
        'volume': df['volume'].values.astype(np.float64),
    }
    print(f"  {len(df):,} bars loaded")

    # -------------------------------------------------------------------------
    # PHASE 1 — T3 Signal Sweep
    # -------------------------------------------------------------------------
    banner("PHASE 1 — T3 Signal Sweep (802,780 combos)")
    if args.skip_p1:
        print(f"  Loading Phase 1 from {args.skip_p1}")
        p1_results = load_pkl(args.skip_p1)
    else:
        p1_results = run_t3_tournament(ohlcv, n_cores=n_cores, quick=args.quick)
        save_pkl(p1_results, outdir / 't3_p1_results.pkl')
        p1_df = results_to_df(p1_results[:args.top_p1])
        p1_df.to_csv(outdir / 't3_p1_results.csv', index=False)
        print(f"  P1 CSV: {outdir/'t3_p1_results.csv'}")

    print_top(p1_results, top_n=20)

    # -------------------------------------------------------------------------
    # PHASE 2 — Confluence Sweep
    # -------------------------------------------------------------------------
    banner("PHASE 2 — Confluence Sweep")
    if args.skip_p2:
        print(f"  Loading Phase 2 from {args.skip_p2}")
        p2_results = load_pkl(args.skip_p2)
    else:
        p2_results = run_confluence_sweep(
            ohlcv, p1_results, n_top=args.top_p1, n_cores=n_cores
        )
        save_pkl(p2_results, outdir / 't3_p2_results.pkl')
        p2_df = phase3_to_df(p2_results, top_n=args.top_p2)
        p2_df.to_csv(outdir / 't3_p2_results.csv', index=False)
        print(f"  P2 CSV: {outdir/'t3_p2_results.csv'}")

    # -------------------------------------------------------------------------
    # PHASE 3 — Management Sweep
    # -------------------------------------------------------------------------
    banner("PHASE 3 — Management Sweep")
    if args.skip_p3:
        print(f"  Loading Phase 3 from {args.skip_p3}")
        p3_results = load_pkl(args.skip_p3)
    else:
        p3_results = run_management_sweep(
            ohlcv, p2_results, n_top=args.top_p2, n_cores=n_cores
        )
        save_pkl(p3_results, outdir / 't3_p3_results.pkl')
        p3_df = phase3_to_df(p3_results, top_n=args.top_p3)
        p3_df.to_csv(outdir / 't3_p3_results.csv', index=False)
        print(f"  P3 CSV: {outdir/'t3_p3_results.csv'}")

    # -------------------------------------------------------------------------
    # PHASE 4 — Walk-Forward Validation
    # -------------------------------------------------------------------------
    banner("PHASE 4 — Walk-Forward Validation")
    validated, survivors = run_walkforward(
        ohlcv, p3_results, n_top=args.top_p3
    )
    save_pkl(validated, outdir / 't3_p4_validated.pkl')

    if survivors:
        final_df = final_table(survivors)
        final_df.to_csv(outdir / 't3_FINAL.csv', index=False)
        print(f"\n  FINAL CSV: {outdir/'t3_FINAL.csv'} ({len(final_df)} survivors)")
    else:
        print("\n  No full survivors. Saving partial passes...")
        partials = [v for v in validated if v['any_passed']]
        if partials:
            partial_df = final_table(partials)
            partial_df.to_csv(outdir / 't3_PARTIAL.csv', index=False)
            print(f"  PARTIAL CSV: {outdir/'t3_PARTIAL.csv'} ({len(partial_df)} entries)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed = time.time() - t_total
    banner("OVERNIGHT PIPELINE COMPLETE")
    print(f"  Total time:      {elapsed/60:.1f} min")
    print(f"  Phase 1 combos:  {len(p1_results):,}")
    print(f"  Phase 2 combos:  {len(p2_results):,}")
    print(f"  Phase 3 combos:  {len(p3_results):,}")
    print(f"  WF survivors:    {len(survivors)}/{args.top_p3}")
    print(f"  Output folder:   {os.path.abspath(args.output)}")
    print()
    if survivors:
        print(f"  Top survivor:")
        v = survivors[0]
        print(f"    Rank:        #{v['rank']}")
        print(f"    IS score:    {v['score']:.2f}")
        print(f"    Avg OOS PF:  {v['avg_oos_pf']:.3f}")
        print(f"    OOS/IS:      {v['oos_is_ratio']:.3f}")
    print()
    print("  Ready for MT5 Real Ticks validation. Good morning! 🟢")
    print()


if __name__ == '__main__':
    main()

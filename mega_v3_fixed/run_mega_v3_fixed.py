# NikaQuant MEGA Optimizer V3 -- FIXED
# Run from TESTER folder: python -m mega_v3_fixed --data XAUUSD_M10.csv --top 50

import numpy as np
import pandas as pd
import argparse
import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nika_optimizer.data_loader import load_ohlcv, prepare_multi_timeframe
from nika_optimizer.signals_v3 import (
    EngineConfig, ConfluenceConfig,
    TRIG_NAMES, MA_NAMES, IND_NAMES, MODE_NAMES,
    NUM_TRIGGER_TYPES, NUM_MA_TYPES, NUM_IND,
    generate_universal_signals, pack_indicator_bits, apply_confluence,
    precompute_shared_indicators,
)
from nika_optimizer.backtest_engine_v2 import (
    make_trade_params, NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_NET_PROFIT, R_EXPECTANCY, R_SHARPE, R_CALMAR, R_FINAL_EQUITY,
    P_BREAK_EVEN_ATR, P_ATR_STOP_MULT, P_TRAIL_ACTIVATE_ATR,
    P_TRAIL_OFFSET_ATR, P_PSAR_EXIT_ENABLED,
    run_backtest,
)
from nika_optimizer.grid_search_v3 import (
    run_phase1, run_phase2, run_phase3,
    save_results, load_results, print_top,
    BE_GRID, TRAIL_ACT, TRAIL_OFF, HARD_STOP, PSAR_EXIT, SPREAD_GRID,
)

from mega_v3_fixed.walk_forward_v3 import WFConfig, run_walk_forward


def main():
    parser = argparse.ArgumentParser(description='NikaQuant MEGA Optimizer V3 FIXED')
    parser.add_argument('--data', required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--cores', type=int, default=None)
    parser.add_argument('--output', default='output_v3')
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--skip-phase1', action='store_true')
    parser.add_argument('--phase1-file', default=None)
    parser.add_argument('--skip-phase3', action='store_true')
    parser.add_argument('--skip-wf', action='store_true')
    parser.add_argument('--top', type=int, default=200)
    parser.add_argument('--phase2-top', type=int, default=50)
    parser.add_argument('--phase3-top', type=int, default=20)
    parser.add_argument('--wf-top', type=int, default=5)
    parser.add_argument('--wf-splits', type=int, default=5)
    parser.add_argument('--wf-train-pct', type=float, default=0.70)
    parser.add_argument('--wf-min-trades', type=int, default=10)

    args = parser.parse_args()
    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_cores = args.cores or os.cpu_count()

    print("\n" + "=" * 80)
    print("  NikaQuant MEGA Optimizer V3 -- FIXED")
    print(f"  11 MA types | 8 trigger types | Heiken Ashi | LinReg | Full confluence")
    print(f"  4 PHASES: Signal -> Confluence -> Trade Mgmt -> Walk-Forward")
    print(f"  Cores: {n_cores} | Output: {output_dir}")
    print("=" * 80)

    t_start = time.time()

    print(f"\n[Step 1] Loading data...")
    ohlcv_df = load_ohlcv(args.data, sep=args.sep)
    mtf = prepare_multi_timeframe(ohlcv_df, ['1h', '4h'])
    ohlcv = {
        'open':   ohlcv_df['open'].values.astype(np.float64),
        'high':   ohlcv_df['high'].values.astype(np.float64),
        'low':    ohlcv_df['low'].values.astype(np.float64),
        'close':  ohlcv_df['close'].values.astype(np.float64),
        'volume': ohlcv_df['volume'].values.astype(np.float64),
        'htf_1h': mtf['1h']['close'].values.astype(np.float64),
        'htf_4h': mtf['4h']['close'].values.astype(np.float64),
    }
    n_bars = len(ohlcv_df)
    n_days = n_bars // 144
    print(f"  {n_bars:,} bars | ~{n_days} trading days")

    # -- Phase 1 --
    if args.skip_phase1 and args.phase1_file:
        print(f"\n[Phase 1] Loading from {args.phase1_file}")
        p1 = load_results(args.phase1_file)
    else:
        p1 = run_phase1(ohlcv, n_cores=n_cores, quick=args.quick)
        save_results(p1, f'{output_dir}/phase1_results.pkl')

    print_top(p1, 30, "Phase 1 -- Signal Tournament")
    t_p1 = time.time()
    print(f"  Phase 1 time: {(t_p1 - t_start)/60:.1f} min")

    # -- Phase 2 --
    top_engines = [eng for eng, res, score in p1[:args.top] if score > 0]
    if not top_engines:
        top_engines = [p1[0][0]]
    print(f"\n[Phase 2] Top {len(top_engines)} engines into confluence sweep...")
    p2 = run_phase2(ohlcv, top_engines, n_cores=n_cores)
    save_results(p2, f'{output_dir}/phase2_results.pkl')
    print_top(p2, 30, "Phase 2 -- With Confluence")
    t_p2 = time.time()
    print(f"  Phase 2 time: {(t_p2 - t_p1)/60:.1f} min")

    # -- Phase 3 --
    p3 = []
    if not args.skip_phase3 and len(p2) > 0:
        top_phase2 = [r for r in p2[:args.phase2_top] if r[3] > 0]
        if not top_phase2:
            top_phase2 = p2[:1]
        print(f"\n[Phase 3] Top {len(top_phase2)} Phase-2 configs into trade management sweep...")
        p3 = run_phase3(ohlcv, top_phase2, n_cores=n_cores)
        save_results(p3, f'{output_dir}/phase3_results.pkl')
        print(f"\n{'='*120}")
        print(f"  TOP 20 Phase 3 -- Trade Management")
        print(f"{'='*120}")
        print(f"  {'#':>3} | {'Score':>7} | {'PF':>6} | {'WR%':>5} | {'Trades':>6} | {'DD%':>6} | {'Net$':>10} | {'BE':>4} | {'Trail':>5} | {'Stop':>4} | {'PSAR':>4} | {'Spread':>6} | Trigger")
        print(f"  {'-'*120}")
        for i, (eng, conf, tp, score, res) in enumerate(p3[:20]):
            tname = TRIG_NAMES[eng.trigger_type]
            if eng.trigger_type == 0:
                tname = MA_NAMES[eng.ma_type]
            print(f"  {i+1:>3} | {score:7.1f} | {res[R_PROFIT_FACTOR]:6.2f} | {res[R_WIN_RATE]:5.1f} | "
                  f"{int(res[R_TOTAL_TRADES]):>6} | {res[R_MAX_DRAWDOWN_PCT]:6.1f} | "
                  f"${res[R_NET_PROFIT]:>9.2f} | "
                  f"{tp[P_BREAK_EVEN_ATR]:4.1f} | {tp[P_TRAIL_ACTIVATE_ATR]:5.1f} | "
                  f"{tp[P_ATR_STOP_MULT]:4.1f} | {int(tp[P_PSAR_EXIT_ENABLED]):>4} | "
                  f"{tp[8]:6.2f} | {tname} L={eng.ma_length}")
        print(f"{'='*120}")
        t_p3 = time.time()
        print(f"  Phase 3 time: {(t_p3 - t_p2)/60:.1f} min")
    else:
        t_p3 = time.time()
        print("\n[Phase 3] Skipped.")

    # -- Phase 4: Walk-Forward --
    wf_results = []
    if not args.skip_wf:
        if p3:
            wf_candidates = p3[:args.wf_top]
        else:
            wf_candidates = [
                (eng, conf, make_trade_params(spread_points=0.30, use_open_entry=1.0), score, res)
                for eng, conf, res, score in p2[:args.wf_top]
            ]

        print(f"\n[Phase 4] Walk-Forward on top {len(wf_candidates)} configs...")
        print(f"  Splits: {args.wf_splits} | Train: {args.wf_train_pct*100:.0f}% | Min trades: {args.wf_min_trades}")

        wf_cfg = WFConfig(
            n_splits=args.wf_splits,
            train_pct=args.wf_train_pct,
            min_trades=args.wf_min_trades,
        )

        for rank, candidate in enumerate(wf_candidates):
            eng  = candidate[0]
            conf = candidate[1]
            tp   = candidate[2]
            tname = TRIG_NAMES[eng.trigger_type]
            if eng.trigger_type == 0:
                tname = MA_NAMES[eng.ma_type]
            print(f"\n  Config #{rank+1}: {tname} L={eng.ma_length} TF={eng.tfactor:.2f} S={eng.sensitivity}")
            try:
                wf = run_walk_forward(ohlcv, eng, conf, tp, wf_cfg)
                oos_pf     = float(np.mean([r[R_PROFIT_FACTOR] for r in wf['oos_results']]))
                oos_wr     = float(np.mean([r[R_WIN_RATE]      for r in wf['oos_results']]))
                oos_trades = int(sum([r[R_TOTAL_TRADES]        for r in wf['oos_results']]))
                is_pf      = float(np.mean([r[R_PROFIT_FACTOR] for r in wf['is_results']]))
                robustness = oos_pf / max(is_pf, 0.01)
                status = (
                    "ROBUST"   if robustness > 0.60 and oos_pf > 1.0
                    else "FRAGILE" if oos_pf > 1.0
                    else "OVERFIT"
                )
                print(f"    IS  PF: {is_pf:.2f}")
                print(f"    OOS PF: {oos_pf:.2f} | WR: {oos_wr:.1f}% | Trades: {oos_trades}")
                print(f"    Robustness: {robustness:.2f} -> {status}")
                wf_results.append({
                    'rank': rank + 1, 'trigger': tname,
                    'engine': eng, 'confluence': conf, 'trade_params': tp,
                    'is_pf': is_pf, 'oos_pf': oos_pf,
                    'oos_wr': oos_wr, 'oos_trades': oos_trades,
                    'robustness': robustness, 'status': status,
                    'holdout_pf': wf['summary']['holdout_pf'],
                    'full_wf': wf,
                })
            except Exception as e:
                print(f"    WF failed: {e}")
                import traceback; traceback.print_exc()
                wf_results.append({'rank': rank+1, 'trigger': tname, 'status': f'FAILED: {e}'})

        save_results(wf_results, f'{output_dir}/phase4_wf_results.pkl')
        t_p4 = time.time()
        print(f"\n  Phase 4 time: {(t_p4 - t_p3)/60:.1f} min")

        print(f"\n{'='*100}")
        print(f"  WALK-FORWARD SUMMARY")
        print(f"{'='*100}")
        print(f"  {'#':>3} | {'Trigger':>12} | {'IS PF':>6} | {'OOS PF':>6} | {'WR':>6} | {'Trades':>8} | {'Holdout':>7} | {'Robust':>6} | Status")
        print(f"  {'-'*100}")
        for wfr in wf_results:
            if 'oos_pf' in wfr:
                print(f"  {wfr['rank']:>3} | {wfr['trigger']:>12} | {wfr['is_pf']:6.2f} | {wfr['oos_pf']:6.2f} | "
                      f"{wfr['oos_wr']:5.1f}% | {wfr['oos_trades']:>8} | {wfr['holdout_pf']:7.2f} | "
                      f"{wfr['robustness']:6.2f} | {wfr['status']}")
            else:
                print(f"  {wfr['rank']:>3} | {wfr['trigger']:>12} | {wfr['status']}")
        print(f"{'='*100}")
    else:
        t_p4 = time.time()
        print("\n[Phase 4] Walk-Forward skipped (--skip-wf).")

    # -- Final Output --
    if wf_results and any('oos_pf' in w and w.get('robustness', 0) > 0.6 for w in wf_results):
        robust = [w for w in wf_results if w.get('robustness', 0) > 0.6 and w.get('oos_pf', 0) > 1.0]
        robust.sort(key=lambda w: w['oos_pf'], reverse=True)
        best = robust[0]
        best_eng  = best['engine']
        best_conf = best['confluence']
        best_tp   = best['trade_params']
        best_source = f"Phase 4 WF (OOS PF={best['oos_pf']:.2f})"
    elif p3:
        best_eng, best_conf, best_tp, _, _ = p3[0]
        best_source = "Phase 3"
    elif p2:
        best_eng, best_conf, _, _ = p2[0]
        best_tp = make_trade_params(spread_points=0.30, use_open_entry=1.0)
        best_source = "Phase 2"
    else:
        best_eng, _, _ = p1[0]
        best_conf = ConfluenceConfig()
        best_tp   = make_trade_params(spread_points=0.30, use_open_entry=1.0)
        best_source = "Phase 1"

    shared = precompute_shared_indicators(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'], best_eng
    )
    sig = generate_universal_signals(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'],
        best_eng, ohlcv.get('htf_1h'), ohlcv.get('htf_4h'), shared=shared,
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
    if best_conf.indicator_mask > 0:
        buy_f, sell_f = apply_confluence(sig.buy, sig.sell, sig.ind_buy, sig.ind_sell, best_conf)
    else:
        buy_f, sell_f = sig.buy, sig.sell
    best_res = run_backtest(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], sig.open_next,
        t3_trend, buy_f, sell_f, sig.atr,
        ibb, ibs, best_conf.indicator_mask, best_conf.min_agree,
        sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
        best_tp, 0, -1,
    )

    tname = TRIG_NAMES[best_eng.trigger_type]
    if best_eng.trigger_type == 0:
        tname = MA_NAMES[best_eng.ma_type]

    print(f"\n{'='*80}")
    print(f"  BEST RESULT (source: {best_source})")
    print(f"{'='*80}")
    print(f"  Trigger:       {tname}")
    print(f"  MA Length:     {best_eng.ma_length}")
    print(f"  T-Factor:      {best_eng.tfactor}")
    print(f"  Sensitivity:   {best_eng.sensitivity}")
    print(f"  Mode:          {MODE_NAMES[best_eng.signal_mode]}")
    print(f"  Heiken Ashi:   {'ON' if best_eng.use_heiken_ashi else 'OFF'}")
    print(f"  PF:            {best_res[R_PROFIT_FACTOR]:.2f}")
    print(f"  Win Rate:      {best_res[R_WIN_RATE]:.1f}%")
    print(f"  Trades:        {int(best_res[R_TOTAL_TRADES])}")
    print(f"  Max DD:        {best_res[R_MAX_DRAWDOWN_PCT]:.1f}%")
    print(f"  Net Profit:    ${best_res[R_NET_PROFIT]:.2f}")
    print(f"  Sharpe:        {best_res[R_SHARPE]:.2f}")
    print(f"  Calmar:        {best_res[R_CALMAR]:.2f}")
    print(f"  Final Equity:  ${best_res[R_FINAL_EQUITY]:.2f}")

    if best_conf.indicator_mask > 0:
        active = [IND_NAMES[j] for j in range(NUM_IND) if best_conf.indicator_mask & (1 << j)]
        print(f"  Confluence:    {'+'.join(active)} ({best_conf.min_agree} of {len(active)} agree)")

    print(f"  Break-Even:    {best_tp[P_BREAK_EVEN_ATR]:.1f} ATR")
    print(f"  Trail Activate:{best_tp[P_TRAIL_ACTIVATE_ATR]:.1f} ATR")
    print(f"  Hard Stop:     {best_tp[P_ATR_STOP_MULT]:.1f} ATR")
    print(f"  PSAR Exit:     {'ON' if best_tp[P_PSAR_EXIT_ENABLED] > 0.5 else 'OFF'}")
    print(f"  Spread:        ${best_tp[8]:.2f}")

    tv_settings = {
        'trigger_type': tname,
        'ma_type': MA_NAMES[best_eng.ma_type] if best_eng.trigger_type == 0 else 'N/A',
        'ma_length': best_eng.ma_length,
        't_factor': best_eng.tfactor,
        'sensitivity': best_eng.sensitivity,
        'signal_mode': MODE_NAMES[best_eng.signal_mode],
        'heiken_ashi': bool(best_eng.use_heiken_ashi),
        'confluence': {
            'active_indicators': [IND_NAMES[j] for j in range(NUM_IND) if best_conf.indicator_mask & (1 << j)],
            'min_agree': best_conf.min_agree,
        },
        'trade_management': {
            'break_even_atr': float(best_tp[P_BREAK_EVEN_ATR]),
            'trail_activate_atr': float(best_tp[P_TRAIL_ACTIVATE_ATR]),
            'trail_offset_atr': float(best_tp[P_TRAIL_OFFSET_ATR]),
            'hard_stop_atr': float(best_tp[P_ATR_STOP_MULT]),
            'psar_exit': bool(best_tp[P_PSAR_EXIT_ENABLED] > 0.5),
            'spread': float(best_tp[8]),
        },
        'results': {name: float(best_res[i]) for i, name in enumerate(RESULT_NAMES)},
    }
    json_path = f'{output_dir}/best_params.json'
    with open(json_path, 'w') as f:
        json.dump(tv_settings, f, indent=2)
    print(f"\n  Saved: {json_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"  Phase 1 (Signals):       {(t_p1 - t_start)/60:.1f} min")
    print(f"  Phase 2 (Confluence):    {(t_p2 - t_p1)/60:.1f} min")
    print(f"  Phase 3 (Trade Mgmt):    {(t_p3 - t_p2)/60:.1f} min")
    print(f"  Phase 4 (Walk-Forward):  {(t_p4 - t_p3)/60:.1f} min")
    print(f"  TOTAL:                   {elapsed/60:.1f} min")
    print(f"{'='*80}")
    print(f"\n  MEGA V3 FIXED -- COMPLETE. DATA IS KING.\n")


if __name__ == '__main__':
    main()

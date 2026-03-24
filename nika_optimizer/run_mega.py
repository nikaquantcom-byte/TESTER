"""
Module 6 — MEGA Optimizer V2 Main Orchestrator
12-hour mode: test EVERYTHING. No shortcuts.

Usage:
    python -m nika_optimizer.run_mega --data "C:/path/to/XAUUSD_M10.csv"
    python -m nika_optimizer.run_mega --data "C:/path/to/XAUUSD_M10.csv" --quick
"""

import numpy as np
import pandas as pd
import argparse
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

from .data_loader import load_ohlcv, prepare_multi_timeframe
from .signals_v2 import (
    T3Params, IndicatorParams, ConfluenceConfig, SignalArrays,
    generate_signals, pack_voter_bits, pack_gate_bits,
    generate_all_confluence_configs,
    NUM_VOTERS, NUM_GATES, VOTER_NAMES, GATE_NAMES,
)
from .backtest_engine_v2 import (
    run_backtest, run_backtest_with_equity, make_trade_params,
    NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_NET_PROFIT, R_EXPECTANCY, R_SHARPE, R_CALMAR,
    R_AVG_WIN_ATR, R_AVG_LOSS_ATR, R_AVG_BARS_HELD, R_FINAL_EQUITY,
    R_MAX_CONSEC_LOSS, R_WINS, R_LOSSES,
    P_BREAK_EVEN_ATR, P_ATR_STOP_MULT, P_TRAIL_ACTIVATE_ATR,
    P_TRAIL_OFFSET_ATR, P_PSAR_EXIT_ENABLED, P_PSAR_MATURITY_MIN,
    P_SWING_LOOKBACK,
)
from .grid_search_v2 import (
    run_phase1, run_phase2, run_phase3, run_phase4,
    save_results, load_results,
    print_phase1_top, print_phase3_top,
    compute_composite_score, generate_trade_combos,
    T3_SLOW, T_FACTOR, SENSITIVITY, FAST_RATIO,
)
from .walk_forward_v2 import WFConfig, run_walk_forward
from .visualize import generate_full_report


def format_tradingview_output(t3p, conf, tp, results, wf_summary=None):
    """Format best parameters for TradingView."""
    out = []
    out.append("=" * 80)
    out.append("  BEST PARAMETERS FOR TRADINGVIEW — MEGA OPTIMIZER V2")
    out.append("  Trend Duration Forecast + Optimizer V7 Pro — XAUUSD M10")
    out.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append("=" * 80)

    out.append("\n── T3 Core Settings ──")
    out.append(f"  T3 Slow Length:       {t3p.slow_len}")
    out.append(f"  T3 Fast Length:       {t3p.fast_len}")
    out.append(f"  T-Factor (vFactor):   {t3p.tfactor}")
    out.append(f"  Sensitivity:          {t3p.sensitivity}")

    out.append("\n── Confluence Settings ──")
    active_voters = [VOTER_NAMES[j] for j in range(NUM_VOTERS) if conf.voter_mask & (1 << j)]
    active_gates = [GATE_NAMES[j] for j in range(NUM_GATES) if conf.gate_mask & (1 << j)]
    out.append(f"  Active Voters:        {', '.join(active_voters) if active_voters else 'NONE (raw T3)'}")
    out.append(f"  Min Voters Agree:     {conf.min_voters_agree}")
    out.append(f"  Active Gates:         {', '.join(active_gates) if active_gates else 'NONE'}")
    out.append(f"  MTF Block:            {'ON' if conf.mtf_block else 'OFF'}")

    out.append("\n── Trade Management ──")
    be = tp[P_BREAK_EVEN_ATR]
    out.append(f"  Break-Even:           {be:.2f} ATR {'(DISABLED)' if be == 0 else ''}")
    out.append(f"  ATR Hard Stop:        {tp[P_ATR_STOP_MULT]:.1f} {'(DISABLED)' if tp[P_ATR_STOP_MULT] == 0 else ''}")
    trail = tp[P_TRAIL_ACTIVATE_ATR]
    out.append(f"  Trailing Stop:        {'DISABLED' if trail > 900 else f'{trail:.1f}/{tp[P_TRAIL_OFFSET_ATR]:.1f} ATR'}")
    out.append(f"  PSAR Exit:            {'ON' if tp[P_PSAR_EXIT_ENABLED] > 0.5 else 'OFF'}")

    out.append("\n── Performance ──")
    out.append(f"  Trades:               {int(results[R_TOTAL_TRADES])}")
    out.append(f"  Win Rate:             {results[R_WIN_RATE]:.1f}%")
    out.append(f"  Profit Factor:        {results[R_PROFIT_FACTOR]:.2f}")
    out.append(f"  Sharpe:               {results[R_SHARPE]:.2f}")
    out.append(f"  Max Drawdown:         {results[R_MAX_DRAWDOWN_PCT]:.1f}%")
    out.append(f"  Expectancy:           {results[R_EXPECTANCY]:.3f} ATR/trade")
    out.append(f"  Net Profit:           ${results[R_NET_PROFIT]:.2f}")

    if wf_summary:
        out.append("\n── Walk-Forward ──")
        out.append(f"  Folds:                {wf_summary['n_folds']}")
        out.append(f"  Profitable:           {wf_summary['profitable_folds']}/{wf_summary['n_folds']} ({wf_summary['profitable_pct']:.0f}%)")
        out.append(f"  Avg OOS PF:           {wf_summary['avg_oos_pf']:.2f}")
        out.append(f"  OOS/IS Ratio:         {wf_summary['avg_oos_is_ratio']:.2f}")
        robustness = "ROBUST" if wf_summary['avg_oos_is_ratio'] >= 0.60 else \
                     "MODERATE" if wf_summary['avg_oos_is_ratio'] >= 0.40 else "OVERFIT RISK"
        out.append(f"  Assessment:           {robustness}")

    out.append("=" * 80)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description='NikaQuant MEGA Optimizer V2 — XAUUSD M10')
    parser.add_argument('--data', required=True, help='Path to M10 OHLCV CSV')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (~5 min)')
    parser.add_argument('--cores', type=int, default=None)
    parser.add_argument('--output', default='output_v2')
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--spread', type=float, default=0.30)
    parser.add_argument('--skip-phase1', action='store_true')
    parser.add_argument('--phase1-file', default=None)
    parser.add_argument('--skip-wf', action='store_true')
    parser.add_argument('--top-t3', type=int, default=300, help='Top T3 configs for Phase 2')
    parser.add_argument('--top-conf', type=int, default=200, help='Top configs for Phase 3→4')

    args = parser.parse_args()
    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_cores = args.cores or os.cpu_count()

    print("\n" + "=" * 80)
    print("  NikaQuant MEGA Optimizer V2 — Test EVERYTHING")
    print(f"  Cores: {n_cores} | Output: {output_dir}")
    print("=" * 80)

    # ── Step 1: Load Data ──
    t_start = time.time()
    print(f"\n[Step 1] Loading data...")
    ohlcv_df = load_ohlcv(args.data, sep=args.sep)
    mtf = prepare_multi_timeframe(ohlcv_df, ['1h', '4h'])

    ohlcv = {
        'open': ohlcv_df['open'].values.astype(np.float64),
        'high': ohlcv_df['high'].values.astype(np.float64),
        'low': ohlcv_df['low'].values.astype(np.float64),
        'close': ohlcv_df['close'].values.astype(np.float64),
        'volume': ohlcv_df['volume'].values.astype(np.float64),
        'htf_1h': mtf['1h']['close'].values.astype(np.float64),
        'htf_4h': mtf['4h']['close'].values.astype(np.float64),
    }
    n_bars = len(ohlcv_df)
    print(f"  {n_bars:,} bars | ~{n_bars // 144} trading days")

    # ── Phase 1: T3 Core ──
    if args.skip_phase1 and args.phase1_file:
        print(f"\n[Phase 1] Loading from {args.phase1_file}")
        p1_results = load_results(args.phase1_file)
    else:
        p1_results = run_phase1(ohlcv, n_cores=n_cores)
        save_results(p1_results, f'{output_dir}/phase1_results.pkl')
        print_phase1_top(p1_results, 30)

    # Extract top T3 configs
    top_n = min(args.top_t3, len(p1_results))
    top_t3 = []
    baseline_pfs = {}
    for t3p, res_none, score_none, res_be, score_be in p1_results[:top_n]:
        top_t3.append(t3p)
        baseline_pfs[t3p] = res_none[R_PROFIT_FACTOR]

    # ── Phase 2: Independent Indicator Testing ──
    print(f"\n[Phase 2] Independent indicator testing on top {len(top_t3)} T3 configs...")
    indicator_deltas = run_phase2(ohlcv, top_t3[:100], baseline_pfs, n_cores=n_cores)
    save_results(indicator_deltas, f'{output_dir}/phase2_indicator_deltas.pkl')

    # ── Phase 3: Confluence Sweep ──
    # Generate all confluence configs (full sweep)
    all_conf = generate_all_confluence_configs(include_no_filter=True)
    print(f"\n  Total unique confluence states: {len(all_conf):,}")

    # For quick mode, limit confluence configs
    if args.quick:
        # Sample: top 100 T3 × 100 random confluence configs
        import random
        random.seed(42)
        sampled_conf = random.sample(all_conf, min(100, len(all_conf)))
        sampled_conf.insert(0, ConfluenceConfig(voter_mask=0, gate_mask=0, min_voters_agree=0, mtf_block=0))
        p3_results = run_phase3(ohlcv, top_t3[:50], sampled_conf, n_cores=n_cores)
    else:
        p3_results = run_phase3(ohlcv, top_t3[:args.top_conf], all_conf, n_cores=n_cores)

    save_results(p3_results, f'{output_dir}/phase3_results.pkl')
    print_phase3_top(p3_results, 30)

    # ── Phase 4: Trade Management ──
    if len(p3_results) > 0 and p3_results[0][3] > 0:
        # Take the single best signal config and precompute its signals
        best_t3p = p3_results[0][0]
        best_conf = p3_results[0][1]

        print(f"\n[Phase 4] Trade management sweep on best signal config...")
        sig = generate_signals(
            ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'],
            best_t3p, htf_close_1h=ohlcv['htf_1h'], htf_close_4h=ohlcv['htf_4h'],
        )

        signal_data = {
            't3_trend': sig.t3_trend,
            'raw_flip_up': sig.raw_flip_up, 'raw_flip_down': sig.raw_flip_down,
            'atr': sig.atr,
            'vb': pack_voter_bits(sig, 'buy'), 'vs': pack_voter_bits(sig, 'sell'),
            'gb': pack_gate_bits(sig),
            'voter_mask': best_conf.voter_mask,
            'gate_mask': best_conf.gate_mask,
            'min_voters': best_conf.min_voters_agree,
            'psar_flip_up': sig.psar_flip_up, 'psar_flip_down': sig.psar_flip_down,
            'swing_high': sig.swing_high, 'swing_low': sig.swing_low,
        }

        p4_results = run_phase4(ohlcv, signal_data, n_cores=n_cores)
        save_results(p4_results, f'{output_dir}/phase4_results.pkl')

        best_tp = p4_results[0][0]
        best_results = p4_results[0][1]
    else:
        print("\n[Phase 4] No passing configs — using defaults")
        best_t3p = p1_results[0][0] if p1_results else T3Params()
        best_conf = ConfluenceConfig(voter_mask=0, gate_mask=0, min_voters_agree=0, mtf_block=0)
        best_tp = make_trade_params(spread_points=args.spread)
        best_results = p1_results[0][1] if p1_results else np.zeros(NUM_RESULTS)

    # ── Phase 5: Walk-Forward ──
    wf_summary = None
    if not args.skip_wf:
        print(f"\n[Phase 5] Walk-Forward Validation...")
        # Build candidate list from top Phase 3+4 results
        candidates = []
        for t3p, conf, res, score in p3_results[:50]:
            if score > 0:
                candidates.append((t3p, conf, make_trade_params(spread_points=args.spread)))

        if candidates:
            wf_summary = run_walk_forward(
                ohlcv, candidates,
                config=WFConfig(),
                htf_1h=ohlcv['htf_1h'], htf_4h=ohlcv['htf_4h'],
            )

    # ── Final Output ──
    print(f"\n[Output] Generating results...")

    # Run final backtest with equity curve
    sig = generate_signals(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'],
        best_t3p, htf_close_1h=ohlcv['htf_1h'], htf_close_4h=ohlcv['htf_4h'],
    )
    vb = pack_voter_bits(sig, 'buy'); vs = pack_voter_bits(sig, 'sell'); gb = pack_gate_bits(sig)
    final_res, equity_curve, trade_log = run_backtest_with_equity(
        ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'],
        sig.t3_trend, sig.raw_flip_up, sig.raw_flip_down, sig.atr,
        vb, vs, gb,
        best_conf.voter_mask, best_conf.gate_mask, best_conf.min_voters_agree,
        sig.psar_flip_up, sig.psar_flip_down, sig.swing_high, sig.swing_low,
        best_tp,
    )

    tv_output = format_tradingview_output(best_t3p, best_conf, best_tp, final_res, wf_summary)
    print(tv_output)

    with open(f'{output_dir}/best_params_tradingview.txt', 'w') as f:
        f.write(tv_output)
    print(f"\n  Saved: {output_dir}/best_params_tradingview.txt")

    # JSON
    json_out = {
        't3': dict(best_t3p._asdict()),
        'confluence': dict(best_conf._asdict()),
        'trade_params': {
            'break_even_atr': float(best_tp[P_BREAK_EVEN_ATR]),
            'atr_stop_mult': float(best_tp[P_ATR_STOP_MULT]),
            'trail_activate_atr': float(best_tp[P_TRAIL_ACTIVATE_ATR]),
            'trail_offset_atr': float(best_tp[P_TRAIL_OFFSET_ATR]),
            'psar_exit': bool(best_tp[P_PSAR_EXIT_ENABLED] > 0.5),
        },
        'results': {name: float(final_res[i]) for i, name in enumerate(RESULT_NAMES)},
    }
    with open(f'{output_dir}/best_params.json', 'w') as f:
        json.dump(json_out, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  MEGA OPTIMIZER V2 COMPLETE — {elapsed/3600:.1f} hours")
    print(f"  Output: {Path(output_dir).resolve()}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

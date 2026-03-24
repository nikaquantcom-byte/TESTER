"""
Module 7 — Main Orchestrator
Runs the full optimization pipeline:
  1. Load M10 OHLCV data
  2. Create MTF data (1H, 4H)
  3. Phase 1: Signal parameter sweep (24 cores)
  4. Phase 2: Trade management sweep (24 cores)
  5. Walk-forward validation
  6. Generate visualizations
  7. Output best parameters formatted for TradingView Pine Script

Usage:
    python run_optimizer.py --data "C:/path/to/XAUUSD_M10.csv"
    python run_optimizer.py --data "C:/path/to/XAUUSD_M10.csv" --quick   (reduced grid for testing)
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
from .signals import SignalParams, generate_signals
from .backtest_engine import (
    run_backtest, run_backtest_with_equity, make_trade_params,
    NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_SHARPE, R_EXPECTANCY, R_NET_PROFIT, R_RECOVERY_FACTOR, R_CALMAR,
    R_AVG_WIN_ATR, R_AVG_LOSS_ATR, R_AVG_BARS_HELD, R_FINAL_EQUITY,
    R_MAX_CONSEC_LOSS, R_WINS, R_LOSSES,
    P_ATR_STOP_MULT, P_TRAIL_ACTIVATE_ATR, P_TRAIL_OFFSET_ATR,
    P_BREAKEVEN_ATR, P_PSAR_EXIT_ENABLED, P_PSAR_MATURITY_PCT,
    P_PSAR_MATURITY_MIN, P_SWING_LOOKBACK,
)
from .grid_search import (
    SignalGrid, TradeGrid, count_combos,
    run_signal_sweep, run_trade_sweep,
    print_top_results, save_results, load_results,
    compute_rank_score,
)
from .walk_forward import WalkForwardConfig, run_walk_forward
from .visualize import generate_full_report


# ─────────────────────────────────────────────────────────────────────────────
# Grid Presets
# ─────────────────────────────────────────────────────────────────────────────

def get_signal_grid(mode: str = 'full') -> SignalGrid:
    """Get signal parameter grid based on mode."""
    if mode == 'quick':
        return SignalGrid(
            t3_slow_lens=[30, 50, 70],
            t3_fast_lens=[5],
            t3_alpha=[0.7],
            t3_sensitivity=[2, 3, 4],
            adx_len=[14],
            adx_threshold=[20.0, 25.0],
            rsi_overbought=[70, 80],
            rsi_oversold=[20, 30],
            min_quality_score=[50.0, 60.0],
            min_trend_strength=[0.0, 0.3],
            trend_metric_thresh=[0.60],
            volatile_atr_thresh=[1.30],
        )
    elif mode == 'medium':
        return SignalGrid(
            t3_slow_lens=list(range(20, 81, 10)),
            t3_fast_lens=[3, 5, 8],
            t3_alpha=[0.7],
            t3_sensitivity=[1, 2, 3, 4, 5],
            adx_len=[10, 14, 20],
            adx_threshold=[15.0, 20.0, 25.0, 30.0],
            rsi_overbought=[70, 80, 90],
            rsi_oversold=[10, 20, 30],
            min_quality_score=[40.0, 50.0, 60.0, 70.0],
            min_trend_strength=[0.0, 0.3, 0.6],
            trend_metric_thresh=[0.40, 0.60, 0.80],
            volatile_atr_thresh=[1.20, 1.30, 1.50],
        )
    else:  # full — massive sweep
        return SignalGrid(
            t3_slow_lens=list(range(15, 101, 5)),
            t3_fast_lens=[3, 5, 8],
            t3_alpha=[0.7],
            t3_sensitivity=[1, 2, 3, 4, 5, 7, 10],
            adx_len=[10, 14, 20],
            adx_threshold=[15.0, 20.0, 25.0, 30.0],
            rsi_overbought=[70, 80, 90],
            rsi_oversold=[10, 20, 30],
            min_quality_score=[40.0, 50.0, 60.0, 70.0],
            min_trend_strength=[0.0, 0.3, 0.6],
            trend_metric_thresh=[0.40, 0.60, 0.80],
            volatile_atr_thresh=[1.20, 1.30, 1.50],
        )


def get_trade_grid(mode: str = 'full') -> TradeGrid:
    """Get trade management parameter grid."""
    if mode == 'quick':
        return TradeGrid(
            atr_stop_mult=[0.0, 3.0],
            trail_activate_atr=[1.5, 2.0, 2.5],
            trail_offset_atr=[0.5],
            breakeven_atr=[1.5, 2.0],
            psar_exit_enabled=[0.0, 1.0],
            psar_maturity_pct=[0.5],
            psar_maturity_min=[10.0],
            swing_lookback=[10.0],
        )
    else:
        return TradeGrid()  # Full default ranges


# ─────────────────────────────────────────────────────────────────────────────
# TradingView Output Formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_for_tradingview(
    signal_params: SignalParams,
    trade_params: np.ndarray,
    results: np.ndarray,
    wf_summary: dict = None,
) -> str:
    """Format best parameters as TradingView Pine Script settings."""
    
    output = []
    output.append("=" * 80)
    output.append("  BEST PARAMETERS FOR TRADINGVIEW")
    output.append("  Trend Duration Forecast + Optimizer V7 Pro — XAUUSD M10")
    output.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)
    
    output.append("\n── Pine Script Settings ─────────────────────────────────────")
    output.append(f"  Mode:                       Custom")
    output.append(f"  T3 Length (Slow):            {signal_params.t3_slow_len}")
    output.append(f"  Ribbon Fast T3 Length:       {signal_params.t3_fast_len}")
    output.append(f"  T3 Alpha (Smoothing):        {signal_params.t3_alpha}")
    output.append(f"  Sensitivity:                 {signal_params.t3_sensitivity}")
    output.append(f"  ADX Length:                  {signal_params.adx_len}")
    output.append(f"  ADX Minimum Threshold:       {signal_params.adx_threshold}")
    output.append(f"  RSI Overbought Level:        {signal_params.rsi_overbought}")
    output.append(f"  RSI Oversold Level:          {signal_params.rsi_oversold}")
    output.append(f"  Min Trend Strength:          {signal_params.min_trend_strength}")
    output.append(f"  Trending Threshold:          {signal_params.trend_metric_thresh}")
    output.append(f"  Volatile ATR Ratio:          {signal_params.volatile_atr_thresh}")
    
    output.append(f"\n── Risk Management Settings ─────────────────────────────────")
    
    hard_stop = trade_params[P_ATR_STOP_MULT]
    output.append(f"  ATR Stop Multiplier:         {hard_stop:.1f} {'(DISABLED)' if hard_stop == 0 else ''}")
    output.append(f"  Trailing Stop Activation:    {trade_params[P_TRAIL_ACTIVATE_ATR]:.1f} ATR")
    output.append(f"  Trailing Stop Offset:        {trade_params[P_TRAIL_OFFSET_ATR]:.1f} ATR")
    output.append(f"  Break-Even Trigger:          {trade_params[P_BREAKEVEN_ATR]:.1f} ATR")
    
    psar_on = trade_params[P_PSAR_EXIT_ENABLED] > 0.5
    output.append(f"  PSAR Exit:                   {'ON' if psar_on else 'OFF'}")
    if psar_on:
        output.append(f"  PSAR Maturity %:             {trade_params[P_PSAR_MATURITY_PCT]:.0%}")
        output.append(f"  PSAR Maturity Min Bars:      {int(trade_params[P_PSAR_MATURITY_MIN])}")
    output.append(f"  Swing Stop Lookback:         {int(trade_params[P_SWING_LOOKBACK])} bars")
    
    output.append(f"\n── Backtest Performance ─────────────────────────────────────")
    output.append(f"  Total Trades:                {int(results[R_TOTAL_TRADES])}")
    output.append(f"  Wins / Losses:               {int(results[R_WINS])} / {int(results[R_LOSSES])}")
    output.append(f"  Win Rate:                    {results[R_WIN_RATE]:.1f}%")
    output.append(f"  Profit Factor:               {results[R_PROFIT_FACTOR]:.2f}")
    output.append(f"  Sharpe Ratio:                {results[R_SHARPE]:.2f}")
    output.append(f"  Recovery Factor:             {results[R_RECOVERY_FACTOR]:.2f}")
    output.append(f"  Calmar Ratio:                {results[R_CALMAR]:.2f}")
    output.append(f"  Expectancy:                  {results[R_EXPECTANCY]:.3f} ATR/trade")
    output.append(f"  Max Drawdown:                {results[R_MAX_DRAWDOWN_PCT]:.1f}%")
    output.append(f"  Max Consecutive Losses:      {int(results[R_MAX_CONSEC_LOSS])}")
    output.append(f"  Avg Win:                     {results[R_AVG_WIN_ATR]:.2f} ATR")
    output.append(f"  Avg Loss:                    {results[R_AVG_LOSS_ATR]:.2f} ATR")
    output.append(f"  Avg Bars Held:               {results[R_AVG_BARS_HELD]:.0f}")
    output.append(f"  Net Profit:                  ${results[R_NET_PROFIT]:.2f}")
    output.append(f"  Final Equity:                ${results[R_FINAL_EQUITY]:.2f}")
    
    if wf_summary:
        output.append(f"\n── Walk-Forward Validation ──────────────────────────────────")
        output.append(f"  Folds:                       {wf_summary['n_folds']}")
        output.append(f"  Profitable Folds:            {wf_summary['profitable_folds']}/{wf_summary['n_folds']} "
                      f"({wf_summary['profitable_pct']:.0f}%)")
        output.append(f"  Avg OOS Profit Factor:       {wf_summary['avg_oos_pf']:.2f}")
        output.append(f"  Avg OOS Win Rate:            {wf_summary['avg_oos_wr']:.1f}%")
        output.append(f"  Avg OOS/IS Rank Ratio:       {wf_summary['avg_oos_is_ratio']:.2f}")
        
        robustness = "ROBUST" if wf_summary['avg_oos_is_ratio'] >= 0.60 else \
                     "MODERATE" if wf_summary['avg_oos_is_ratio'] >= 0.40 else "OVERFIT RISK"
        output.append(f"  Overall Assessment:          {robustness}")
    
    output.append(f"\n── Quick-Paste for TradingView ──────────────────────────────")
    output.append(f"  Set these in the indicator settings dialog:")
    output.append(f"    Mode = Custom")
    output.append(f"    Manual T3 Length = {signal_params.t3_slow_len}")
    output.append(f"    Ribbon Fast T3 Length = {signal_params.t3_fast_len}")
    output.append(f"    Manual Sensitivity = {signal_params.t3_sensitivity}")
    output.append(f"    ADX Length = {signal_params.adx_len}")
    output.append(f"    ADX Minimum Threshold = {signal_params.adx_threshold}")
    output.append(f"    RSI Overbought = {signal_params.rsi_overbought}")
    output.append(f"    RSI Oversold = {signal_params.rsi_oversold}")
    output.append(f"    Min Trend Strength = {signal_params.min_trend_strength}")
    output.append(f"    Trending Threshold = {signal_params.trend_metric_thresh}")
    output.append(f"    Volatile ATR Ratio = {signal_params.volatile_atr_thresh}")
    output.append(f"    ATR Stop Multiplier = {hard_stop:.1f}")
    output.append(f"    Trailing Stop Activation = {trade_params[P_TRAIL_ACTIVATE_ATR]:.1f}")
    output.append(f"    Trailing Stop Offset = {trade_params[P_TRAIL_OFFSET_ATR]:.1f}")
    output.append(f"    Break-Even Trigger = {trade_params[P_BREAKEVEN_ATR]:.1f}")
    output.append(f"    Swing Stop Lookback = {int(trade_params[P_SWING_LOOKBACK])}")
    output.append("=" * 80)
    
    return "\n".join(output)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NikaQuant V7 Pro — Python Optimizer for XAUUSD M10',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimizer.py --data "C:/MT5_Data/XAUUSD_M10.csv"
  python run_optimizer.py --data "C:/MT5_Data/XAUUSD_M10.csv" --quick
  python run_optimizer.py --data "C:/MT5_Data/XAUUSD_M10.csv" --mode medium --cores 16
  python run_optimizer.py --data "C:/MT5_Data/XAUUSD_M10.csv" --skip-phase1 --phase1-file output/phase1_results.pkl
        """
    )
    
    parser.add_argument('--data', required=True, help='Path to M10 OHLCV CSV file')
    parser.add_argument('--mode', choices=['quick', 'medium', 'full'], default='full',
                       help='Grid size: quick (~1K combos), medium (~50K), full (~500K+)')
    parser.add_argument('--quick', action='store_true', help='Shortcut for --mode quick')
    parser.add_argument('--cores', type=int, default=None, help='CPU cores (default: all)')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--sep', default='\t', help='CSV separator (default: tab)')
    parser.add_argument('--spread', type=float, default=0.30, help='XAUUSD spread in $ (default: 0.30)')
    parser.add_argument('--initial-equity', type=float, default=10000.0, help='Starting equity')
    parser.add_argument('--risk-pct', type=float, default=1.0, help='Risk % per trade')
    parser.add_argument('--skip-phase1', action='store_true', help='Skip Phase 1, load from file')
    parser.add_argument('--phase1-file', default=None, help='Phase 1 results file to load')
    parser.add_argument('--skip-wf', action='store_true', help='Skip walk-forward validation')
    parser.add_argument('--wf-folds', type=int, default=8, help='Walk-forward folds')
    
    args = parser.parse_args()
    
    if args.quick:
        args.mode = 'quick'
    
    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    n_cores = args.cores or os.cpu_count()
    
    print("\n" + "=" * 80)
    print("  NikaQuant V7 Pro — XAUUSD M10 Optimizer")
    print(f"  Mode: {args.mode} | Cores: {n_cores} | Output: {output_dir}")
    print("=" * 80)
    
    # ── Step 1: Load Data ──
    t_start = time.time()
    
    print(f"\n[Step 1] Loading M10 OHLCV data...")
    ohlcv_df = load_ohlcv(args.data, sep=args.sep)
    
    # Create MTF data
    mtf_data = prepare_multi_timeframe(ohlcv_df, ['1h', '4h'])
    
    # Convert to numpy
    ohlcv_np = {
        'open': ohlcv_df['open'].values.astype(np.float64),
        'high': ohlcv_df['high'].values.astype(np.float64),
        'low': ohlcv_df['low'].values.astype(np.float64),
        'close': ohlcv_df['close'].values.astype(np.float64),
        'volume': ohlcv_df['volume'].values.astype(np.float64),
        'htf_close_1h': mtf_data['1h']['close'].values.astype(np.float64),
        'htf_close_4h': mtf_data['4h']['close'].values.astype(np.float64),
        'swing_lookback': 10,
    }
    
    n_bars = len(ohlcv_df)
    print(f"  {n_bars:,} bars loaded | ~{n_bars / 144:.0f} trading days")
    
    # ── Step 2: Phase 1 — Signal Parameter Sweep ──
    default_trade_params = make_trade_params(
        spread_points=args.spread,
        initial_equity=args.initial_equity,
        risk_pct=args.risk_pct,
    )
    
    if args.skip_phase1 and args.phase1_file:
        print(f"\n[Step 2] Loading Phase 1 results from {args.phase1_file}...")
        signal_results = load_results(args.phase1_file)
    else:
        print(f"\n[Step 2] Phase 1 — Signal Parameter Sweep...")
        signal_grid = get_signal_grid(args.mode)
        
        # Count combos
        from .grid_search import generate_signal_combos
        combos = generate_signal_combos(signal_grid)
        print(f"  Grid produces {len(combos):,} valid signal combinations")
        
        signal_results = run_signal_sweep(
            ohlcv_np,
            signal_grid,
            default_trade_params,
            n_cores=n_cores,
        )
        
        # Save Phase 1 results
        save_results(signal_results, f'{output_dir}/phase1_results.pkl')
        print_top_results(signal_results, top_n=20, phase="— Phase 1 (Signals)")
    
    # ── Step 3: Phase 2 — Trade Management Sweep ──
    if len(signal_results) > 0 and signal_results[0][2] > 0:
        best_signal_params = signal_results[0][0]
        print(f"\n[Step 3] Phase 2 — Trade Management Sweep...")
        print(f"  Using best signal params: T3 L{best_signal_params.t3_slow_len}/"
              f"S{best_signal_params.t3_sensitivity} | "
              f"ADX {best_signal_params.adx_len}/{best_signal_params.adx_threshold}")
        
        trade_grid = get_trade_grid(args.mode)
        
        trade_results = run_trade_sweep(
            ohlcv_np,
            best_signal_params,
            trade_grid,
            htf_close_1h=ohlcv_np['htf_close_1h'],
            htf_close_4h=ohlcv_np['htf_close_4h'],
            n_cores=n_cores,
        )
        
        save_results(trade_results, f'{output_dir}/phase2_results.pkl')
        print_top_results(trade_results, top_n=10, phase="— Phase 2 (Trade Mgmt)")
        
        best_trade_params = trade_results[0][0]
        best_overall_results = trade_results[0][1]
    else:
        print("\n[Step 3] No passing signal combos — using defaults for trade params")
        best_signal_params = signal_results[0][0] if signal_results else SignalParams()
        best_trade_params = default_trade_params
        best_overall_results = signal_results[0][1] if signal_results else np.zeros(NUM_RESULTS)
    
    # ── Step 4: Walk-Forward Validation ──
    wf_summary = None
    if not args.skip_wf:
        print(f"\n[Step 4] Walk-Forward Validation ({args.wf_folds} folds)...")
        
        # Use top 50 signal params as candidates for WF
        top_candidates = [sp for sp, _, rank in signal_results[:50] if rank > 0]
        if len(top_candidates) < 5:
            top_candidates = [sp for sp, _, _ in signal_results[:50]]
        
        wf_config = WalkForwardConfig(
            is_window_bars=144 * 126,   # ~6 months IS
            oos_window_bars=144 * 42,   # ~2 months OOS
            step_bars=144 * 42,         # Step by OOS size
            top_n_for_oos=5,
        )
        
        wf_summary = run_walk_forward(
            ohlcv_np,
            top_candidates,
            best_trade_params,
            config=wf_config,
            htf_close_1h=ohlcv_np['htf_close_1h'],
            htf_close_4h=ohlcv_np['htf_close_4h'],
        )
    
    # ── Step 5: Generate Equity Curve for Best Params ──
    print(f"\n[Step 5] Running final backtest with best parameters...")
    
    sig = generate_signals(
        ohlcv_np['open'], ohlcv_np['high'], ohlcv_np['low'],
        ohlcv_np['close'], ohlcv_np['volume'],
        best_signal_params,
        htf_close_1h=ohlcv_np['htf_close_1h'],
        htf_close_4h=ohlcv_np['htf_close_4h'],
    )
    
    final_results, equity_curve, trade_log = run_backtest_with_equity(
        ohlcv_np['open'], ohlcv_np['high'], ohlcv_np['low'],
        ohlcv_np['close'],
        sig.t3_trend, sig.atr,
        sig.psar_flip_up, sig.psar_flip_down,
        sig.swing_high, sig.swing_low,
        sig.signal_buy, sig.signal_sell,
        best_trade_params,
    )
    
    # ── Step 6: Visualizations ──
    print(f"\n[Step 6] Generating visualizations...")
    generate_full_report(
        signal_results=signal_results,
        wf_summary=wf_summary,
        equity_curve=equity_curve,
        trade_log=trade_log,
        output_dir=output_dir,
    )
    
    # ── Step 7: Output Results ──
    print(f"\n[Step 7] Formatting results for TradingView...")
    
    tv_output = format_for_tradingview(
        best_signal_params,
        best_trade_params,
        final_results,
        wf_summary,
    )
    
    print(tv_output)
    
    # Save to file
    tv_filepath = f'{output_dir}/best_params_tradingview.txt'
    with open(tv_filepath, 'w') as f:
        f.write(tv_output)
    print(f"\n  Saved to: {tv_filepath}")
    
    # Save JSON for programmatic use
    json_output = {
        'signal_params': dict(best_signal_params._asdict()),
        'trade_params': {
            'atr_stop_mult': best_trade_params[P_ATR_STOP_MULT],
            'trail_activate_atr': best_trade_params[P_TRAIL_ACTIVATE_ATR],
            'trail_offset_atr': best_trade_params[P_TRAIL_OFFSET_ATR],
            'breakeven_atr': best_trade_params[P_BREAKEVEN_ATR],
            'psar_exit_enabled': bool(best_trade_params[P_PSAR_EXIT_ENABLED] > 0.5),
            'psar_maturity_pct': best_trade_params[P_PSAR_MATURITY_PCT],
            'psar_maturity_min': int(best_trade_params[P_PSAR_MATURITY_MIN]),
            'swing_lookback': int(best_trade_params[P_SWING_LOOKBACK]),
        },
        'results': {name: float(final_results[i]) for i, name in enumerate(RESULT_NAMES)},
        'walk_forward': {
            'avg_oos_pf': wf_summary['avg_oos_pf'] if wf_summary else None,
            'profitable_folds_pct': wf_summary['profitable_pct'] if wf_summary else None,
            'oos_is_ratio': wf_summary['avg_oos_is_ratio'] if wf_summary else None,
        } if wf_summary else None,
    }
    
    json_filepath = f'{output_dir}/best_params.json'
    with open(json_filepath, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved JSON: {json_filepath}")
    
    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  OPTIMIZATION COMPLETE — {elapsed:.0f}s total")
    print(f"  Output directory: {Path(output_dir).resolve()}")
    print(f"{'='*80}")
    
    return json_output


if __name__ == '__main__':
    main()

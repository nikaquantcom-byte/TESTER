"""
Module 6 — Visualization
Heatmaps, equity curves, parameter robustness plateaus, walk-forward fold charts.
Uses matplotlib with dark theme matching nikaquant style.
Optionally uses VectorBT Pro's plotting if available.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .signals import SignalParams
from .backtest_engine import (
    NUM_RESULTS, RESULT_NAMES,
    R_PROFIT_FACTOR, R_WIN_RATE, R_MAX_DRAWDOWN_PCT, R_TOTAL_TRADES,
    R_SHARPE, R_EXPECTANCY, R_NET_PROFIT, R_RECOVERY_FACTOR, R_CALMAR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Style configuration
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG = '#0d0d0d'
DARK_FG = '#1a1a1a'
TEXT_COLOR = '#e0e0e0'
GREEN = '#12e49e'
RED = '#ec5610'
ORANGE = '#ff9800'
BLUE = '#4fc3f7'

# Custom red-to-green colormap
NIKA_CMAP = LinearSegmentedColormap.from_list(
    'nika', 
    [(0.0, '#ec5610'), (0.3, '#ff9800'), (0.5, '#ffeb3b'), (0.7, '#8bc34a'), (1.0, '#12e49e')]
)

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': DARK_FG,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': '#222222',
    'grid.alpha': 0.5,
    'font.size': 10,
    'font.family': 'monospace',
})


def _save_fig(fig, filepath: str, dpi: int = 150):
    """Save figure and close."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Saved: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 6A. Parameter Heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_2d(
    results_list: List[Tuple],
    x_param: str,
    y_param: str,
    metric_idx: int = R_PROFIT_FACTOR,
    metric_name: str = 'Profit Factor',
    output_dir: str = 'output',
    filename: str = None,
    aggregation: str = 'max',
):
    """
    2D heatmap of a metric across two signal parameters.
    Aggregates across all other params using max/mean.
    """
    # Extract param values
    param_fields = SignalParams._fields
    x_idx = param_fields.index(x_param)
    y_idx = param_fields.index(y_param)
    
    # Collect data
    data_dict = {}
    for params, results, rank in results_list:
        x_val = params[x_idx]
        y_val = params[y_idx]
        metric_val = results[metric_idx]
        
        key = (x_val, y_val)
        if key not in data_dict:
            data_dict[key] = []
        data_dict[key].append(metric_val)
    
    # Get unique values
    x_vals = sorted(set(k[0] for k in data_dict.keys()))
    y_vals = sorted(set(k[1] for k in data_dict.keys()))
    
    # Build matrix
    matrix = np.full((len(y_vals), len(x_vals)), np.nan)
    for (x_val, y_val), metrics in data_dict.items():
        xi = x_vals.index(x_val)
        yi = y_vals.index(y_val)
        if aggregation == 'max':
            matrix[yi, xi] = np.max(metrics)
        elif aggregation == 'mean':
            matrix[yi, xi] = np.mean(metrics)
        elif aggregation == 'median':
            matrix[yi, xi] = np.median(metrics)
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(x_vals) * 0.8), max(6, len(y_vals) * 0.6)))
    
    im = ax.imshow(matrix, aspect='auto', cmap=NIKA_CMAP, origin='lower',
                   interpolation='nearest')
    
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha='right')
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    
    ax.set_xlabel(x_param.replace('_', ' ').title())
    ax.set_ylabel(y_param.replace('_', ' ').title())
    ax.set_title(f'{metric_name} Heatmap ({aggregation.title()})\n{x_param} vs {y_param}',
                 fontsize=12, fontweight='bold')
    
    # Add value annotations
    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            val = matrix[yi, xi]
            if not np.isnan(val):
                text_color = 'white' if val < np.nanpercentile(matrix, 60) else 'black'
                ax.text(xi, yi, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, color=text_color)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_name)
    
    if filename is None:
        filename = f'heatmap_{x_param}_vs_{y_param}_{metric_name.lower().replace(" ", "_")}.png'
    
    _save_fig(fig, f'{output_dir}/{filename}')


def generate_all_heatmaps(results_list: List[Tuple], output_dir: str = 'output'):
    """Generate the most important heatmaps automatically."""
    print(f"\n[visualize] Generating heatmaps...")
    
    # Only use results that pass gates for cleaner heatmaps
    passing = [r for r in results_list if r[2] > 0]
    if len(passing) < 10:
        passing = results_list[:max(100, len(results_list))]
    
    heatmap_pairs = [
        ('t3_slow_len', 't3_sensitivity', R_PROFIT_FACTOR, 'Profit Factor'),
        ('t3_slow_len', 't3_sensitivity', R_WIN_RATE, 'Win Rate %'),
        ('t3_slow_len', 'adx_threshold', R_PROFIT_FACTOR, 'Profit Factor'),
        ('t3_slow_len', 'min_quality_score', R_PROFIT_FACTOR, 'Profit Factor'),
        ('adx_threshold', 'min_trend_strength', R_PROFIT_FACTOR, 'Profit Factor'),
        ('t3_slow_len', 't3_sensitivity', R_EXPECTANCY, 'Expectancy (ATR)'),
        ('rsi_overbought', 'rsi_oversold', R_PROFIT_FACTOR, 'Profit Factor'),
        ('trend_metric_thresh', 'volatile_atr_thresh', R_PROFIT_FACTOR, 'Profit Factor'),
    ]
    
    for x_param, y_param, metric_idx, metric_name in heatmap_pairs:
        try:
            plot_heatmap_2d(passing, x_param, y_param, metric_idx, metric_name, output_dir)
        except (ValueError, IndexError) as e:
            print(f"  Skipping {x_param} vs {y_param}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6B. Equity Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curve(
    equity_curve: np.ndarray,
    dates: np.ndarray = None,
    trade_log: np.ndarray = None,
    title: str = 'Equity Curve — Best Parameters',
    output_dir: str = 'output',
    filename: str = 'equity_curve.png',
):
    """Plot equity curve with drawdown shading and trade markers."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                                     gridspec_kw={'hspace': 0.15})
    
    x = np.arange(len(equity_curve))
    
    # Equity curve
    ax1.plot(x, equity_curve, color=GREEN, linewidth=1.2, label='Equity')
    
    # Running peak
    peak = np.maximum.accumulate(equity_curve)
    ax1.plot(x, peak, color='#444444', linewidth=0.5, linestyle='--', alpha=0.7, label='Peak')
    
    # Fill drawdown
    dd_mask = equity_curve < peak
    ax1.fill_between(x, equity_curve, peak, where=dd_mask,
                     color=RED, alpha=0.15, label='Drawdown')
    
    # Trade markers from log
    if trade_log is not None and len(trade_log) > 0:
        wins = trade_log[trade_log[:, 5] > 0]  # pnl_pct > 0
        losses = trade_log[trade_log[:, 5] <= 0]
        
        if len(wins) > 0:
            win_bars = wins[:, 0].astype(int)  # entry bars
            valid = win_bars < len(equity_curve)
            ax1.scatter(win_bars[valid], equity_curve[win_bars[valid]],
                       color=GREEN, s=8, alpha=0.6, zorder=5, label=f'Wins ({len(wins)})')
        
        if len(losses) > 0:
            loss_bars = losses[:, 0].astype(int)
            valid = loss_bars < len(equity_curve)
            ax1.scatter(loss_bars[valid], equity_curve[loss_bars[valid]],
                       color=RED, s=8, alpha=0.6, zorder=5, label=f'Losses ({len(losses)})')
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Drawdown subplot
    drawdown_pct = np.where(peak > 0, (equity_curve - peak) / peak * 100, 0)
    ax2.fill_between(x, drawdown_pct, 0, color=RED, alpha=0.5)
    ax2.plot(x, drawdown_pct, color=RED, linewidth=0.8)
    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Bar Index')
    ax2.grid(True, alpha=0.3)
    
    # Add max DD annotation
    min_dd_idx = np.argmin(drawdown_pct)
    min_dd_val = drawdown_pct[min_dd_idx]
    ax2.annotate(f'Max DD: {min_dd_val:.1f}%',
                xy=(min_dd_idx, min_dd_val),
                xytext=(min_dd_idx + len(x)*0.05, min_dd_val * 0.7),
                color=RED, fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
    
    _save_fig(fig, f'{output_dir}/{filename}')


# ─────────────────────────────────────────────────────────────────────────────
# 6C. Parameter Robustness (Plateau) Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_param_robustness(
    results_list: List[Tuple],
    param_name: str,
    metric_idx: int = R_PROFIT_FACTOR,
    metric_name: str = 'Profit Factor',
    output_dir: str = 'output',
):
    """
    1D robustness plot: metric distribution across one parameter.
    Shows median + interquartile range to identify robust plateaus vs sharp peaks.
    """
    param_fields = SignalParams._fields
    p_idx = param_fields.index(param_name)
    
    # Collect
    data_by_val = {}
    for params, results, rank in results_list:
        val = params[p_idx]
        if val not in data_by_val:
            data_by_val[val] = []
        data_by_val[val].append(results[metric_idx])
    
    x_vals = sorted(data_by_val.keys())
    medians = [np.median(data_by_val[v]) for v in x_vals]
    q25 = [np.percentile(data_by_val[v], 25) for v in x_vals]
    q75 = [np.percentile(data_by_val[v], 75) for v in x_vals]
    maxes = [np.max(data_by_val[v]) for v in x_vals]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(x_vals))
    
    # IQR shading
    ax.fill_between(x_pos, q25, q75, color=BLUE, alpha=0.2, label='IQR (25th-75th)')
    
    # Median line
    ax.plot(x_pos, medians, color=GREEN, linewidth=2, marker='o', markersize=6, label='Median')
    
    # Max line
    ax.plot(x_pos, maxes, color=ORANGE, linewidth=1, linestyle='--', marker='s', 
            markersize=4, label='Max', alpha=0.7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha='right')
    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel(metric_name)
    ax.set_title(f'Parameter Robustness: {param_name}\n'
                 f'Wide plateaus = robust | Sharp peaks = overfit',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Annotate best plateau region
    best_med_idx = np.argmax(medians)
    ax.axvline(x=best_med_idx, color=GREEN, linestyle=':', alpha=0.5)
    ax.annotate(f'Best: {x_vals[best_med_idx]}',
               xy=(best_med_idx, medians[best_med_idx]),
               xytext=(best_med_idx + 0.5, medians[best_med_idx]),
               color=GREEN, fontsize=9, fontweight='bold')
    
    _save_fig(fig, f'{output_dir}/robustness_{param_name}.png')


def generate_all_robustness(results_list: List[Tuple], output_dir: str = 'output'):
    """Generate robustness plots for key parameters."""
    print(f"\n[visualize] Generating robustness plots...")
    
    key_params = [
        't3_slow_len', 't3_sensitivity', 'adx_threshold',
        'min_quality_score', 'min_trend_strength',
        'trend_metric_thresh', 'volatile_atr_thresh',
    ]
    
    for param in key_params:
        try:
            plot_param_robustness(results_list, param, R_PROFIT_FACTOR, 'Profit Factor', output_dir)
            plot_param_robustness(results_list, param, R_EXPECTANCY, 'Expectancy (ATR)', output_dir)
        except (ValueError, IndexError) as e:
            print(f"  Skipping {param}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6D. Walk-Forward Fold Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_walk_forward_summary(
    wf_summary: Dict,
    output_dir: str = 'output',
    filename: str = 'walk_forward_summary.png',
):
    """Plot walk-forward OOS performance across folds."""
    fold_results = wf_summary['fold_results']
    n_folds = len(fold_results)
    
    if n_folds == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    folds = list(range(1, n_folds + 1))
    
    # PF across folds
    ax = axes[0, 0]
    oos_pfs = [f['best_oos_results'][R_PROFIT_FACTOR] for f in fold_results]
    colors = [GREEN if pf > 1.0 else RED for pf in oos_pfs]
    ax.bar(folds, oos_pfs, color=colors, alpha=0.8, edgecolor='#333333')
    ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='Break-even')
    ax.axhline(y=np.mean(oos_pfs), color=BLUE, linestyle='-', alpha=0.7, 
               label=f'Avg: {np.mean(oos_pfs):.2f}')
    ax.set_title('OOS Profit Factor per Fold', fontweight='bold')
    ax.set_xlabel('Fold')
    ax.set_ylabel('PF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Win rate across folds
    ax = axes[0, 1]
    oos_wrs = [f['best_oos_results'][R_WIN_RATE] for f in fold_results]
    ax.bar(folds, oos_wrs, color=BLUE, alpha=0.8, edgecolor='#333333')
    ax.axhline(y=50, color='white', linestyle='--', alpha=0.5)
    ax.set_title('OOS Win Rate % per Fold', fontweight='bold')
    ax.set_xlabel('Fold')
    ax.set_ylabel('WR %')
    ax.grid(True, alpha=0.3)
    
    # IS vs OOS rank ratio
    ax = axes[1, 0]
    if 'oos_is_ratio' in fold_results[0]:
        ratios = [f['oos_is_ratio'] for f in fold_results]
        colors = [GREEN if r >= 0.5 else ORANGE if r >= 0.3 else RED for r in ratios]
        ax.bar(folds, ratios, color=colors, alpha=0.8, edgecolor='#333333')
        ax.axhline(y=0.5, color=GREEN, linestyle='--', alpha=0.5, label='Robust threshold')
        ax.axhline(y=0.3, color=RED, linestyle='--', alpha=0.5, label='Overfit threshold')
        ax.set_title('OOS/IS Rank Ratio per Fold', fontweight='bold')
        ax.set_ylabel('Ratio')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14, color=TEXT_COLOR)
    ax.set_xlabel('Fold')
    ax.grid(True, alpha=0.3)
    
    # Max DD across folds
    ax = axes[1, 1]
    oos_dds = [abs(f['best_oos_results'][R_MAX_DRAWDOWN_PCT]) for f in fold_results]
    colors = [GREEN if dd < 15 else ORANGE if dd < 25 else RED for dd in oos_dds]
    ax.bar(folds, oos_dds, color=colors, alpha=0.8, edgecolor='#333333')
    ax.axhline(y=15, color=GREEN, linestyle='--', alpha=0.5)
    ax.axhline(y=25, color=RED, linestyle='--', alpha=0.5)
    ax.set_title('OOS Max Drawdown % per Fold', fontweight='bold')
    ax.set_xlabel('Fold')
    ax.set_ylabel('DD %')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Walk-Forward Validation Summary', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    _save_fig(fig, f'{output_dir}/{filename}')


# ─────────────────────────────────────────────────────────────────────────────
# 6E. Trade Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_trade_distribution(
    trade_log: np.ndarray,
    output_dir: str = 'output',
    filename: str = 'trade_distribution.png',
):
    """Plot trade P&L distribution and holding period distribution."""
    if trade_log is None or len(trade_log) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    pnl_pct = trade_log[:, 5]    # P&L %
    pnl_atr = trade_log[:, 6]    # P&L ATR
    bars_held = trade_log[:, 1] - trade_log[:, 0]  # exit - entry bar
    
    # P&L distribution (ATR)
    ax = axes[0]
    wins = pnl_atr[pnl_atr > 0]
    losses = pnl_atr[pnl_atr <= 0]
    bins = np.linspace(min(pnl_atr), max(pnl_atr), 40)
    ax.hist(wins, bins=bins, color=GREEN, alpha=0.7, label=f'Wins ({len(wins)})')
    ax.hist(losses, bins=bins, color=RED, alpha=0.7, label=f'Losses ({len(losses)})')
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax.set_title('P&L Distribution (ATR)', fontweight='bold')
    ax.set_xlabel('P&L (ATR units)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Holding period
    ax = axes[1]
    ax.hist(bars_held, bins=30, color=BLUE, alpha=0.7, edgecolor='#333333')
    ax.axvline(x=np.median(bars_held), color=ORANGE, linestyle='--',
              label=f'Median: {np.median(bars_held):.0f} bars')
    ax.set_title('Holding Period Distribution', fontweight='bold')
    ax.set_xlabel('Bars Held')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Cumulative P&L
    ax = axes[2]
    cum_pnl = np.cumsum(pnl_atr)
    ax.plot(cum_pnl, color=GREEN, linewidth=1.5)
    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                   where=cum_pnl > 0, color=GREEN, alpha=0.1)
    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                   where=cum_pnl <= 0, color=RED, alpha=0.1)
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax.set_title('Cumulative P&L (ATR)', fontweight='bold')
    ax.set_xlabel('Trade #')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    _save_fig(fig, f'{output_dir}/{filename}')


# ─────────────────────────────────────────────────────────────────────────────
# 6F. Master Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_report(
    signal_results: List[Tuple],
    trade_results: List[Tuple] = None,
    wf_summary: Dict = None,
    equity_curve: np.ndarray = None,
    trade_log: np.ndarray = None,
    output_dir: str = 'output',
):
    """Generate all visualizations for the optimization run."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  GENERATING VISUALIZATION REPORT")
    print(f"{'='*60}")
    
    # Heatmaps from signal sweep
    if signal_results:
        generate_all_heatmaps(signal_results, output_dir)
        generate_all_robustness(signal_results, output_dir)
    
    # Walk-forward
    if wf_summary:
        plot_walk_forward_summary(wf_summary, output_dir)
    
    # Equity curve
    if equity_curve is not None:
        plot_equity_curve(equity_curve, trade_log=trade_log, output_dir=output_dir)
    
    # Trade distribution
    if trade_log is not None and len(trade_log) > 0:
        plot_trade_distribution(trade_log, output_dir)
    
    print(f"\n[visualize] All charts saved to {output_dir}/")

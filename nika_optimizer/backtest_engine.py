"""
Module 3 — Numba @njit Backtest Engine
Sequential bar-by-bar trade management that can't be vectorized:
  - T3 flip as primary entry/exit
  - ATR-based hard stop (sweepable on/off + multiplier)
  - Trailing stop (activation ATR, offset ATR, swing-based)
  - Break-even trigger
  - PSAR flip exit (after trade maturity)
  - Equity tracking, drawdown, profit factor, win rate

Returns structured results array for each parameter combo.
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict as NumbaDict


# ─────────────────────────────────────────────────────────────────────────────
# Trade management parameters (passed as flat array for Numba compatibility)
# ─────────────────────────────────────────────────────────────────────────────
# Index mapping for trade_params array:
P_ATR_STOP_MULT       = 0   # ATR multiplier for hard stop (0 = disabled)
P_TRAIL_ACTIVATE_ATR  = 1   # ATR move before trailing activates
P_TRAIL_OFFSET_ATR    = 2   # Trail offset from swing low/high in ATR units
P_BREAKEVEN_ATR       = 3   # ATR move before stop moves to entry
P_PSAR_EXIT_ENABLED   = 4   # 1.0 = PSAR exit on, 0.0 = off
P_PSAR_MATURITY_PCT   = 5   # % of forecast bars before PSAR exit allowed
P_PSAR_MATURITY_MIN   = 6   # Minimum bars before PSAR exit
P_SWING_LOOKBACK      = 7   # Bars for swing high/low lookback
P_SPREAD_POINTS       = 8   # Spread in price points (for realistic fills)
P_COMMISSION_PCT      = 9   # Round-trip commission as % of trade value
P_INITIAL_EQUITY      = 10  # Starting equity
P_RISK_PCT            = 11  # Risk per trade as % of equity (0 = fixed 1 lot)
NUM_TRADE_PARAMS      = 12


def make_trade_params(
    atr_stop_mult: float = 0.0,       # 0 = disabled (proven to kill PF)
    trail_activate_atr: float = 2.0,
    trail_offset_atr: float = 0.5,
    breakeven_atr: float = 2.0,
    psar_exit_enabled: float = 1.0,
    psar_maturity_pct: float = 0.5,
    psar_maturity_min: float = 10.0,
    swing_lookback: float = 10.0,
    spread_points: float = 0.30,       # XAUUSD typical spread ~$0.30
    commission_pct: float = 0.0,
    initial_equity: float = 10000.0,
    risk_pct: float = 1.0,
) -> np.ndarray:
    """Create trade_params array from named parameters."""
    p = np.zeros(NUM_TRADE_PARAMS, dtype=np.float64)
    p[P_ATR_STOP_MULT]      = atr_stop_mult
    p[P_TRAIL_ACTIVATE_ATR] = trail_activate_atr
    p[P_TRAIL_OFFSET_ATR]   = trail_offset_atr
    p[P_BREAKEVEN_ATR]      = breakeven_atr
    p[P_PSAR_EXIT_ENABLED]  = psar_exit_enabled
    p[P_PSAR_MATURITY_PCT]  = psar_maturity_pct
    p[P_PSAR_MATURITY_MIN]  = psar_maturity_min
    p[P_SWING_LOOKBACK]     = swing_lookback
    p[P_SPREAD_POINTS]      = spread_points
    p[P_COMMISSION_PCT]     = commission_pct
    p[P_INITIAL_EQUITY]     = initial_equity
    p[P_RISK_PCT]           = risk_pct
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Result indices
# ─────────────────────────────────────────────────────────────────────────────
R_TOTAL_TRADES     = 0
R_WINS             = 1
R_LOSSES           = 2
R_WIN_RATE         = 3
R_GROSS_PROFIT     = 4
R_GROSS_LOSS       = 5
R_NET_PROFIT       = 6
R_PROFIT_FACTOR    = 7
R_MAX_DRAWDOWN_PCT = 8
R_SHARPE           = 9
R_RECOVERY_FACTOR  = 10
R_AVG_WIN_ATR      = 11
R_AVG_LOSS_ATR     = 12
R_EXPECTANCY       = 13
R_MAX_CONSEC_LOSS  = 14
R_AVG_BARS_HELD    = 15
R_FINAL_EQUITY     = 16
R_CALMAR           = 17
NUM_RESULTS        = 18


RESULT_NAMES = [
    'total_trades', 'wins', 'losses', 'win_rate',
    'gross_profit', 'gross_loss', 'net_profit', 'profit_factor',
    'max_drawdown_pct', 'sharpe', 'recovery_factor',
    'avg_win_atr', 'avg_loss_atr', 'expectancy',
    'max_consec_loss', 'avg_bars_held', 'final_equity', 'calmar',
]


# ─────────────────────────────────────────────────────────────────────────────
# Core Numba Backtest Function
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def run_backtest(
    # OHLCV
    open_: np.ndarray,     # float64[n]
    high: np.ndarray,      # float64[n]
    low: np.ndarray,       # float64[n]
    close: np.ndarray,     # float64[n]
    # Signals from the signal engine
    t3_trend: np.ndarray,  # int32[n]: 1=UP, -1=DOWN, 0=NONE
    atr: np.ndarray,       # float64[n]
    psar_flip_up: np.ndarray,   # bool[n]
    psar_flip_down: np.ndarray, # bool[n]
    swing_high: np.ndarray,     # float64[n]
    swing_low: np.ndarray,      # float64[n]
    signal_buy: np.ndarray,     # bool[n] — filtered buy signals
    signal_sell: np.ndarray,    # bool[n] — filtered sell signals
    # Trade management parameters
    trade_params: np.ndarray,   # float64[NUM_TRADE_PARAMS]
    # Optional: subset range for walk-forward
    start_idx: int = 0,
    end_idx: int = -1,
) -> np.ndarray:
    """
    Bar-by-bar backtest engine with full trade management.
    
    Exit priority (same as Pine V7):
      1. Trailing stop hit (if active)
      2. Hard ATR stop hit (if enabled)
      3. PSAR flip exit (if enabled and trade matured)
      4. T3 flip (primary exit — always active)
    
    Returns: float64[NUM_RESULTS] with performance metrics.
    """
    n = len(close)
    if end_idx < 0:
        end_idx = n

    # Extract params
    atr_stop_mult      = trade_params[P_ATR_STOP_MULT]
    trail_activate_atr  = trade_params[P_TRAIL_ACTIVATE_ATR]
    trail_offset_atr    = trade_params[P_TRAIL_OFFSET_ATR]
    breakeven_atr       = trade_params[P_BREAKEVEN_ATR]
    psar_exit_enabled   = trade_params[P_PSAR_EXIT_ENABLED] > 0.5
    psar_maturity_pct   = trade_params[P_PSAR_MATURITY_PCT]
    psar_maturity_min   = int(trade_params[P_PSAR_MATURITY_MIN])
    spread              = trade_params[P_SPREAD_POINTS]
    commission_pct      = trade_params[P_COMMISSION_PCT]
    initial_equity      = trade_params[P_INITIAL_EQUITY]
    risk_pct            = trade_params[P_RISK_PCT]
    use_hard_stop       = atr_stop_mult > 0.0

    # Trade state
    in_trade = False
    trade_dir = 0        # 1 = long, -1 = short
    entry_price = 0.0
    entry_atr = 0.0
    hard_stop = 0.0
    trailing_stop = 0.0
    trail_active = False
    breakeven_done = False
    bars_in_trade = 0
    forecast_bars = 0.0  # estimated from duration (use 30 as default proxy)

    # Equity tracking
    equity = initial_equity
    peak_equity = initial_equity
    max_dd_pct = 0.0

    # Stats
    total_trades = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sum_win_atr = 0.0
    sum_loss_atr = 0.0
    total_bars_held = 0
    consec_losses = 0
    max_consec_losses = 0

    # For Sharpe: collect per-trade returns
    max_trades = 10000
    trade_returns = np.zeros(max_trades, dtype=np.float64)
    trade_count_for_sharpe = 0

    for i in range(start_idx, end_idx):
        # ─── In-trade management ───
        if in_trade:
            bars_in_trade += 1
            
            favorable_move = 0.0
            if trade_dir == 1:
                favorable_move = close[i] - entry_price
            else:
                favorable_move = entry_price - close[i]
            
            favorable_atr = favorable_move / max(entry_atr, 0.01)
            
            # Break-even
            if not breakeven_done and favorable_atr >= breakeven_atr:
                breakeven_done = True
                if trade_dir == 1:
                    hard_stop = max(hard_stop, entry_price) if use_hard_stop else entry_price
                else:
                    hard_stop = min(hard_stop, entry_price) if use_hard_stop else entry_price
                # If hard stop was disabled, we still set it at entry for break-even
                if not use_hard_stop:
                    hard_stop = entry_price
            
            # Trailing stop activation
            if not trail_active and favorable_atr >= trail_activate_atr:
                trail_active = True
            
            # Update trailing stop
            if trail_active:
                if trade_dir == 1:
                    new_trail = swing_low[i] - atr[i] * trail_offset_atr
                    if trailing_stop == 0.0 or new_trail > trailing_stop:
                        trailing_stop = new_trail
                else:
                    new_trail = swing_high[i] + atr[i] * trail_offset_atr
                    if trailing_stop == 0.0 or new_trail < trailing_stop:
                        trailing_stop = new_trail
            
            # ─── Exit checks (priority order) ───
            exit_triggered = False
            exit_price = 0.0
            
            # 1. Trailing stop
            if trail_active and trailing_stop != 0.0:
                if trade_dir == 1 and low[i] <= trailing_stop:
                    exit_triggered = True
                    exit_price = max(trailing_stop, low[i])  # worst case is low
                elif trade_dir == -1 and high[i] >= trailing_stop:
                    exit_triggered = True
                    exit_price = min(trailing_stop, high[i])
            
            # 2. Hard stop (if enabled and not already exited)
            if not exit_triggered and use_hard_stop and hard_stop != 0.0:
                if trade_dir == 1 and low[i] <= hard_stop:
                    exit_triggered = True
                    exit_price = max(hard_stop, low[i])
                elif trade_dir == -1 and high[i] >= hard_stop:
                    exit_triggered = True
                    exit_price = min(hard_stop, high[i])
            
            # 3. Break-even stop (if break-even was set but hard stop disabled)
            if not exit_triggered and breakeven_done and not use_hard_stop:
                if trade_dir == 1 and low[i] <= hard_stop:
                    exit_triggered = True
                    exit_price = max(hard_stop, low[i])
                elif trade_dir == -1 and high[i] >= hard_stop:
                    exit_triggered = True
                    exit_price = min(hard_stop, high[i])
            
            # 4. PSAR flip exit (only after maturity)
            if not exit_triggered and psar_exit_enabled:
                min_bars_psar = max(psar_maturity_min, int(forecast_bars * psar_maturity_pct))
                if bars_in_trade >= min_bars_psar:
                    if trade_dir == 1 and psar_flip_down[i]:
                        exit_triggered = True
                        exit_price = close[i]
                    elif trade_dir == -1 and psar_flip_up[i]:
                        exit_triggered = True
                        exit_price = close[i]
            
            # 5. T3 flip (primary exit — always active)
            if not exit_triggered:
                if trade_dir == 1 and t3_trend[i] == -1 and (i == 0 or t3_trend[i-1] != -1):
                    exit_triggered = True
                    exit_price = close[i]
                elif trade_dir == -1 and t3_trend[i] == 1 and (i == 0 or t3_trend[i-1] != 1):
                    exit_triggered = True
                    exit_price = close[i]
                # Also exit if trend goes flat from our direction
                elif trade_dir == 1 and t3_trend[i] != 1 and t3_trend[i] != trade_dir:
                    exit_triggered = True
                    exit_price = close[i]
                elif trade_dir == -1 and t3_trend[i] != -1 and t3_trend[i] != trade_dir:
                    exit_triggered = True
                    exit_price = close[i]
            
            # ─── Process exit ───
            if exit_triggered:
                # Apply spread
                if trade_dir == 1:
                    exit_price -= spread * 0.5
                else:
                    exit_price += spread * 0.5
                
                pnl = 0.0
                if trade_dir == 1:
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price
                
                pnl_atr = pnl / max(entry_atr, 0.01)
                
                # Commission
                commission = entry_price * commission_pct / 100.0 * 2  # round trip
                pnl -= commission
                
                # Equity update (using position size based on risk)
                if risk_pct > 0:
                    risk_amount = equity * risk_pct / 100.0
                    # Position size = risk_amount / (entry_atr * some_stop_distance)
                    # Simplified: PnL as % of equity based on ATR move
                    stop_distance = entry_atr * max(atr_stop_mult, trail_activate_atr, 2.0)
                    if stop_distance > 0:
                        pos_size = risk_amount / stop_distance
                        equity_change = pnl * pos_size / max(equity, 1.0) * equity
                    else:
                        equity_change = pnl
                else:
                    equity_change = pnl
                
                pnl_pct = equity_change / max(equity, 1.0) * 100.0
                equity += equity_change
                peak_equity = max(peak_equity, equity)
                
                if peak_equity > 0:
                    dd = (equity - peak_equity) / peak_equity * 100.0
                    if dd < max_dd_pct:
                        max_dd_pct = dd
                
                # Stats
                total_trades += 1
                total_bars_held += bars_in_trade
                
                if trade_count_for_sharpe < max_trades:
                    trade_returns[trade_count_for_sharpe] = pnl_pct
                    trade_count_for_sharpe += 1
                
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                    sum_win_atr += pnl_atr
                    consec_losses = 0
                else:
                    losses += 1
                    gross_loss += abs(pnl)
                    sum_loss_atr += abs(pnl_atr)
                    consec_losses += 1
                    if consec_losses > max_consec_losses:
                        max_consec_losses = consec_losses
                
                # Reset trade state
                in_trade = False
                trade_dir = 0
                
                # Check if we should immediately enter opposite direction (T3 flip)
                # This matches Pine V7 behavior: close old → open new in same bar
                if t3_trend[i] == 1 and signal_buy[i]:
                    in_trade = True
                    trade_dir = 1
                    entry_price = close[i] + spread * 0.5
                    entry_atr = atr[i]
                    hard_stop = entry_price - atr[i] * atr_stop_mult if use_hard_stop else 0.0
                    trailing_stop = 0.0
                    trail_active = False
                    breakeven_done = False
                    bars_in_trade = 0
                    forecast_bars = 30.0  # default estimate
                elif t3_trend[i] == -1 and signal_sell[i]:
                    in_trade = True
                    trade_dir = -1
                    entry_price = close[i] - spread * 0.5
                    entry_atr = atr[i]
                    hard_stop = entry_price + atr[i] * atr_stop_mult if use_hard_stop else 0.0
                    trailing_stop = 0.0
                    trail_active = False
                    breakeven_done = False
                    bars_in_trade = 0
                    forecast_bars = 30.0
        
        # ─── Entry (if not in trade) ───
        if not in_trade:
            if signal_buy[i]:
                in_trade = True
                trade_dir = 1
                entry_price = close[i] + spread * 0.5
                entry_atr = atr[i]
                hard_stop = entry_price - atr[i] * atr_stop_mult if use_hard_stop else 0.0
                trailing_stop = 0.0
                trail_active = False
                breakeven_done = False
                bars_in_trade = 0
                forecast_bars = 30.0
            elif signal_sell[i]:
                in_trade = True
                trade_dir = -1
                entry_price = close[i] - spread * 0.5
                entry_atr = atr[i]
                hard_stop = entry_price + atr[i] * atr_stop_mult if use_hard_stop else 0.0
                trailing_stop = 0.0
                trail_active = False
                breakeven_done = False
                bars_in_trade = 0
                forecast_bars = 30.0
    
    # ─── Compile results ───
    results = np.zeros(NUM_RESULTS, dtype=np.float64)
    results[R_TOTAL_TRADES] = total_trades
    results[R_WINS] = wins
    results[R_LOSSES] = losses
    results[R_WIN_RATE] = wins / max(total_trades, 1) * 100.0
    results[R_GROSS_PROFIT] = gross_profit
    results[R_GROSS_LOSS] = gross_loss
    results[R_NET_PROFIT] = gross_profit - gross_loss
    results[R_PROFIT_FACTOR] = gross_profit / max(gross_loss, 0.01)
    results[R_MAX_DRAWDOWN_PCT] = max_dd_pct  # negative value
    results[R_AVG_WIN_ATR] = sum_win_atr / max(wins, 1)
    results[R_AVG_LOSS_ATR] = sum_loss_atr / max(losses, 1)
    results[R_MAX_CONSEC_LOSS] = max_consec_losses
    results[R_AVG_BARS_HELD] = total_bars_held / max(total_trades, 1)
    results[R_FINAL_EQUITY] = equity
    
    # Expectancy: WR * avg_win - LR * avg_loss (in ATR units)
    wr = wins / max(total_trades, 1)
    lr = losses / max(total_trades, 1)
    avg_w = sum_win_atr / max(wins, 1)
    avg_l = sum_loss_atr / max(losses, 1)
    results[R_EXPECTANCY] = wr * avg_w - lr * avg_l
    
    # Sharpe ratio (annualized, assuming ~36 M10 bars per day, ~252 trading days)
    if trade_count_for_sharpe > 1:
        ret_slice = trade_returns[:trade_count_for_sharpe]
        mean_ret = 0.0
        for j in range(trade_count_for_sharpe):
            mean_ret += ret_slice[j]
        mean_ret /= trade_count_for_sharpe
        
        var_ret = 0.0
        for j in range(trade_count_for_sharpe):
            var_ret += (ret_slice[j] - mean_ret) ** 2
        var_ret /= max(trade_count_for_sharpe - 1, 1)
        std_ret = var_ret ** 0.5
        
        if std_ret > 0:
            # Approximate annualization
            trades_per_year = total_trades / max((end_idx - start_idx) / (36 * 252), 0.1)
            results[R_SHARPE] = mean_ret / std_ret * (trades_per_year ** 0.5)
        else:
            results[R_SHARPE] = 0.0
    
    # Recovery factor
    net = gross_profit - gross_loss
    abs_dd = abs(max_dd_pct)
    results[R_RECOVERY_FACTOR] = net / max(abs_dd * initial_equity / 100.0, 0.01)
    
    # Calmar ratio (annualized return / max drawdown)
    if abs_dd > 0:
        total_return_pct = (equity - initial_equity) / initial_equity * 100.0
        years = max((end_idx - start_idx) / (36 * 252), 0.1)
        annual_return_pct = total_return_pct / years
        results[R_CALMAR] = annual_return_pct / abs_dd
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Equity curve generator (for visualization)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def run_backtest_with_equity(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    t3_trend: np.ndarray,
    atr: np.ndarray,
    psar_flip_up: np.ndarray,
    psar_flip_down: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    signal_buy: np.ndarray,
    signal_sell: np.ndarray,
    trade_params: np.ndarray,
    start_idx: int = 0,
    end_idx: int = -1,
):
    """
    Same as run_backtest but also returns per-bar equity curve and trade log.
    Use for the final best parameter set visualization only (slower due to logging).
    
    Returns: (results, equity_curve, trade_log)
      - equity_curve: float64[n]
      - trade_log: float64[max_trades, 7] — [entry_bar, exit_bar, direction, entry_price, exit_price, pnl_pct, pnl_atr]
    """
    n = len(close)
    if end_idx < 0:
        end_idx = n

    atr_stop_mult      = trade_params[P_ATR_STOP_MULT]
    trail_activate_atr  = trade_params[P_TRAIL_ACTIVATE_ATR]
    trail_offset_atr    = trade_params[P_TRAIL_OFFSET_ATR]
    breakeven_atr       = trade_params[P_BREAKEVEN_ATR]
    psar_exit_enabled   = trade_params[P_PSAR_EXIT_ENABLED] > 0.5
    psar_maturity_pct   = trade_params[P_PSAR_MATURITY_PCT]
    psar_maturity_min   = int(trade_params[P_PSAR_MATURITY_MIN])
    spread              = trade_params[P_SPREAD_POINTS]
    commission_pct      = trade_params[P_COMMISSION_PCT]
    initial_equity      = trade_params[P_INITIAL_EQUITY]
    risk_pct            = trade_params[P_RISK_PCT]
    use_hard_stop       = atr_stop_mult > 0.0

    equity_curve = np.full(n, initial_equity, dtype=np.float64)
    max_logged = 5000
    trade_log = np.zeros((max_logged, 7), dtype=np.float64)
    log_idx = 0

    in_trade = False
    trade_dir = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    hard_stop = 0.0
    trailing_stop = 0.0
    trail_active = False
    breakeven_done = False
    bars_in_trade = 0
    forecast_bars = 30.0

    equity = initial_equity
    peak_equity = initial_equity
    max_dd_pct = 0.0

    total_trades = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sum_win_atr = 0.0
    sum_loss_atr = 0.0
    total_bars_held = 0
    consec_losses = 0
    max_consec_losses = 0

    max_ret = 10000
    trade_returns = np.zeros(max_ret, dtype=np.float64)
    trade_count_for_sharpe = 0

    def process_exit(exit_price_raw, bar_idx):
        """Inline exit processing. Returns updated values via nonlocal would be ideal
        but numba doesn't support closures with nonlocal well, so we inline."""
        pass  # Placeholder — logic is inlined below

    for i in range(start_idx, end_idx):
        equity_curve[i] = equity

        if in_trade:
            bars_in_trade += 1

            favorable_move = 0.0
            if trade_dir == 1:
                favorable_move = close[i] - entry_price
            else:
                favorable_move = entry_price - close[i]

            favorable_atr = favorable_move / max(entry_atr, 0.01)

            if not breakeven_done and favorable_atr >= breakeven_atr:
                breakeven_done = True
                if trade_dir == 1:
                    hard_stop = max(hard_stop, entry_price) if use_hard_stop else entry_price
                else:
                    hard_stop = min(hard_stop, entry_price) if use_hard_stop else entry_price
                if not use_hard_stop:
                    hard_stop = entry_price

            if not trail_active and favorable_atr >= trail_activate_atr:
                trail_active = True

            if trail_active:
                if trade_dir == 1:
                    new_trail = swing_low[i] - atr[i] * trail_offset_atr
                    if trailing_stop == 0.0 or new_trail > trailing_stop:
                        trailing_stop = new_trail
                else:
                    new_trail = swing_high[i] + atr[i] * trail_offset_atr
                    if trailing_stop == 0.0 or new_trail < trailing_stop:
                        trailing_stop = new_trail

            exit_triggered = False
            exit_price = 0.0

            if trail_active and trailing_stop != 0.0:
                if trade_dir == 1 and low[i] <= trailing_stop:
                    exit_triggered = True
                    exit_price = max(trailing_stop, low[i])
                elif trade_dir == -1 and high[i] >= trailing_stop:
                    exit_triggered = True
                    exit_price = min(trailing_stop, high[i])

            if not exit_triggered and (use_hard_stop or breakeven_done) and hard_stop != 0.0:
                if trade_dir == 1 and low[i] <= hard_stop:
                    exit_triggered = True
                    exit_price = max(hard_stop, low[i])
                elif trade_dir == -1 and high[i] >= hard_stop:
                    exit_triggered = True
                    exit_price = min(hard_stop, high[i])

            if not exit_triggered and psar_exit_enabled:
                min_bars_psar = max(psar_maturity_min, int(forecast_bars * psar_maturity_pct))
                if bars_in_trade >= min_bars_psar:
                    if trade_dir == 1 and psar_flip_down[i]:
                        exit_triggered = True
                        exit_price = close[i]
                    elif trade_dir == -1 and psar_flip_up[i]:
                        exit_triggered = True
                        exit_price = close[i]

            if not exit_triggered:
                if trade_dir == 1 and t3_trend[i] == -1 and (i == 0 or t3_trend[i-1] != -1):
                    exit_triggered = True
                    exit_price = close[i]
                elif trade_dir == -1 and t3_trend[i] == 1 and (i == 0 or t3_trend[i-1] != 1):
                    exit_triggered = True
                    exit_price = close[i]
                elif trade_dir == 1 and t3_trend[i] != 1 and t3_trend[i] != trade_dir:
                    exit_triggered = True
                    exit_price = close[i]
                elif trade_dir == -1 and t3_trend[i] != -1 and t3_trend[i] != trade_dir:
                    exit_triggered = True
                    exit_price = close[i]

            if exit_triggered:
                if trade_dir == 1:
                    exit_price -= spread * 0.5
                else:
                    exit_price += spread * 0.5

                pnl = 0.0
                if trade_dir == 1:
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price

                pnl_atr = pnl / max(entry_atr, 0.01)
                commission = entry_price * commission_pct / 100.0 * 2
                pnl -= commission

                if risk_pct > 0:
                    risk_amount = equity * risk_pct / 100.0
                    stop_distance = entry_atr * max(atr_stop_mult, trail_activate_atr, 2.0)
                    if stop_distance > 0:
                        pos_size = risk_amount / stop_distance
                        equity_change = pnl * pos_size / max(equity, 1.0) * equity
                    else:
                        equity_change = pnl
                else:
                    equity_change = pnl

                pnl_pct = equity_change / max(equity, 1.0) * 100.0
                equity += equity_change
                peak_equity = max(peak_equity, equity)

                if peak_equity > 0:
                    dd = (equity - peak_equity) / peak_equity * 100.0
                    if dd < max_dd_pct:
                        max_dd_pct = dd

                total_trades += 1
                total_bars_held += bars_in_trade

                if trade_count_for_sharpe < max_ret:
                    trade_returns[trade_count_for_sharpe] = pnl_pct
                    trade_count_for_sharpe += 1

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                    sum_win_atr += pnl_atr
                    consec_losses = 0
                else:
                    losses += 1
                    gross_loss += abs(pnl)
                    sum_loss_atr += abs(pnl_atr)
                    consec_losses += 1
                    if consec_losses > max_consec_losses:
                        max_consec_losses = consec_losses

                # Log trade
                if log_idx < max_logged:
                    trade_log[log_idx, 0] = entry_bar
                    trade_log[log_idx, 1] = i
                    trade_log[log_idx, 2] = trade_dir
                    trade_log[log_idx, 3] = entry_price
                    trade_log[log_idx, 4] = exit_price
                    trade_log[log_idx, 5] = pnl_pct
                    trade_log[log_idx, 6] = pnl_atr
                    log_idx += 1

                in_trade = False
                trade_dir = 0

                # Immediate re-entry on T3 flip
                if t3_trend[i] == 1 and signal_buy[i]:
                    in_trade = True
                    trade_dir = 1
                    entry_price = close[i] + spread * 0.5
                    entry_atr = atr[i]
                    entry_bar = i
                    hard_stop = entry_price - atr[i] * atr_stop_mult if use_hard_stop else 0.0
                    trailing_stop = 0.0
                    trail_active = False
                    breakeven_done = False
                    bars_in_trade = 0
                    forecast_bars = 30.0
                elif t3_trend[i] == -1 and signal_sell[i]:
                    in_trade = True
                    trade_dir = -1
                    entry_price = close[i] - spread * 0.5
                    entry_atr = atr[i]
                    entry_bar = i
                    hard_stop = entry_price + atr[i] * atr_stop_mult if use_hard_stop else 0.0
                    trailing_stop = 0.0
                    trail_active = False
                    breakeven_done = False
                    bars_in_trade = 0
                    forecast_bars = 30.0

        if not in_trade:
            if signal_buy[i]:
                in_trade = True
                trade_dir = 1
                entry_price = close[i] + spread * 0.5
                entry_atr = atr[i]
                entry_bar = i
                hard_stop = entry_price - atr[i] * atr_stop_mult if use_hard_stop else 0.0
                trailing_stop = 0.0
                trail_active = False
                breakeven_done = False
                bars_in_trade = 0
                forecast_bars = 30.0
            elif signal_sell[i]:
                in_trade = True
                trade_dir = -1
                entry_price = close[i] - spread * 0.5
                entry_atr = atr[i]
                entry_bar = i
                hard_stop = entry_price + atr[i] * atr_stop_mult if use_hard_stop else 0.0
                trailing_stop = 0.0
                trail_active = False
                breakeven_done = False
                bars_in_trade = 0
                forecast_bars = 30.0

        equity_curve[i] = equity

    # Fill remaining equity curve
    for i in range(end_idx, n):
        equity_curve[i] = equity

    # Compile results
    results = np.zeros(NUM_RESULTS, dtype=np.float64)
    results[R_TOTAL_TRADES] = total_trades
    results[R_WINS] = wins
    results[R_LOSSES] = losses
    results[R_WIN_RATE] = wins / max(total_trades, 1) * 100.0
    results[R_GROSS_PROFIT] = gross_profit
    results[R_GROSS_LOSS] = gross_loss
    results[R_NET_PROFIT] = gross_profit - gross_loss
    results[R_PROFIT_FACTOR] = gross_profit / max(gross_loss, 0.01)
    results[R_MAX_DRAWDOWN_PCT] = max_dd_pct
    results[R_AVG_WIN_ATR] = sum_win_atr / max(wins, 1)
    results[R_AVG_LOSS_ATR] = sum_loss_atr / max(losses, 1)
    results[R_MAX_CONSEC_LOSS] = max_consec_losses
    results[R_AVG_BARS_HELD] = total_bars_held / max(total_trades, 1)
    results[R_FINAL_EQUITY] = equity

    wr = wins / max(total_trades, 1)
    lr = losses / max(total_trades, 1)
    avg_w = sum_win_atr / max(wins, 1)
    avg_l = sum_loss_atr / max(losses, 1)
    results[R_EXPECTANCY] = wr * avg_w - lr * avg_l

    if trade_count_for_sharpe > 1:
        ret_slice = trade_returns[:trade_count_for_sharpe]
        mean_ret = 0.0
        for j in range(trade_count_for_sharpe):
            mean_ret += ret_slice[j]
        mean_ret /= trade_count_for_sharpe
        var_ret = 0.0
        for j in range(trade_count_for_sharpe):
            var_ret += (ret_slice[j] - mean_ret) ** 2
        var_ret /= max(trade_count_for_sharpe - 1, 1)
        std_ret = var_ret ** 0.5
        if std_ret > 0:
            trades_per_year = total_trades / max((end_idx - start_idx) / (36 * 252), 0.1)
            results[R_SHARPE] = mean_ret / std_ret * (trades_per_year ** 0.5)

    net = gross_profit - gross_loss
    abs_dd = abs(max_dd_pct)
    results[R_RECOVERY_FACTOR] = net / max(abs_dd * initial_equity / 100.0, 0.01)

    if abs_dd > 0:
        total_return_pct = (equity - initial_equity) / initial_equity * 100.0
        years = max((end_idx - start_idx) / (36 * 252), 0.1)
        annual_return_pct = total_return_pct / years
        results[R_CALMAR] = annual_return_pct / abs_dd

    return results, equity_curve, trade_log[:log_idx]

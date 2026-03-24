"""
Module 2 — Numba @njit Backtest Engine V2 (MEGA Optimizer)
Bitmask confluence filtering inside the backtest loop.

Key changes from V1:
- Confluence via bitmask: voter_bits + gate_bits + min_agree
- Break-even is Tier 1 (index 0 in trade params)
- Entry: T3 flip AND all gates pass AND N voters agree
- Exit priority: trailing > hard stop > break-even > PSAR exit > T3 flip
"""

import numpy as np
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
# Trade param indices
# ─────────────────────────────────────────────────────────────────────────────
P_BREAK_EVEN_ATR     = 0   # 0 = disabled (TIER 1!)
P_BE_PROFIT_ATR      = 1   # extra profit above entry when BE triggers (0 = move to entry exactly)
P_ATR_STOP_MULT      = 2   # 0 = disabled
P_TRAIL_ACTIVATE_ATR = 3   # 999 = disabled
P_TRAIL_OFFSET_ATR   = 4
P_PSAR_EXIT_ENABLED  = 5   # 0 or 1
P_PSAR_MATURITY_MIN  = 6
P_SWING_LOOKBACK     = 7
P_SPREAD_POINTS      = 8   # SWEEPABLE: test $0.10 to $1.00
P_INITIAL_EQUITY     = 9
P_RISK_PCT           = 10
P_USE_OPEN_ENTRY     = 11  # 1 = enter on open[i+1] (realistic), 0 = enter on close[i] (legacy)
NUM_TRADE_PARAMS     = 12

# Result indices
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


def make_trade_params(
    break_even_atr: float = 0.0,
    be_profit_atr: float = 0.0,     # extra profit above entry on BE
    atr_stop_mult: float = 0.0,
    trail_activate_atr: float = 999.0,
    trail_offset_atr: float = 0.5,
    psar_exit_enabled: float = 0.0,
    psar_maturity_min: float = 10.0,
    swing_lookback: float = 10.0,
    spread_points: float = 0.30,
    initial_equity: float = 10000.0,
    risk_pct: float = 0.0,
    use_open_entry: float = 1.0,    # 1 = realistic (open[i+1]), 0 = legacy (close[i])
) -> np.ndarray:
    p = np.zeros(NUM_TRADE_PARAMS, dtype=np.float64)
    p[P_BREAK_EVEN_ATR]     = break_even_atr
    p[P_BE_PROFIT_ATR]      = be_profit_atr
    p[P_ATR_STOP_MULT]      = atr_stop_mult
    p[P_TRAIL_ACTIVATE_ATR] = trail_activate_atr
    p[P_TRAIL_OFFSET_ATR]   = trail_offset_atr
    p[P_PSAR_EXIT_ENABLED]  = psar_exit_enabled
    p[P_PSAR_MATURITY_MIN]  = psar_maturity_min
    p[P_SWING_LOOKBACK]     = swing_lookback
    p[P_SPREAD_POINTS]      = spread_points
    p[P_INITIAL_EQUITY]     = initial_equity
    p[P_RISK_PCT]           = risk_pct
    p[P_USE_OPEN_ENTRY]     = use_open_entry
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Numba popcount helper
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline='always')
def popcount(x):
    """Count set bits in an int8/int."""
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Core Backtest — Numba @njit
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def run_backtest(
    # OHLCV
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    open_next: np.ndarray,  # open[i+1] for realistic entry
    # T3 signals
    t3_trend: np.ndarray,
    raw_flip_up: np.ndarray, raw_flip_down: np.ndarray,
    atr: np.ndarray,
    # Indicator bitmasks — ALL 8 equal (int16 per bar)
    ind_bits_buy: np.ndarray, ind_bits_sell: np.ndarray,
    # Confluence config
    active_mask: int, min_agree: int,
    # Exit helpers
    psar_flip_up: np.ndarray, psar_flip_down: np.ndarray,
    swing_high: np.ndarray, swing_low: np.ndarray,
    # Trade management
    trade_params: np.ndarray,
    # Range
    start_idx: int = 0, end_idx: int = -1,
) -> np.ndarray:
    """Bar-by-bar backtest. All 8 indicators equal, entry on open[i+1]."""
    n = len(close)
    if end_idx < 0:
        end_idx = n

    # Extract trade params
    be_atr           = trade_params[P_BREAK_EVEN_ATR]
    be_profit        = trade_params[P_BE_PROFIT_ATR]
    atr_stop_mult    = trade_params[P_ATR_STOP_MULT]
    trail_act_atr    = trade_params[P_TRAIL_ACTIVATE_ATR]
    trail_off_atr    = trade_params[P_TRAIL_OFFSET_ATR]
    psar_exit_on     = trade_params[P_PSAR_EXIT_ENABLED] > 0.5
    psar_mat_min     = int(trade_params[P_PSAR_MATURITY_MIN])
    spread           = trade_params[P_SPREAD_POINTS]
    initial_equity   = trade_params[P_INITIAL_EQUITY]
    use_open_entry   = trade_params[P_USE_OPEN_ENTRY] > 0.5
    use_hard_stop    = atr_stop_mult > 0.0
    use_be           = be_atr > 0.0
    use_trail        = trail_act_atr < 900.0

    # State
    in_trade = False; trade_dir = 0
    entry_price = 0.0; entry_atr = 0.0
    hard_stop = 0.0; trailing_stop = 0.0
    trail_active = False; be_done = False
    bars_in_trade = 0

    equity = initial_equity; peak_equity = initial_equity; max_dd_pct = 0.0

    total_trades = 0; wins = 0; losses = 0
    gross_profit = 0.0; gross_loss = 0.0
    sum_win_atr = 0.0; sum_loss_atr = 0.0
    total_bars_held = 0; consec_losses = 0; max_consec_losses = 0

    max_ret = 10000
    trade_returns = np.zeros(max_ret, dtype=np.float64)
    tc_sharpe = 0

    def _check_entry_buy(i):
        if not raw_flip_up[i]: return False
        if min_agree > 0:
            agree = popcount(ind_bits_buy[i] & active_mask)
            if agree < min_agree: return False
        return True

    def _check_entry_sell(i):
        if not raw_flip_down[i]: return False
        if min_agree > 0:
            agree = popcount(ind_bits_sell[i] & active_mask)
            if agree < min_agree: return False
        return True

    def _enter_long(i):
        nonlocal in_trade, trade_dir, entry_price, entry_atr
        nonlocal hard_stop, trailing_stop, trail_active, be_done, bars_in_trade
        in_trade = True; trade_dir = 1
        # FIX #1: Entry on open[i+1] (realistic) or close[i] (legacy)
        base_price = open_next[i] if use_open_entry else close[i]
        entry_price = base_price + spread * 0.5
        entry_atr = atr[i]
        hard_stop = entry_price - atr[i] * atr_stop_mult if use_hard_stop else 0.0
        trailing_stop = 0.0; trail_active = False; be_done = False; bars_in_trade = 0

    def _enter_short(i):
        nonlocal in_trade, trade_dir, entry_price, entry_atr
        nonlocal hard_stop, trailing_stop, trail_active, be_done, bars_in_trade
        in_trade = True; trade_dir = -1
        base_price = open_next[i] if use_open_entry else close[i]
        entry_price = base_price - spread * 0.5
        entry_atr = atr[i]
        hard_stop = entry_price + atr[i] * atr_stop_mult if use_hard_stop else 0.0
        trailing_stop = 0.0; trail_active = False; be_done = False; bars_in_trade = 0

    for i in range(start_idx, end_idx):
        if in_trade:
            bars_in_trade += 1
            fav = (close[i] - entry_price) if trade_dir == 1 else (entry_price - close[i])
            fav_atr = fav / max(entry_atr, 0.01)

            # Break-even (with optional profit lock)
            if use_be and not be_done and fav_atr >= be_atr:
                be_done = True
                # FIX: BE can move to entry + profit (not just entry)
                be_level = entry_price + be_profit * entry_atr * trade_dir
                if trade_dir == 1:
                    hard_stop = max(hard_stop, be_level) if use_hard_stop else be_level
                else:
                    hard_stop = min(hard_stop, be_level) if use_hard_stop else be_level
                if not use_hard_stop:
                    hard_stop = be_level

            # Trailing activation
            if use_trail and not trail_active and fav_atr >= trail_act_atr:
                trail_active = True

            # Update trailing stop
            if trail_active:
                if trade_dir == 1:
                    new_t = swing_low[i] - atr[i] * trail_off_atr
                    if trailing_stop == 0.0 or new_t > trailing_stop: trailing_stop = new_t
                else:
                    new_t = swing_high[i] + atr[i] * trail_off_atr
                    if trailing_stop == 0.0 or new_t < trailing_stop: trailing_stop = new_t

            # ── Exit checks ──
            exit_triggered = False; exit_price = 0.0

            # 1. Trailing stop
            if trail_active and trailing_stop != 0.0:
                if trade_dir == 1 and low[i] <= trailing_stop:
                    exit_triggered = True; exit_price = max(trailing_stop, low[i])
                elif trade_dir == -1 and high[i] >= trailing_stop:
                    exit_triggered = True; exit_price = min(trailing_stop, high[i])

            # 2. Hard stop
            if not exit_triggered and (use_hard_stop or (use_be and be_done)) and hard_stop != 0.0:
                if trade_dir == 1 and low[i] <= hard_stop:
                    exit_triggered = True; exit_price = max(hard_stop, low[i])
                elif trade_dir == -1 and high[i] >= hard_stop:
                    exit_triggered = True; exit_price = min(hard_stop, high[i])

            # 3. PSAR exit
            if not exit_triggered and psar_exit_on:
                if bars_in_trade >= psar_mat_min:
                    if trade_dir == 1 and psar_flip_down[i]:
                        exit_triggered = True; exit_price = close[i]
                    elif trade_dir == -1 and psar_flip_up[i]:
                        exit_triggered = True; exit_price = close[i]

            # 4. T3 flip exit
            if not exit_triggered:
                if trade_dir == 1 and t3_trend[i] != 1 and (i == 0 or t3_trend[i-1] == 1):
                    exit_triggered = True; exit_price = close[i]
                elif trade_dir == -1 and t3_trend[i] != -1 and (i == 0 or t3_trend[i-1] == -1):
                    exit_triggered = True; exit_price = close[i]

            if exit_triggered:
                if trade_dir == 1: exit_price -= spread * 0.5
                else: exit_price += spread * 0.5

                pnl = (exit_price - entry_price) if trade_dir == 1 else (entry_price - exit_price)
                pnl_atr = pnl / max(entry_atr, 0.01)

                equity += pnl
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    dd = (equity - peak_equity) / peak_equity * 100.0
                    if dd < max_dd_pct: max_dd_pct = dd

                total_trades += 1; total_bars_held += bars_in_trade
                pnl_pct = pnl / max(entry_price, 0.01) * 100.0
                if tc_sharpe < max_ret:
                    trade_returns[tc_sharpe] = pnl_pct; tc_sharpe += 1

                if pnl > 0:
                    wins += 1; gross_profit += pnl; sum_win_atr += pnl_atr; consec_losses = 0
                else:
                    losses += 1; gross_loss += abs(pnl); sum_loss_atr += abs(pnl_atr)
                    consec_losses += 1
                    if consec_losses > max_consec_losses: max_consec_losses = consec_losses

                in_trade = False; trade_dir = 0

                # Immediate re-entry
                if _check_entry_buy(i): _enter_long(i)
                elif _check_entry_sell(i): _enter_short(i)

        # Entry (not in trade)
        if not in_trade:
            if _check_entry_buy(i): _enter_long(i)
            elif _check_entry_sell(i): _enter_short(i)

    # ── Results ──
    results = np.zeros(NUM_RESULTS, dtype=np.float64)
    results[R_TOTAL_TRADES] = total_trades
    results[R_WINS] = wins; results[R_LOSSES] = losses
    results[R_WIN_RATE] = wins / max(total_trades, 1) * 100.0
    results[R_GROSS_PROFIT] = gross_profit; results[R_GROSS_LOSS] = gross_loss
    results[R_NET_PROFIT] = gross_profit - gross_loss
    results[R_PROFIT_FACTOR] = gross_profit / max(gross_loss, 0.01)
    results[R_MAX_DRAWDOWN_PCT] = max_dd_pct
    results[R_AVG_WIN_ATR] = sum_win_atr / max(wins, 1)
    results[R_AVG_LOSS_ATR] = sum_loss_atr / max(losses, 1)
    results[R_MAX_CONSEC_LOSS] = max_consec_losses
    results[R_AVG_BARS_HELD] = total_bars_held / max(total_trades, 1)
    results[R_FINAL_EQUITY] = equity

    wr = wins / max(total_trades, 1); lr = losses / max(total_trades, 1)
    results[R_EXPECTANCY] = wr * sum_win_atr / max(wins, 1) - lr * sum_loss_atr / max(losses, 1)

    if tc_sharpe > 1:
        ret = trade_returns[:tc_sharpe]
        mean_r = 0.0
        for j in range(tc_sharpe): mean_r += ret[j]
        mean_r /= tc_sharpe
        var_r = 0.0
        for j in range(tc_sharpe): var_r += (ret[j] - mean_r) ** 2
        var_r /= max(tc_sharpe - 1, 1)
        std_r = var_r ** 0.5
        if std_r > 0:
            tpy = total_trades / max((end_idx - start_idx) / (36 * 252), 0.1)
            results[R_SHARPE] = mean_r / std_r * (tpy ** 0.5)

    net = gross_profit - gross_loss; abs_dd = abs(max_dd_pct)
    results[R_RECOVERY_FACTOR] = net / max(abs_dd * initial_equity / 100.0, 0.01)
    if abs_dd > 0:
        total_ret = (equity - initial_equity) / initial_equity * 100.0
        years = max((end_idx - start_idx) / (36 * 252), 0.1)
        results[R_CALMAR] = (total_ret / years) / abs_dd

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Equity curve variant (for visualization of best params)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def run_backtest_with_equity(
    open_, high, low, close,
    t3_trend, raw_flip_up, raw_flip_down, atr,
    voter_bits_buy, voter_bits_sell, gate_bits,
    active_voter_mask, active_gate_mask, min_voters_agree,
    psar_flip_up, psar_flip_down, swing_high, swing_low,
    trade_params,
    start_idx=0, end_idx=-1,
):
    """Same as run_backtest but also returns equity curve and trade log."""
    n = len(close)
    if end_idx < 0: end_idx = n

    be_atr = trade_params[P_BREAK_EVEN_ATR]
    atr_stop_mult = trade_params[P_ATR_STOP_MULT]
    trail_act_atr = trade_params[P_TRAIL_ACTIVATE_ATR]
    trail_off_atr = trade_params[P_TRAIL_OFFSET_ATR]
    psar_exit_on = trade_params[P_PSAR_EXIT_ENABLED] > 0.5
    psar_mat_min = int(trade_params[P_PSAR_MATURITY_MIN])
    spread = trade_params[P_SPREAD_POINTS]
    initial_equity = trade_params[P_INITIAL_EQUITY]
    use_hard_stop = atr_stop_mult > 0.0
    use_be = be_atr > 0.0
    use_trail = trail_act_atr < 900.0

    equity_curve = np.full(n, initial_equity, dtype=np.float64)
    max_logged = 5000
    trade_log = np.zeros((max_logged, 7), dtype=np.float64)
    log_idx = 0

    in_trade = False; trade_dir = 0
    entry_price = 0.0; entry_atr = 0.0; entry_bar = 0
    hard_stop = 0.0; trailing_stop = 0.0
    trail_active = False; be_done = False; bars_in_trade = 0

    equity = initial_equity; peak_equity = initial_equity; max_dd_pct = 0.0
    total_trades = 0; wins = 0; losses = 0
    gross_profit = 0.0; gross_loss = 0.0
    sum_win_atr = 0.0; sum_loss_atr = 0.0
    total_bars_held = 0; consec_losses = 0; max_consec_losses = 0
    max_ret = 10000; trade_returns = np.zeros(max_ret, dtype=np.float64); tc_sharpe = 0

    for i in range(start_idx, end_idx):
        equity_curve[i] = equity

        if in_trade:
            bars_in_trade += 1
            fav = (close[i] - entry_price) if trade_dir == 1 else (entry_price - close[i])
            fav_atr = fav / max(entry_atr, 0.01)

            if use_be and not be_done and fav_atr >= be_atr:
                be_done = True
                if trade_dir == 1:
                    hard_stop = max(hard_stop, entry_price) if use_hard_stop else entry_price
                else:
                    hard_stop = min(hard_stop, entry_price) if use_hard_stop else entry_price
                if not use_hard_stop: hard_stop = entry_price

            if use_trail and not trail_active and fav_atr >= trail_act_atr:
                trail_active = True

            if trail_active:
                if trade_dir == 1:
                    new_t = swing_low[i] - atr[i] * trail_off_atr
                    if trailing_stop == 0.0 or new_t > trailing_stop: trailing_stop = new_t
                else:
                    new_t = swing_high[i] + atr[i] * trail_off_atr
                    if trailing_stop == 0.0 or new_t < trailing_stop: trailing_stop = new_t

            exit_triggered = False; exit_price = 0.0

            if trail_active and trailing_stop != 0.0:
                if trade_dir == 1 and low[i] <= trailing_stop:
                    exit_triggered = True; exit_price = max(trailing_stop, low[i])
                elif trade_dir == -1 and high[i] >= trailing_stop:
                    exit_triggered = True; exit_price = min(trailing_stop, high[i])

            if not exit_triggered and (use_hard_stop or (use_be and be_done)) and hard_stop != 0.0:
                if trade_dir == 1 and low[i] <= hard_stop:
                    exit_triggered = True; exit_price = max(hard_stop, low[i])
                elif trade_dir == -1 and high[i] >= hard_stop:
                    exit_triggered = True; exit_price = min(hard_stop, high[i])

            if not exit_triggered and psar_exit_on and bars_in_trade >= psar_mat_min:
                if trade_dir == 1 and psar_flip_down[i]:
                    exit_triggered = True; exit_price = close[i]
                elif trade_dir == -1 and psar_flip_up[i]:
                    exit_triggered = True; exit_price = close[i]

            if not exit_triggered:
                if trade_dir == 1 and t3_trend[i] != 1 and (i == 0 or t3_trend[i-1] == 1):
                    exit_triggered = True; exit_price = close[i]
                elif trade_dir == -1 and t3_trend[i] != -1 and (i == 0 or t3_trend[i-1] == -1):
                    exit_triggered = True; exit_price = close[i]

            if exit_triggered:
                if trade_dir == 1: exit_price -= spread * 0.5
                else: exit_price += spread * 0.5
                pnl = (exit_price - entry_price) if trade_dir == 1 else (entry_price - exit_price)
                pnl_atr = pnl / max(entry_atr, 0.01)
                equity += pnl; peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    dd = (equity - peak_equity) / peak_equity * 100.0
                    if dd < max_dd_pct: max_dd_pct = dd
                total_trades += 1; total_bars_held += bars_in_trade
                pnl_pct = pnl / max(entry_price, 0.01) * 100.0
                if tc_sharpe < max_ret: trade_returns[tc_sharpe] = pnl_pct; tc_sharpe += 1
                if pnl > 0:
                    wins += 1; gross_profit += pnl; sum_win_atr += pnl_atr; consec_losses = 0
                else:
                    losses += 1; gross_loss += abs(pnl); sum_loss_atr += abs(pnl_atr)
                    consec_losses += 1
                    if consec_losses > max_consec_losses: max_consec_losses = consec_losses
                if log_idx < max_logged:
                    trade_log[log_idx, 0] = entry_bar; trade_log[log_idx, 1] = i
                    trade_log[log_idx, 2] = trade_dir; trade_log[log_idx, 3] = entry_price
                    trade_log[log_idx, 4] = exit_price; trade_log[log_idx, 5] = pnl_pct
                    trade_log[log_idx, 6] = pnl_atr; log_idx += 1
                in_trade = False; trade_dir = 0

                # Re-entry check
                buy_ok = raw_flip_up[i] and (gate_bits[i] & active_gate_mask) == active_gate_mask
                if buy_ok and min_voters_agree > 0:
                    buy_ok = popcount(voter_bits_buy[i] & active_voter_mask) >= min_voters_agree
                sell_ok = raw_flip_down[i] and (gate_bits[i] & active_gate_mask) == active_gate_mask
                if sell_ok and min_voters_agree > 0:
                    sell_ok = popcount(voter_bits_sell[i] & active_voter_mask) >= min_voters_agree
                if buy_ok:
                    in_trade = True; trade_dir = 1
                    entry_price = close[i] + spread * 0.5; entry_atr = atr[i]; entry_bar = i
                    hard_stop = entry_price - atr[i] * atr_stop_mult if use_hard_stop else 0.0
                    trailing_stop = 0.0; trail_active = False; be_done = False; bars_in_trade = 0
                elif sell_ok:
                    in_trade = True; trade_dir = -1
                    entry_price = close[i] - spread * 0.5; entry_atr = atr[i]; entry_bar = i
                    hard_stop = entry_price + atr[i] * atr_stop_mult if use_hard_stop else 0.0
                    trailing_stop = 0.0; trail_active = False; be_done = False; bars_in_trade = 0

        if not in_trade:
            buy_ok = raw_flip_up[i] and (gate_bits[i] & active_gate_mask) == active_gate_mask
            if buy_ok and min_voters_agree > 0:
                buy_ok = popcount(voter_bits_buy[i] & active_voter_mask) >= min_voters_agree
            sell_ok = raw_flip_down[i] and (gate_bits[i] & active_gate_mask) == active_gate_mask
            if sell_ok and min_voters_agree > 0:
                sell_ok = popcount(voter_bits_sell[i] & active_voter_mask) >= min_voters_agree
            if buy_ok:
                in_trade = True; trade_dir = 1
                entry_price = close[i] + spread * 0.5; entry_atr = atr[i]; entry_bar = i
                hard_stop = entry_price - atr[i] * atr_stop_mult if use_hard_stop else 0.0
                trailing_stop = 0.0; trail_active = False; be_done = False; bars_in_trade = 0
            elif sell_ok:
                in_trade = True; trade_dir = -1
                entry_price = close[i] - spread * 0.5; entry_atr = atr[i]; entry_bar = i
                hard_stop = entry_price + atr[i] * atr_stop_mult if use_hard_stop else 0.0
                trailing_stop = 0.0; trail_active = False; be_done = False; bars_in_trade = 0

        equity_curve[i] = equity

    for i in range(end_idx, n): equity_curve[i] = equity

    results = np.zeros(NUM_RESULTS, dtype=np.float64)
    results[R_TOTAL_TRADES] = total_trades
    results[R_WINS] = wins; results[R_LOSSES] = losses
    results[R_WIN_RATE] = wins / max(total_trades, 1) * 100.0
    results[R_GROSS_PROFIT] = gross_profit; results[R_GROSS_LOSS] = gross_loss
    results[R_NET_PROFIT] = gross_profit - gross_loss
    results[R_PROFIT_FACTOR] = gross_profit / max(gross_loss, 0.01)
    results[R_MAX_DRAWDOWN_PCT] = max_dd_pct
    results[R_AVG_WIN_ATR] = sum_win_atr / max(wins, 1)
    results[R_AVG_LOSS_ATR] = sum_loss_atr / max(losses, 1)
    results[R_MAX_CONSEC_LOSS] = max_consec_losses
    results[R_AVG_BARS_HELD] = total_bars_held / max(total_trades, 1)
    results[R_FINAL_EQUITY] = equity
    wr = wins / max(total_trades, 1); lr = losses / max(total_trades, 1)
    results[R_EXPECTANCY] = wr * sum_win_atr / max(wins, 1) - lr * sum_loss_atr / max(losses, 1)
    if tc_sharpe > 1:
        ret = trade_returns[:tc_sharpe]; mean_r = 0.0
        for j in range(tc_sharpe): mean_r += ret[j]
        mean_r /= tc_sharpe; var_r = 0.0
        for j in range(tc_sharpe): var_r += (ret[j] - mean_r) ** 2
        var_r /= max(tc_sharpe - 1, 1); std_r = var_r ** 0.5
        if std_r > 0:
            tpy = total_trades / max((end_idx - start_idx) / (36 * 252), 0.1)
            results[R_SHARPE] = mean_r / std_r * (tpy ** 0.5)
    net = gross_profit - gross_loss; abs_dd = abs(max_dd_pct)
    results[R_RECOVERY_FACTOR] = net / max(abs_dd * initial_equity / 100.0, 0.01)
    if abs_dd > 0:
        total_ret = (equity - initial_equity) / initial_equity * 100.0
        years = max((end_idx - start_idx) / (36 * 252), 0.1)
        results[R_CALMAR] = (total_ret / years) / abs_dd

    return results, equity_curve, trade_log[:log_idx]

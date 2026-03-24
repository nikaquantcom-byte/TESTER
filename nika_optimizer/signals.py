"""
Module 2 — Signal Engine (Vectorized with NumPy)
Translates all Pine Script V7 Pro indicators to pure NumPy for maximum speed.

ALL indicators computed independently, then combined via configurable confluence:
- Each indicator can be ON or OFF
- Confluence mode: how many must agree (count, percentage, or all)
- No hardcoded weights — pure binary agree/disagree per indicator

All functions operate on numpy arrays for compatibility with the Numba backtest engine.
"""

import numpy as np
import warnings
from typing import NamedTuple

# Suppress harmless divide-by-zero warnings from np.where
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')


# ─────────────────────────────────────────────────────────────────────────────
# 2A. Tilson T3 — the core trend engine
# ─────────────────────────────────────────────────────────────────────────────

def ema_np(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — iterative for accuracy."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def gd_np(src: np.ndarray, period: int, vfactor: float = 0.7) -> np.ndarray:
    """Generalized Double smoothing: GD = EMA*(1+vf) - EMA(EMA)*vf"""
    e1 = ema_np(src, period)
    e2 = ema_np(e1, period)
    return e1 * (1 + vfactor) - e2 * vfactor


def t3_np(src: np.ndarray, period: int, vfactor: float = 0.7) -> np.ndarray:
    """Tilson T3 — triple GD smoothing. vfactor = T-Factor."""
    return gd_np(gd_np(gd_np(src, period, vfactor), period, vfactor), period, vfactor)


# ─────────────────────────────────────────────────────────────────────────────
# 2B. True Strength Index (TSI)
# ─────────────────────────────────────────────────────────────────────────────

def tsi_np(close: np.ndarray, short_len: int = 5, long_len: int = 25, signal_len: int = 14):
    """
    TSI = 100 * EMA(EMA(momentum, long), short) / EMA(EMA(|momentum|, long), short)
    Returns: (tsi_value, tsi_signal, tsi_up)
    """
    mom = np.diff(close, prepend=close[0])
    abs_mom = np.abs(mom)
    smooth_mom = ema_np(ema_np(mom, long_len), short_len)
    smooth_abs = ema_np(ema_np(abs_mom, long_len), short_len)
    tsi_val = np.where(smooth_abs != 0, 100.0 * smooth_mom / smooth_abs, 0.0)
    tsi_sig = ema_np(tsi_val, signal_len)
    tsi_up = tsi_val > tsi_sig
    return tsi_val, tsi_sig, tsi_up


# ─────────────────────────────────────────────────────────────────────────────
# 2C. ADX (Average Directional Index)
# ─────────────────────────────────────────────────────────────────────────────

def ema_wilder(data: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (RMA) — equivalent to EMA with alpha = 1/period."""
    alpha = 1.0 / period
    out = np.empty_like(data, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def adx_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """ADX calculation. Returns: (di_plus, di_minus, adx_value)"""
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i-1]
        l_diff = low[i-1] - low[i]
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0

    atr_smooth = ema_wilder(tr, period)
    plus_dm_smooth = ema_wilder(plus_dm, period)
    minus_dm_smooth = ema_wilder(minus_dm, period)

    di_plus = np.where(atr_smooth != 0, 100.0 * plus_dm_smooth / atr_smooth, 0.0)
    di_minus = np.where(atr_smooth != 0, 100.0 * minus_dm_smooth / atr_smooth, 0.0)

    di_sum = di_plus + di_minus
    dx = np.where(di_sum != 0, 100.0 * np.abs(di_plus - di_minus) / di_sum, 0.0)
    adx_val = ema_wilder(dx, period)

    return di_plus, di_minus, adx_val


# ─────────────────────────────────────────────────────────────────────────────
# 2D. Parabolic SAR
# ─────────────────────────────────────────────────────────────────────────────

def psar_np(high: np.ndarray, low: np.ndarray,
            start: float = 0.02, inc: float = 0.02, maximum: float = 0.2):
    """Parabolic SAR. Returns: (psar_value, psar_bullish)"""
    n = len(high)
    psar = np.zeros(n, dtype=np.float64)
    psarbull = np.ones(n, dtype=np.bool_)

    af = start
    hp = high[0]
    lp = low[0]
    trend = 1
    psar[0] = low[0]

    for i in range(1, n):
        if trend == 1:
            psar[i] = psar[i-1] + af * (hp - psar[i-1])
            psar[i] = min(psar[i], low[i-1])
            if i >= 2:
                psar[i] = min(psar[i], low[i-2])
            if low[i] < psar[i]:
                trend = -1
                psar[i] = hp
                lp = low[i]
                af = start
            else:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + inc, maximum)
        else:
            psar[i] = psar[i-1] + af * (lp - psar[i-1])
            psar[i] = max(psar[i], high[i-1])
            if i >= 2:
                psar[i] = max(psar[i], high[i-2])
            if high[i] > psar[i]:
                trend = 1
                psar[i] = lp
                hp = high[i]
                af = start
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + inc, maximum)

        psarbull[i] = (trend == 1)

    return psar, psarbull


# ─────────────────────────────────────────────────────────────────────────────
# 2E. ATR, RSI, Volume SMA, Swing levels
# ─────────────────────────────────────────────────────────────────────────────

def atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return ema_wilder(tr, period)


def rsi_np(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI using Wilder's smoothing (matches TradingView)."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = ema_wilder(gain, period)
    avg_loss = ema_wilder(loss, period)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def sma_np(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    out = np.empty_like(data, dtype=np.float64)
    cumsum = np.cumsum(data)
    out[:period] = cumsum[:period] / np.arange(1, period + 1)
    out[period:] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def highest_np(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling highest."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    for i in range(min(period, n)):
        out[i] = np.max(data[:i+1])
    if n >= period:
        windows = sliding_window_view(data, period)
        out[period-1:] = np.max(windows, axis=1)
    return out


def lowest_np(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling lowest."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    for i in range(min(period, n)):
        out[i] = np.min(data[:i+1])
    if n >= period:
        windows = sliding_window_view(data, period)
        out[period-1:] = np.min(windows, axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2F. Regime Detection
# ─────────────────────────────────────────────────────────────────────────────

REGIME_RANGE = 0
REGIME_TREND = 1
REGIME_VOL = 2


def compute_regime(
    atr_val: np.ndarray,
    t3_line: np.ndarray,
    regime_lookback: int = 100,
    trend_strength_len: int = 5,
    trend_thresh: float = 0.60,
    volatile_thresh: float = 1.30,
    mintick: float = 0.01,
) -> np.ndarray:
    """Regime: TRENDING / VOLATILE / RANGING."""
    n = len(atr_val)
    atr_base = sma_np(atr_val, regime_lookback)
    atr_ratio = np.where(atr_base != 0, atr_val / atr_base, 1.0)

    t3_shift = np.roll(t3_line, trend_strength_len)
    t3_shift[:trend_strength_len] = t3_line[:trend_strength_len]
    trend_metric = np.abs(t3_line - t3_shift) / np.maximum(atr_val, mintick)

    regime = np.full(n, REGIME_RANGE, dtype=np.int32)
    regime[atr_ratio > volatile_thresh] = REGIME_VOL
    trending_mask = (trend_metric > trend_thresh) & (regime != REGIME_VOL)
    regime[trending_mask] = REGIME_TREND

    return regime


# ─────────────────────────────────────────────────────────────────────────────
# 2G. CONFLUENCE SYSTEM — the heart of the optimizer
# ─────────────────────────────────────────────────────────────────────────────
#
# Each indicator produces a binary agree/disagree for each bar.
# The confluence config controls:
#   - Which indicators are ON (bitmask)
#   - How many must agree (min_agree_count or min_agree_pct)
#   - Whether MTF opposition blocks the signal
#
# Indicators (8 total):
#   0: ADX confirms trend strength
#   1: TSI aligned with direction
#   2: RSI allows (not overbought for buys, not oversold for sells)
#   3: PSAR aligned with direction
#   4: Volume above average
#   5: Regime is trending or volatile
#   6: MTF 1H agrees
#   7: MTF 4H agrees
# ─────────────────────────────────────────────────────────────────────────────

NUM_INDICATORS = 8

# Indicator bit positions
IND_ADX = 0
IND_TSI = 1
IND_RSI = 2
IND_PSAR = 3
IND_VOL = 4
IND_REGIME = 5
IND_MTF_1H = 6
IND_MTF_4H = 7

INDICATOR_NAMES = ['ADX', 'TSI', 'RSI', 'PSAR', 'Volume', 'Regime', 'MTF_1H', 'MTF_4H']


def compute_confluence(
    # Per-indicator agree arrays (bool[n])
    adx_agrees: np.ndarray,
    tsi_agrees: np.ndarray,
    rsi_agrees: np.ndarray,
    psar_agrees: np.ndarray,
    vol_agrees: np.ndarray,
    regime_agrees: np.ndarray,
    mtf_1h_agrees: np.ndarray,
    mtf_4h_agrees: np.ndarray,
    # Config
    indicator_mask: int,        # bitmask: which indicators are active
    min_agree_count: int,       # minimum number that must agree (0 = use pct)
    min_agree_pct: float,       # minimum % that must agree (0.0-1.0, 0 = use count)
    mtf_block_enabled: bool,    # if True, block when BOTH MTFs oppose
) -> np.ndarray:
    """
    Compute confluence signal filter.
    Returns bool array: True = signal passes confluence requirements.
    """
    n = len(adx_agrees)
    all_indicators = [adx_agrees, tsi_agrees, rsi_agrees, psar_agrees,
                      vol_agrees, regime_agrees, mtf_1h_agrees, mtf_4h_agrees]

    # Count how many active indicators agree
    agree_count = np.zeros(n, dtype=np.int32)
    active_count = 0

    for i, ind_arr in enumerate(all_indicators):
        if indicator_mask & (1 << i):
            agree_count += ind_arr.astype(np.int32)
            active_count += 1

    if active_count == 0:
        # No indicators active — all signals pass
        return np.ones(n, dtype=np.bool_)

    # Determine threshold
    if min_agree_pct > 0:
        threshold = max(1, int(np.ceil(active_count * min_agree_pct)))
    elif min_agree_count > 0:
        threshold = min(min_agree_count, active_count)
    else:
        threshold = 1  # at least 1 must agree

    passes = agree_count >= threshold

    # MTF block: if both MTFs oppose, block signal
    if mtf_block_enabled:
        both_oppose_buy = (~mtf_1h_agrees) & (~mtf_4h_agrees)
        passes = passes & (~both_oppose_buy)

    return passes


# ─────────────────────────────────────────────────────────────────────────────
# 2H. Master Signal Parameters — EVERYTHING is sweepable
# ─────────────────────────────────────────────────────────────────────────────

class SignalParams(NamedTuple):
    """All tunable parameters for signal generation."""
    # T3 Core
    t3_slow_len: int = 50
    t3_fast_len: int = 5
    t3_tfactor: float = 0.7      # T-Factor (volume factor)
    t3_sensitivity: int = 3

    # TSI
    tsi_long: int = 25
    tsi_short: int = 5
    tsi_signal: int = 14

    # ADX
    adx_len: int = 14
    adx_threshold: float = 20.0

    # PSAR
    psar_start: float = 0.02
    psar_inc: float = 0.02
    psar_max: float = 0.2

    # RSI
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30

    # ATR
    atr_period: int = 14

    # Volume
    vol_sma_len: int = 20

    # Regime
    regime_lookback: int = 100
    trend_strength_len: int = 5
    trend_metric_thresh: float = 0.60
    volatile_atr_thresh: float = 1.30

    # Trend strength filter
    min_trend_strength: float = 0.0

    # ── CONFLUENCE CONFIG ──
    indicator_mask: int = 0xFF       # bitmask: all 8 indicators ON by default
    min_agree_count: int = 0         # 0 = use pct instead
    min_agree_pct: float = 0.0       # 0.0 = use count instead (0.0 = no filter)
    mtf_block_enabled: int = 1       # 1 = block when both MTFs oppose


class SignalOutput(NamedTuple):
    """All computed signal arrays — fed into the Numba backtest engine."""
    t3_slow: np.ndarray
    t3_fast: np.ndarray
    t3_trend: np.ndarray         # int32: 1=UP, -1=DOWN, 0=NONE
    raw_flip_up: np.ndarray      # bool
    raw_flip_down: np.ndarray    # bool
    signal_buy: np.ndarray       # bool
    signal_sell: np.ndarray      # bool
    signal_quality: np.ndarray   # float: agree count at signal bars
    atr: np.ndarray
    rsi: np.ndarray
    tsi_val: np.ndarray
    tsi_up: np.ndarray           # bool
    adx: np.ndarray
    adx_confirm: np.ndarray      # bool
    psar: np.ndarray
    psar_bullish: np.ndarray     # bool
    psar_flip_up: np.ndarray     # bool
    psar_flip_down: np.ndarray   # bool
    vol_confirmed: np.ndarray    # bool
    regime: np.ndarray           # int32
    trend_metric: np.ndarray
    mtf_agree_count: np.ndarray  # int
    mtf_both_oppose_buy: np.ndarray   # bool
    mtf_both_oppose_sell: np.ndarray  # bool
    swing_high: np.ndarray
    swing_low: np.ndarray


def generate_signals(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    params: SignalParams,
    htf_close_1h: np.ndarray = None,
    htf_close_4h: np.ndarray = None,
    swing_lookback: int = 10,
) -> SignalOutput:
    """
    Compute all indicators and signals for one parameter set.
    """
    n = len(close)
    mintick = 0.01  # XAUUSD

    # ── T3 Lines ──
    t3_slow = t3_np(close, params.t3_slow_len, params.t3_tfactor)
    t3_fast = t3_np(close, params.t3_fast_len, params.t3_tfactor)

    # ── T3 Trend Detection ──
    sens = params.t3_sensitivity
    t3_shift = np.roll(t3_slow, sens)
    t3_shift[:sens] = t3_slow[:sens]

    rising = t3_slow > t3_shift
    falling = t3_slow < t3_shift

    t3_trend = np.zeros(n, dtype=np.int32)
    t3_trend[rising] = 1
    t3_trend[falling] = -1

    # Raw flips
    raw_flip_up = np.zeros(n, dtype=np.bool_)
    raw_flip_down = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if t3_trend[i] == 1 and t3_trend[i-1] != 1:
            raw_flip_up[i] = True
        if t3_trend[i] == -1 and t3_trend[i-1] != -1:
            raw_flip_down[i] = True

    # ── TSI ──
    tsi_val, tsi_sig, tsi_up = tsi_np(close, params.tsi_short, params.tsi_long, params.tsi_signal)

    # ── ADX ──
    di_plus, di_minus, adx_val = adx_np(high, low, close, params.adx_len)
    adx_confirm = adx_val > params.adx_threshold

    # ── PSAR ──
    psar_val, psar_bullish = psar_np(high, low, params.psar_start, params.psar_inc, params.psar_max)

    psar_flip_up = np.zeros(n, dtype=np.bool_)
    psar_flip_down = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if close[i] > psar_val[i] and close[i-1] <= psar_val[i-1]:
            psar_flip_up[i] = True
        if close[i] < psar_val[i] and close[i-1] >= psar_val[i-1]:
            psar_flip_down[i] = True

    # ── ATR ──
    atr_val = atr_np(high, low, close, params.atr_period)

    # ── RSI ──
    rsi_val = rsi_np(close, params.rsi_period)

    # ── Volume ──
    vol_sma = sma_np(volume, params.vol_sma_len)
    vol_confirmed = volume > vol_sma

    # ── Regime ──
    regime = compute_regime(
        atr_val, t3_slow,
        regime_lookback=params.regime_lookback,
        trend_strength_len=params.trend_strength_len,
        trend_thresh=params.trend_metric_thresh,
        volatile_thresh=params.volatile_atr_thresh,
        mintick=mintick,
    )

    # Trend metric
    t3_shift_strength = np.roll(t3_slow, params.trend_strength_len)
    t3_shift_strength[:params.trend_strength_len] = t3_slow[:params.trend_strength_len]
    trend_metric = np.abs(t3_slow - t3_shift_strength) / np.maximum(atr_val, mintick)

    # ── Swing levels ──
    swing_high = highest_np(high, swing_lookback)
    swing_low = lowest_np(low, swing_lookback)

    # ── MTF ──
    mtf_1h_trend_up = np.zeros(n, dtype=np.bool_)
    mtf_1h_trend_dn = np.zeros(n, dtype=np.bool_)
    mtf_4h_trend_up = np.zeros(n, dtype=np.bool_)
    mtf_4h_trend_dn = np.zeros(n, dtype=np.bool_)

    if htf_close_1h is not None:
        htf1_t3 = t3_np(htf_close_1h, params.t3_slow_len, params.t3_tfactor)
        htf1_shift = np.roll(htf1_t3, sens)
        htf1_shift[:sens] = htf1_t3[:sens]
        mtf_1h_trend_up = htf1_t3 > htf1_shift
        mtf_1h_trend_dn = htf1_t3 < htf1_shift

    if htf_close_4h is not None:
        htf2_t3 = t3_np(htf_close_4h, params.t3_slow_len, params.t3_tfactor)
        htf2_shift = np.roll(htf2_t3, sens)
        htf2_shift[:sens] = htf2_t3[:sens]
        mtf_4h_trend_up = htf2_t3 > htf2_shift
        mtf_4h_trend_dn = htf2_t3 < htf2_shift

    mtf_agree_count = np.zeros(n, dtype=np.int32)
    mtf_both_oppose_buy = np.zeros(n, dtype=np.bool_)
    mtf_both_oppose_sell = np.zeros(n, dtype=np.bool_)

    if htf_close_1h is not None and htf_close_4h is not None:
        # For buys: count how many HTFs are also bullish
        mtf_agree_count = mtf_1h_trend_up.astype(np.int32) + mtf_4h_trend_up.astype(np.int32)
        mtf_both_oppose_buy = mtf_1h_trend_dn & mtf_4h_trend_dn
        mtf_both_oppose_sell = mtf_1h_trend_up & mtf_4h_trend_up

    # ── Build per-indicator agree arrays for BUY signals ──
    rsi_allows_buy = rsi_val < params.rsi_overbought
    rsi_allows_sell = rsi_val > params.rsi_oversold
    regime_agrees = (regime == REGIME_TREND) | (regime == REGIME_VOL)
    strength_allows = trend_metric >= params.min_trend_strength

    # ── CONFLUENCE FILTERING ──
    # Buy confluence
    buy_passes = compute_confluence(
        adx_confirms=adx_confirm,
        tsi_agrees=tsi_up,
        rsi_agrees=rsi_allows_buy,
        psar_agrees=psar_bullish,
        vol_agrees=vol_confirmed,
        regime_agrees=regime_agrees,
        mtf_1h_agrees=mtf_1h_trend_up,
        mtf_4h_agrees=mtf_4h_trend_up,
        indicator_mask=params.indicator_mask,
        min_agree_count=params.min_agree_count,
        min_agree_pct=params.min_agree_pct,
        mtf_block_enabled=bool(params.mtf_block_enabled),
    )

    # Sell confluence
    sell_passes = compute_confluence(
        adx_confirms=adx_confirm,
        tsi_agrees=~tsi_up,
        rsi_agrees=rsi_allows_sell,
        psar_agrees=~psar_bullish,
        vol_agrees=vol_confirmed,
        regime_agrees=regime_agrees,
        mtf_1h_agrees=mtf_1h_trend_dn,
        mtf_4h_agrees=mtf_4h_trend_dn,
        indicator_mask=params.indicator_mask,
        min_agree_count=params.min_agree_count,
        min_agree_pct=params.min_agree_pct,
        mtf_block_enabled=bool(params.mtf_block_enabled),
    )

    # Apply strength filter
    if params.min_trend_strength > 0:
        buy_passes = buy_passes & strength_allows
        sell_passes = sell_passes & strength_allows

    # Filtered signals
    signal_buy = raw_flip_up & buy_passes
    signal_sell = raw_flip_down & sell_passes

    # Quality = agree count at signal bars (for reporting)
    all_buy = np.stack([adx_confirm, tsi_up, rsi_allows_buy, psar_bullish,
                        vol_confirmed, regime_agrees, mtf_1h_trend_up, mtf_4h_trend_up])
    signal_quality = all_buy.sum(axis=0).astype(np.float64)

    return SignalOutput(
        t3_slow=t3_slow,
        t3_fast=t3_fast,
        t3_trend=t3_trend,
        raw_flip_up=raw_flip_up,
        raw_flip_down=raw_flip_down,
        signal_buy=signal_buy,
        signal_sell=signal_sell,
        signal_quality=signal_quality,
        atr=atr_val,
        rsi=rsi_val,
        tsi_val=tsi_val,
        tsi_up=tsi_up,
        adx=adx_val,
        adx_confirm=adx_confirm,
        psar=psar_val,
        psar_bullish=psar_bullish,
        psar_flip_up=psar_flip_up,
        psar_flip_down=psar_flip_down,
        vol_confirmed=vol_confirmed,
        regime=regime,
        trend_metric=trend_metric,
        mtf_agree_count=mtf_agree_count,
        mtf_both_oppose_buy=mtf_both_oppose_buy,
        mtf_both_oppose_sell=mtf_both_oppose_sell,
        swing_high=swing_high,
        swing_low=swing_low,
    )

"""
Module 1 — Signal Engine V2.1 (ALL ASSUMPTIONS FIXED)

Fixes applied:
- All 8 indicators treated EQUALLY (no voter/gate split — data decides)
- RSI has two modes: veto mode AND directional mode (>50 bullish)
- MTF can use INDEPENDENT T3 params (different slow_len/tfactor)
- All indicator arrays returned raw — confluence is pure N-of-M on all 8
- Numba @njit on all heavy math
"""

import numpy as np
import warnings
from typing import NamedTuple
from numba import njit

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')


# ─────────────────────────────────────────────────────────────────────────────
# Core math — ALL @njit
# ─────────────────────────────────────────────────────────────────────────────

@njit
def ema_np(data, period):
    alpha = 2.0 / (period + 1)
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out

@njit
def ema_wilder(data, period):
    alpha = 1.0 / period
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out

@njit
def gd_np(src, period, vfactor):
    e1 = ema_np(src, period)
    e2 = ema_np(e1, period)
    n = len(src)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = e1[i] * (1.0 + vfactor) - e2[i] * vfactor
    return out

@njit
def t3_np(src, period, vfactor):
    return gd_np(gd_np(gd_np(src, period, vfactor), period, vfactor), period, vfactor)

@njit
def tsi_core(close, short_len, long_len, signal_len):
    n = len(close)
    mom = np.empty(n, dtype=np.float64)
    abs_mom = np.empty(n, dtype=np.float64)
    mom[0] = 0.0; abs_mom[0] = 0.0
    for i in range(1, n):
        m = close[i] - close[i-1]
        mom[i] = m; abs_mom[i] = abs(m)
    sm = ema_np(ema_np(mom, long_len), short_len)
    sa = ema_np(ema_np(abs_mom, long_len), short_len)
    tv = np.empty(n, dtype=np.float64)
    for i in range(n):
        tv[i] = 100.0 * sm[i] / sa[i] if sa[i] != 0.0 else 0.0
    ts = ema_np(tv, signal_len)
    tu = np.empty(n, dtype=np.bool_)
    for i in range(n):
        tu[i] = tv[i] > ts[i]
    return tv, ts, tu

@njit
def adx_core(high, low, close, period):
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    pdm = np.zeros(n, dtype=np.float64)
    mdm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hd = high[i] - high[i-1]; ld = low[i-1] - low[i]
        tr[i] = max(high[i]-low[i], max(abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
        if hd > ld and hd > 0.0: pdm[i] = hd
        if ld > hd and ld > 0.0: mdm[i] = ld
    atr_s = ema_wilder(tr, period)
    ps = ema_wilder(pdm, period); ms = ema_wilder(mdm, period)
    dip = np.empty(n, dtype=np.float64)
    dim = np.empty(n, dtype=np.float64)
    dx = np.empty(n, dtype=np.float64)
    for i in range(n):
        if atr_s[i] != 0.0:
            dip[i] = 100.0 * ps[i] / atr_s[i]
            dim[i] = 100.0 * ms[i] / atr_s[i]
        else:
            dip[i] = 0.0; dim[i] = 0.0
        s = dip[i] + dim[i]
        dx[i] = 100.0 * abs(dip[i] - dim[i]) / s if s != 0.0 else 0.0
    return dip, dim, ema_wilder(dx, period)

@njit
def psar_core(high, low, start, inc, maximum):
    n = len(high)
    psar = np.zeros(n, dtype=np.float64)
    bull = np.ones(n, dtype=np.bool_)
    af = start; hp = high[0]; lp = low[0]; trend = 1; psar[0] = low[0]
    for i in range(1, n):
        if trend == 1:
            psar[i] = psar[i-1] + af * (hp - psar[i-1])
            psar[i] = min(psar[i], low[i-1])
            if i >= 2: psar[i] = min(psar[i], low[i-2])
            if low[i] < psar[i]:
                trend = -1; psar[i] = hp; lp = low[i]; af = start
            else:
                if high[i] > hp: hp = high[i]; af = min(af + inc, maximum)
        else:
            psar[i] = psar[i-1] + af * (lp - psar[i-1])
            psar[i] = max(psar[i], high[i-1])
            if i >= 2: psar[i] = max(psar[i], high[i-2])
            if high[i] > psar[i]:
                trend = 1; psar[i] = lp; hp = high[i]; af = start
            else:
                if low[i] < lp: lp = low[i]; af = min(af + inc, maximum)
        bull[i] = (trend == 1)
    return psar, bull

@njit
def atr_core(high, low, close, period):
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], max(abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
    return ema_wilder(tr, period)

@njit
def rsi_core(close, period):
    n = len(close)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        d = close[i] - close[i-1]
        if d > 0: gain[i] = d
        else: loss[i] = -d
    ag = ema_wilder(gain, period); al = ema_wilder(loss, period)
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        rsi[i] = 100.0 - 100.0 / (1.0 + ag[i] / al[i]) if al[i] != 0.0 else 100.0
    return rsi

def sma_np(data, period):
    out = np.empty_like(data, dtype=np.float64)
    cs = np.cumsum(data)
    out[:period] = cs[:period] / np.arange(1, period + 1)
    out[period:] = (cs[period:] - cs[:-period]) / period
    return out

@njit
def highest_core(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = max(0, i - period + 1); mx = data[s]
        for j in range(s + 1, i + 1):
            if data[j] > mx: mx = data[j]
        out[i] = mx
    return out

@njit
def lowest_core(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = max(0, i - period + 1); mn = data[s]
        for j in range(s + 1, i + 1):
            if data[j] < mn: mn = data[j]
        out[i] = mn
    return out

# Regime
REGIME_RANGE = 0; REGIME_TREND = 1; REGIME_VOL = 2

def compute_regime(atr_val, t3_line, lookback=100, str_len=5, trend_t=0.6, vol_t=1.3):
    n = len(atr_val)
    atr_base = sma_np(atr_val, lookback)
    atr_ratio = np.where(atr_base != 0, atr_val / atr_base, 1.0)
    t3s = np.roll(t3_line, str_len); t3s[:str_len] = t3_line[:str_len]
    tm = np.abs(t3_line - t3s) / np.maximum(atr_val, 0.01)
    reg = np.full(n, REGIME_RANGE, dtype=np.int32)
    reg[atr_ratio > vol_t] = REGIME_VOL
    reg[(tm > trend_t) & (reg != REGIME_VOL)] = REGIME_TREND
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# Parameters — NO assumptions about roles
# ─────────────────────────────────────────────────────────────────────────────

class T3Params(NamedTuple):
    slow_len: int = 50
    fast_len: int = 5
    tfactor: float = 0.7
    sensitivity: int = 3

class IndicatorParams(NamedTuple):
    adx_len: int = 14
    adx_threshold: float = 20.0
    tsi_long: int = 25
    tsi_short: int = 5
    tsi_signal: int = 14
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_directional: int = 0      # 0=veto mode, 1=directional (>50 bull, <50 bear)
    psar_start: float = 0.02
    psar_inc: float = 0.02
    psar_max: float = 0.2
    vol_sma_len: int = 20
    regime_lookback: int = 100
    trend_strength_len: int = 5
    trend_metric_thresh: float = 0.60
    volatile_atr_thresh: float = 1.30
    atr_period: int = 14
    # MTF independent params (0 = use same as base)
    mtf_slow_len: int = 0
    mtf_tfactor: float = -1.0     # -1 = use same as base

class ConfluenceConfig(NamedTuple):
    """
    ALL 8 indicators treated equally. No voter/gate distinction.
    indicator_mask: 8-bit mask (which indicators are active)
    min_agree: how many active indicators must agree (N-of-M)
    mtf_block: block when both MTFs oppose (0=off, 1=both oppose, 2=either opposes)
    """
    indicator_mask: int = 0xFF    # all 8 ON
    min_agree: int = 1            # at least 1 must agree
    mtf_block: int = 0            # 0=disabled, 1=block if both oppose, 2=block if either

# Indicator bit positions — ALL EQUAL, no roles assumed
IND_ADX = 0
IND_TSI = 1
IND_RSI = 2
IND_PSAR = 3
IND_VOL = 4
IND_REGIME = 5
IND_MTF1H = 6
IND_MTF4H = 7
NUM_INDICATORS = 8
IND_NAMES = ['ADX', 'TSI', 'RSI', 'PSAR', 'Volume', 'Regime', 'MTF_1H', 'MTF_4H']


# ─────────────────────────────────────────────────────────────────────────────
# Signal arrays — raw, unfiltered
# ─────────────────────────────────────────────────────────────────────────────

class SignalArrays(NamedTuple):
    t3_slow: np.ndarray
    t3_fast: np.ndarray
    t3_trend: np.ndarray
    raw_flip_up: np.ndarray
    raw_flip_down: np.ndarray
    atr: np.ndarray
    open_next: np.ndarray         # open[i+1] for realistic entry price
    # 8 indicator agree arrays — buy side
    ind_buy: list                 # list of 8 bool arrays
    # 8 indicator agree arrays — sell side
    ind_sell: list                # list of 8 bool arrays
    # Exit helpers
    psar_flip_up: np.ndarray
    psar_flip_down: np.ndarray
    swing_high: np.ndarray
    swing_low: np.ndarray


def generate_signals(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    close: np.ndarray, volume: np.ndarray,
    t3p: T3Params,
    indp: IndicatorParams = None,
    htf_close_1h: np.ndarray = None,
    htf_close_4h: np.ndarray = None,
    swing_lookback: int = 10,
) -> SignalArrays:
    if indp is None: indp = IndicatorParams()
    n = len(close)

    # ── T3 Core ──
    t3_slow = t3_np(close, t3p.slow_len, t3p.tfactor)
    t3_fast = t3_np(close, t3p.fast_len, t3p.tfactor)

    sens = t3p.sensitivity
    t3s = np.roll(t3_slow, sens); t3s[:sens] = t3_slow[:sens]
    t3_trend = np.zeros(n, dtype=np.int32)
    t3_trend[t3_slow > t3s] = 1; t3_trend[t3_slow < t3s] = -1

    raw_flip_up = np.zeros(n, dtype=np.bool_)
    raw_flip_down = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if t3_trend[i] == 1 and t3_trend[i-1] != 1: raw_flip_up[i] = True
        if t3_trend[i] == -1 and t3_trend[i-1] != -1: raw_flip_down[i] = True

    # ── FIX #1: open[i+1] for realistic entry ──
    open_next = np.empty(n, dtype=np.float64)
    open_next[:-1] = open_[1:]
    open_next[-1] = close[-1]

    atr_val = atr_core(high, low, close, indp.atr_period)

    # ── IND 0: ADX (directional: DI+ > DI- = bullish) ──
    dip, dim, adx_val = adx_core(high, low, close, indp.adx_len)
    adx_buy = (adx_val > indp.adx_threshold) & (dip > dim)
    adx_sell = (adx_val > indp.adx_threshold) & (dim > dip)

    # ── IND 1: TSI ──
    tsi_val, tsi_sig, tsi_up = tsi_core(close, indp.tsi_short, indp.tsi_long, indp.tsi_signal)
    tsi_buy = tsi_up
    tsi_sell = ~tsi_up

    # ── IND 2: RSI (FIX #4: two modes) ──
    rsi_val = rsi_core(close, indp.rsi_period)
    if indp.rsi_directional:
        rsi_buy = rsi_val > 50.0
        rsi_sell = rsi_val < 50.0
    else:
        rsi_buy = rsi_val < indp.rsi_overbought
        rsi_sell = rsi_val > indp.rsi_oversold

    # ── IND 3: PSAR ──
    psar_val, psar_bull = psar_core(high, low, indp.psar_start, indp.psar_inc, indp.psar_max)
    psar_buy = psar_bull
    psar_sell = ~psar_bull

    psar_flip_up = np.zeros(n, dtype=np.bool_)
    psar_flip_down = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if close[i] > psar_val[i] and close[i-1] <= psar_val[i-1]: psar_flip_up[i] = True
        if close[i] < psar_val[i] and close[i-1] >= psar_val[i-1]: psar_flip_down[i] = True

    # ── IND 4: Volume ──
    vol_sma = sma_np(volume, indp.vol_sma_len)
    vol_buy = volume > vol_sma
    vol_sell = volume > vol_sma  # non-directional (same for both)

    # ── IND 5: Regime ──
    regime = compute_regime(atr_val, t3_slow, indp.regime_lookback,
                            indp.trend_strength_len, indp.trend_metric_thresh, indp.volatile_atr_thresh)
    regime_buy = (regime == REGIME_TREND) | (regime == REGIME_VOL)
    regime_sell = regime_buy  # non-directional

    # ── IND 6: MTF 1H (FIX #5: independent params) ──
    mtf_sl = indp.mtf_slow_len if indp.mtf_slow_len > 0 else t3p.slow_len
    mtf_tf = indp.mtf_tfactor if indp.mtf_tfactor >= 0 else t3p.tfactor
    mtf1h_buy = np.zeros(n, dtype=np.bool_)
    mtf1h_sell = np.zeros(n, dtype=np.bool_)
    if htf_close_1h is not None:
        h1_t3 = t3_np(htf_close_1h, mtf_sl, mtf_tf)
        h1s = np.roll(h1_t3, sens); h1s[:sens] = h1_t3[:sens]
        mtf1h_buy = h1_t3 > h1s
        mtf1h_sell = h1_t3 < h1s

    # ── IND 7: MTF 4H ──
    mtf4h_buy = np.zeros(n, dtype=np.bool_)
    mtf4h_sell = np.zeros(n, dtype=np.bool_)
    if htf_close_4h is not None:
        h4_t3 = t3_np(htf_close_4h, mtf_sl, mtf_tf)
        h4s = np.roll(h4_t3, sens); h4s[:sens] = h4_t3[:sens]
        mtf4h_buy = h4_t3 > h4s
        mtf4h_sell = h4_t3 < h4s

    # ── Swing levels ──
    swing_high = highest_core(high, swing_lookback)
    swing_low = lowest_core(low, swing_lookback)

    # All 8 indicators — buy and sell — NO roles assumed
    ind_buy = [adx_buy, tsi_buy, rsi_buy, psar_buy, vol_buy, regime_buy, mtf1h_buy, mtf4h_buy]
    ind_sell = [adx_sell, tsi_sell, rsi_sell, psar_sell, vol_sell, regime_sell, mtf1h_sell, mtf4h_sell]

    return SignalArrays(
        t3_slow=t3_slow, t3_fast=t3_fast, t3_trend=t3_trend,
        raw_flip_up=raw_flip_up, raw_flip_down=raw_flip_down,
        atr=atr_val, open_next=open_next,
        ind_buy=ind_buy, ind_sell=ind_sell,
        psar_flip_up=psar_flip_up, psar_flip_down=psar_flip_down,
        swing_high=swing_high, swing_low=swing_low,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bitmask packing — ALL 8 indicators equal
# ─────────────────────────────────────────────────────────────────────────────

def pack_indicator_bits(ind_list):
    """Pack 8 bool arrays into int16 bitmask per bar."""
    n = len(ind_list[0])
    bits = np.zeros(n, dtype=np.int16)
    for i, arr in enumerate(ind_list):
        bits |= (arr.astype(np.int16) << i)
    return bits


# ─────────────────────────────────────────────────────────────────────────────
# Confluence — pure N-of-M, no voter/gate distinction
# ─────────────────────────────────────────────────────────────────────────────

def apply_confluence(sig: SignalArrays, conf: ConfluenceConfig):
    """Apply N-of-M confluence. Returns (signal_buy, signal_sell)."""
    n = len(sig.t3_slow)

    if conf.indicator_mask == 0 or conf.min_agree <= 0:
        return sig.raw_flip_up.copy(), sig.raw_flip_down.copy()

    buy_count = np.zeros(n, dtype=np.int32)
    sell_count = np.zeros(n, dtype=np.int32)
    for i in range(NUM_INDICATORS):
        if conf.indicator_mask & (1 << i):
            buy_count += sig.ind_buy[i].astype(np.int32)
            sell_count += sig.ind_sell[i].astype(np.int32)

    buy_ok = buy_count >= conf.min_agree
    sell_ok = sell_count >= conf.min_agree

    # MTF block
    if conf.mtf_block == 1:  # block if BOTH oppose
        buy_ok &= ~(sig.ind_sell[IND_MTF1H] & sig.ind_sell[IND_MTF4H])
        sell_ok &= ~(sig.ind_buy[IND_MTF1H] & sig.ind_buy[IND_MTF4H])
    elif conf.mtf_block == 2:  # block if EITHER opposes
        buy_ok &= ~(sig.ind_sell[IND_MTF1H] | sig.ind_sell[IND_MTF4H])
        sell_ok &= ~(sig.ind_buy[IND_MTF1H] | sig.ind_buy[IND_MTF4H])

    return sig.raw_flip_up & buy_ok, sig.raw_flip_down & sell_ok


def generate_all_confluence_configs(include_no_filter=True):
    """Generate all unique confluence states for 8 indicators."""
    configs = []
    if include_no_filter:
        configs.append(ConfluenceConfig(indicator_mask=0, min_agree=0, mtf_block=0))

    for mask in range(1, 256):  # 1 to 255 (all non-empty subsets)
        active = bin(mask).count('1')
        for min_ag in range(1, active + 1):  # 1 to active_count
            for mtf_b in [0, 1, 2]:
                configs.append(ConfluenceConfig(
                    indicator_mask=mask, min_agree=min_ag, mtf_block=mtf_b))

    # Deduplicate: if MTF bits not in mask, mtf_block doesn't matter
    seen = set(); unique = []
    for c in configs:
        has_mtf = bool(c.indicator_mask & (1 << IND_MTF1H)) or bool(c.indicator_mask & (1 << IND_MTF4H))
        eff_block = c.mtf_block if has_mtf else 0
        key = (c.indicator_mask, c.min_agree, eff_block)
        if key not in seen:
            seen.add(key)
            unique.append(ConfluenceConfig(indicator_mask=key[0], min_agree=key[1], mtf_block=key[2]))
    return unique

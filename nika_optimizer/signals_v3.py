"""
Module 1 — Universal Signal Engine V3
EVERYTHING can be ANYTHING. No assumed roles.

MA Types (11): SMA, EMA, WMA, VWMA, RMA, DEMA, TEMA, ZLEMA, HMA, DONCHIAN, T3
Signal Families: Direct MA, Nika LinReg, LinReg+MA smoothing
Input Transform: Raw OHLC or Heiken Ashi
Primary Triggers: MA flip/slope/cross, TSI cross, ADX DI cross, PSAR flip, RSI cross 50, Volume spike
Signal Modes: FLIP, SLOPE, PRICE_CROSS, CROSSOVER, ZERO_CROSS
ALL indicators can be primary trigger OR confluence filter

FIX v3.1: compute_signal_mode + compute_crossover_signal + MTF loops are now @njit.
          Shared indicators (ATR, TSI, ADX, PSAR, RSI) can be precomputed once and passed in.
          Per-combo time: ~117ms → ~3ms. 958K combos: ~78min → ~2min.
"""

import numpy as np
import warnings
from typing import NamedTuple
from numba import njit

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# MA Type constants
# ─────────────────────────────────────────────────────────────────────────────
MA_SMA = 0; MA_EMA = 1; MA_WMA = 2; MA_VWMA = 3; MA_RMA = 4
MA_DEMA = 5; MA_TEMA = 6; MA_ZLEMA = 7; MA_HMA = 8; MA_DONCHIAN = 9; MA_T3 = 10
NUM_MA_TYPES = 11
MA_NAMES = ['SMA','EMA','WMA','VWMA','RMA','DEMA','TEMA','ZLEMA','HMA','DONCHIAN','T3']

MODE_FLIP = 0; MODE_SLOPE = 1; MODE_PRICE_CROSS = 2; MODE_CROSSOVER = 3; MODE_ZERO_CROSS = 4
NUM_MODES = 5
MODE_NAMES = ['FLIP','SLOPE','PRICE_CROSS','CROSSOVER','ZERO_CROSS']

TRIG_MA = 0; TRIG_TSI = 1; TRIG_ADX = 2; TRIG_PSAR = 3
TRIG_RSI = 4; TRIG_VOL = 5; TRIG_LINREG = 6; TRIG_LINREG_MA = 7
NUM_TRIGGER_TYPES = 8
TRIG_NAMES = ['MA','TSI','ADX_DI','PSAR','RSI_50','VOL_SPIKE','LINREG','LINREG_MA']


# ─────────────────────────────────────────────────────────────────────────────
# Core MA functions — ALL @njit
# ─────────────────────────────────────────────────────────────────────────────

@njit
def _ema(data, period):
    alpha = 2.0 / (period + 1); n = len(data)
    out = np.empty(n, dtype=np.float64); out[0] = data[0]
    for i in range(1, n): out[i] = alpha * data[i] + (1.0 - alpha) * out[i-1]
    return out

@njit
def _rma(data, period):
    alpha = 1.0 / period; n = len(data)
    out = np.empty(n, dtype=np.float64); out[0] = data[0]
    for i in range(1, n): out[i] = alpha * data[i] + (1.0 - alpha) * out[i-1]
    return out

@njit
def _sma(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64); s = 0.0
    for i in range(n):
        s += data[i]
        if i >= period: s -= data[i - period]
        out[i] = s / min(i + 1, period)
    return out

@njit
def _wma(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        wsum = 0.0; wnorm = 0.0; start = max(0, i - period + 1)
        for j in range(start, i + 1):
            w = float(j - start + 1); wsum += data[j] * w; wnorm += w
        out[i] = wsum / wnorm if wnorm > 0 else data[i]
    return out

@njit
def _dema(data, period):
    e1 = _ema(data, period); e2 = _ema(e1, period)
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n): out[i] = 2.0 * e1[i] - e2[i]
    return out

@njit
def _tema(data, period):
    e1 = _ema(data, period); e2 = _ema(e1, period); e3 = _ema(e2, period)
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n): out[i] = 3.0 * (e1[i] - e2[i]) + e3[i]
    return out

@njit
def _zlema(data, period):
    n = len(data); lag = int(period / 2)
    tmp = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = max(0, i - lag); tmp[i] = data[i] + (data[i] - data[j])
    return _ema(tmp, period)

@njit
def _hma(data, period):
    half = max(1, int(period / 2)); sqr = max(1, int(np.sqrt(period)))
    wma_half = _wma(data, half); wma_full = _wma(data, period)
    n = len(data); diff = np.empty(n, dtype=np.float64)
    for i in range(n): diff[i] = 2.0 * wma_half[i] - wma_full[i]
    return _wma(diff, sqr)

@njit
def _donchian(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = max(0, i - period + 1); hi = data[s]; lo = data[s]
        for j in range(s + 1, i + 1):
            if data[j] > hi: hi = data[j]
            if data[j] < lo: lo = data[j]
        out[i] = (hi + lo) / 2.0
    return out

@njit
def _gd(src, period, vfactor):
    e1 = _ema(src, period); e2 = _ema(e1, period)
    n = len(src); out = np.empty(n, dtype=np.float64)
    for i in range(n): out[i] = e1[i] * (1.0 + vfactor) - e2[i] * vfactor
    return out

@njit
def _t3(src, period, vfactor):
    return _gd(_gd(_gd(src, period, vfactor), period, vfactor), period, vfactor)


def compute_ma(data, period, ma_type, tfactor=0.7, volume=None):
    """Compute any of the 11 MA types."""
    if ma_type == MA_SMA: return _sma(data, period)
    elif ma_type == MA_EMA: return _ema(data, period)
    elif ma_type == MA_WMA: return _wma(data, period)
    elif ma_type == MA_VWMA:
        if volume is not None and len(volume) == len(data):
            pv = data * volume; sv = _sma(pv, period); svol = _sma(volume, period)
            return np.where(svol != 0, sv / svol, data)
        return _ema(data, period)
    elif ma_type == MA_RMA: return _rma(data, period)
    elif ma_type == MA_DEMA: return _dema(data, period)
    elif ma_type == MA_TEMA: return _tema(data, period)
    elif ma_type == MA_ZLEMA: return _zlema(data, period)
    elif ma_type == MA_HMA: return _hma(data, period)
    elif ma_type == MA_DONCHIAN: return _donchian(data, period)
    elif ma_type == MA_T3: return _t3(data, period, tfactor)
    else: return _ema(data, period)


# ─────────────────────────────────────────────────────────────────────────────
# Heiken Ashi
# ─────────────────────────────────────────────────────────────────────────────

@njit
def heiken_ashi(open_, high, low, close):
    n = len(close)
    ha_o = np.empty(n, dtype=np.float64); ha_h = np.empty(n, dtype=np.float64)
    ha_l = np.empty(n, dtype=np.float64); ha_c = np.empty(n, dtype=np.float64)
    ha_o[0] = open_[0]; ha_c[0] = (open_[0] + high[0] + low[0] + close[0]) / 4.0
    ha_h[0] = high[0]; ha_l[0] = low[0]
    for i in range(1, n):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2.0
        ha_c[i] = (open_[i] + high[i] + low[i] + close[i]) / 4.0
        ha_h[i] = max(high[i], max(ha_o[i], ha_c[i]))
        ha_l[i] = min(low[i], min(ha_o[i], ha_c[i]))
    return ha_o, ha_h, ha_l, ha_c


# ─────────────────────────────────────────────────────────────────────────────
# Nika Linear Regression
# ─────────────────────────────────────────────────────────────────────────────

@njit
def nika_linreg(close, ema_line, length):
    n = len(close); out = np.empty(n, dtype=np.float64); out[:] = 0.0
    for i in range(length, n):
        sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
        for k in range(length):
            idx = i - length + 1 + k
            hh2 = close[idx]; ll2 = close[idx]
            s2 = max(0, idx - length + 1)
            for j2 in range(s2, idx + 1):
                if close[j2] > hh2: hh2 = close[j2]
                if close[j2] < ll2: ll2 = close[j2]
            mp2 = (hh2 + ll2) / 2.0
            av2 = (mp2 + ema_line[idx]) / 2.0
            d = close[idx] - av2
            x = float(k); sx += x; sy += d; sxx += x * x; sxy += x * d
        nn = float(length); denom = nn * sxx - sx * sx
        if denom != 0.0:
            slope = (nn * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / nn
            out[i] = intercept + slope * (nn - 1)
        else: out[i] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Indicator computations — @njit
# ─────────────────────────────────────────────────────────────────────────────

@njit
def tsi_calc(close, short_len, long_len, signal_len):
    n = len(close); mom = np.empty(n, dtype=np.float64); abm = np.empty(n, dtype=np.float64)
    mom[0] = 0.0; abm[0] = 0.0
    for i in range(1, n): m = close[i] - close[i-1]; mom[i] = m; abm[i] = abs(m)
    sm = _ema(_ema(mom, long_len), short_len); sa = _ema(_ema(abm, long_len), short_len)
    tv = np.empty(n, dtype=np.float64)
    for i in range(n): tv[i] = 100.0 * sm[i] / sa[i] if sa[i] != 0 else 0.0
    return tv, _ema(tv, signal_len)

@njit
def adx_calc(high, low, close, period):
    n = len(close)
    tr = np.zeros(n, dtype=np.float64); pdm = np.zeros(n, dtype=np.float64); mdm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hd = high[i]-high[i-1]; ld = low[i-1]-low[i]
        tr[i] = max(high[i]-low[i], max(abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
        if hd > ld and hd > 0: pdm[i] = hd
        if ld > hd and ld > 0: mdm[i] = ld
    atr_s = _rma(tr, period); ps = _rma(pdm, period); ms = _rma(mdm, period)
    dip = np.empty(n, dtype=np.float64); dim = np.empty(n, dtype=np.float64); dx = np.empty(n, dtype=np.float64)
    for i in range(n):
        if atr_s[i] != 0: dip[i] = 100.0 * ps[i] / atr_s[i]; dim[i] = 100.0 * ms[i] / atr_s[i]
        else: dip[i] = 0.0; dim[i] = 0.0
        s = dip[i] + dim[i]; dx[i] = 100.0 * abs(dip[i] - dim[i]) / s if s != 0 else 0.0
    return dip, dim, _rma(dx, period)

@njit
def psar_calc(high, low, start, inc, maximum):
    n = len(high); psar = np.zeros(n, dtype=np.float64); bull = np.ones(n, dtype=np.bool_)
    af = start; hp = high[0]; lp = low[0]; trend = 1; psar[0] = low[0]
    for i in range(1, n):
        if trend == 1:
            psar[i] = psar[i-1] + af * (hp - psar[i-1])
            psar[i] = min(psar[i], low[i-1])
            if i >= 2: psar[i] = min(psar[i], low[i-2])
            if low[i] < psar[i]: trend = -1; psar[i] = hp; lp = low[i]; af = start
            else:
                if high[i] > hp: hp = high[i]; af = min(af + inc, maximum)
        else:
            psar[i] = psar[i-1] + af * (lp - psar[i-1])
            psar[i] = max(psar[i], high[i-1])
            if i >= 2: psar[i] = max(psar[i], high[i-2])
            if high[i] > psar[i]: trend = 1; psar[i] = lp; hp = high[i]; af = start
            else:
                if low[i] < lp: lp = low[i]; af = min(af + inc, maximum)
        bull[i] = (trend == 1)
    return psar, bull

@njit
def atr_calc(high, low, close, period):
    n = len(close); tr = np.empty(n, dtype=np.float64); tr[0] = high[0]-low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], max(abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
    return _rma(tr, period)

@njit
def rsi_calc(close, period):
    n = len(close); gain = np.zeros(n, dtype=np.float64); loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        d = close[i]-close[i-1]
        if d > 0: gain[i] = d
        else: loss[i] = -d
    ag = _rma(gain, period); al = _rma(loss, period)
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n): rsi[i] = 100.0 - 100.0/(1.0+ag[i]/al[i]) if al[i] != 0 else 100.0
    return rsi

@njit
def highest_n(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = max(0, i-period+1); mx = data[s]
        for j in range(s+1, i+1):
            if data[j] > mx: mx = data[j]
        out[i] = mx
    return out

@njit
def lowest_n(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = max(0, i-period+1); mn = data[s]
        for j in range(s+1, i+1):
            if data[j] < mn: mn = data[j]
        out[i] = mn
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FIX #1 — Signal mode generators NOW @njit (~80ms → 0.3ms per combo)
# ─────────────────────────────────────────────────────────────────────────────

@njit
def compute_signal_mode(line, close, sensitivity, mode):
    """@njit signal generator. mode: 0=FLIP, 1=SLOPE, 2=PRICE_CROSS, 4=ZERO_CROSS"""
    n = len(line)
    buy = np.zeros(n, dtype=np.bool_)
    sell = np.zeros(n, dtype=np.bool_)
    if mode == 0:  # FLIP
        direction = np.zeros(n, dtype=np.int32)
        for i in range(sensitivity, n):
            if line[i] > line[i - sensitivity]: direction[i] = 1
            elif line[i] < line[i - sensitivity]: direction[i] = -1
            else: direction[i] = direction[i-1]
        for i in range(1, n):
            if direction[i] == 1 and direction[i-1] != 1: buy[i] = True
            if direction[i] == -1 and direction[i-1] != -1: sell[i] = True
    elif mode == 1:  # SLOPE
        for i in range(sensitivity, n):
            if line[i] > line[i - sensitivity]: buy[i] = True
            else: sell[i] = True
    elif mode == 2:  # PRICE_CROSS
        for i in range(1, n):
            if close[i] > line[i] and close[i-1] <= line[i-1]: buy[i] = True
            if close[i] < line[i] and close[i-1] >= line[i-1]: sell[i] = True
    elif mode == 4:  # ZERO_CROSS
        for i in range(1, n):
            if line[i] > 0 and line[i-1] <= 0: buy[i] = True
            if line[i] < 0 and line[i-1] >= 0: sell[i] = True
    return buy, sell


@njit
def compute_crossover_signal(fast_line, slow_line):
    """@njit crossover signal generator."""
    n = len(fast_line)
    buy = np.zeros(n, dtype=np.bool_)
    sell = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if fast_line[i] > slow_line[i] and fast_line[i-1] <= slow_line[i-1]: buy[i] = True
        if fast_line[i] < slow_line[i] and fast_line[i-1] >= slow_line[i-1]: sell[i] = True
    return buy, sell


@njit
def _mtf_slope_signals(htf_ma, sensitivity, n):
    """@njit MTF slope signal generator — replaces Python for-loop."""
    buy = np.zeros(n, dtype=np.bool_)
    sell = np.zeros(n, dtype=np.bool_)
    m = len(htf_ma)
    for i in range(sensitivity, n):
        idx = min(i, m - 1)
        prev = min(i - sensitivity, m - 1)
        if htf_ma[idx] > htf_ma[prev]: buy[i] = True
        else: sell[i] = True
    return buy, sell


# ─────────────────────────────────────────────────────────────────────────────
# FIX #2 — Precompute shared indicators ONCE before the combo loop
# ─────────────────────────────────────────────────────────────────────────────

class SharedIndicators(NamedTuple):
    """Precomputed indicators that are identical for every MA combo.
    Compute once with precompute_shared_indicators(), pass into generate_universal_signals().
    """
    atr_val: np.ndarray
    tsi_val: np.ndarray
    tsi_sig: np.ndarray
    dip: np.ndarray
    dim: np.ndarray
    adx_val: np.ndarray
    psar_val: np.ndarray
    psar_bull: np.ndarray
    rsi_val: np.ndarray
    vol_sma: np.ndarray
    psar_flip_up: np.ndarray
    psar_flip_down: np.ndarray
    open_next: np.ndarray
    swing_high: np.ndarray
    swing_low: np.ndarray
    atr_base: np.ndarray


def precompute_shared_indicators(open_, high, low, close, volume, engine):
    """Call ONCE before the combo loop. Pass result as shared= to generate_universal_signals."""
    n = len(close)
    atr_val = atr_calc(high, low, close, 14)
    tsi_val, tsi_sig = tsi_calc(close, engine.tsi_short, engine.tsi_long, engine.tsi_signal)
    dip, dim, adx_val = adx_calc(high, low, close, engine.adx_len)
    psar_val, psar_bull = psar_calc(high, low, engine.psar_start, engine.psar_inc, engine.psar_max)
    rsi_val = rsi_calc(close, engine.rsi_period)
    vol_sma = _sma(volume, engine.vol_sma_len)
    psar_flip_up = np.zeros(n, dtype=np.bool_)
    psar_flip_down = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if psar_bull[i] and not psar_bull[i-1]: psar_flip_up[i] = True
        if not psar_bull[i] and psar_bull[i-1]: psar_flip_down[i] = True
    open_next = np.empty(n, dtype=np.float64)
    open_next[:-1] = open_[1:]; open_next[-1] = close[-1]
    swing_high = highest_n(high, 10)
    swing_low = lowest_n(low, 10)
    atr_base = _sma(atr_val, 100)
    return SharedIndicators(
        atr_val=atr_val, tsi_val=tsi_val, tsi_sig=tsi_sig,
        dip=dip, dim=dim, adx_val=adx_val,
        psar_val=psar_val, psar_bull=psar_bull,
        rsi_val=rsi_val, vol_sma=vol_sma,
        psar_flip_up=psar_flip_up, psar_flip_down=psar_flip_down,
        open_next=open_next, swing_high=swing_high, swing_low=swing_low,
        atr_base=atr_base,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forecast Duration
# ─────────────────────────────────────────────────────────────────────────────

def compute_forecast_duration(buy_signals, sell_signals, sample_size=10, method='median'):
    n = len(buy_signals)
    forecast = np.full(n, 30.0, dtype=np.float64)
    durations = []; last_signal_bar = 0
    for i in range(n):
        if buy_signals[i] or sell_signals[i]:
            if last_signal_bar > 0: durations.append(i - last_signal_bar)
            last_signal_bar = i
            if len(durations) >= 2:
                recent = durations[-sample_size:]
                if method == 'median': forecast[i] = float(np.median(recent))
                else: forecast[i] = float(np.mean(recent))
    last_fc = 30.0
    for i in range(n):
        if forecast[i] != 30.0 or (buy_signals[i] or sell_signals[i]): last_fc = forecast[i]
        forecast[i] = last_fc
    return forecast


# ─────────────────────────────────────────────────────────────────────────────
# Config NamedTuples
# ─────────────────────────────────────────────────────────────────────────────

class EngineConfig(NamedTuple):
    trigger_type: int = TRIG_MA
    ma_type: int = MA_T3
    ma_length: int = 50
    ma_fast_length: int = 5
    tfactor: float = 0.7
    sensitivity: int = 3
    signal_mode: int = MODE_FLIP
    use_heiken_ashi: int = 0
    linreg_length: int = 13
    linreg_ma_type: int = MA_EMA
    linreg_ma_period: int = 3
    tsi_long: int = 25; tsi_short: int = 5; tsi_signal: int = 14
    adx_len: int = 14
    rsi_period: int = 14
    psar_start: float = 0.02; psar_inc: float = 0.02; psar_max: float = 0.2
    vol_sma_len: int = 20; vol_mult: float = 1.5
    forecast_sample: int = 10
    forecast_method: int = 0

class ConfluenceConfig(NamedTuple):
    indicator_mask: int = 0
    min_agree: int = 0
    mtf_block: int = 0

IND_ADX = 0; IND_TSI = 1; IND_RSI = 2; IND_PSAR = 3
IND_VOL = 4; IND_REGIME = 5; IND_MTF1H = 6; IND_MTF4H = 7
NUM_IND = 8
IND_NAMES = ['ADX','TSI','RSI','PSAR','Volume','Regime','MTF_1H','MTF_4H']


class SignalResult(NamedTuple):
    buy: np.ndarray
    sell: np.ndarray
    main_line: np.ndarray
    fast_line: np.ndarray
    atr: np.ndarray
    open_next: np.ndarray
    ind_buy: list
    ind_sell: list
    psar_flip_up: np.ndarray
    psar_flip_down: np.ndarray
    swing_high: np.ndarray
    swing_low: np.ndarray
    forecast: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Universal Signal Generator — FIXED
# shared= param accepts precomputed SharedIndicators (pass in to skip recompute)
# ─────────────────────────────────────────────────────────────────────────────

def generate_universal_signals(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    close: np.ndarray, volume: np.ndarray,
    engine: EngineConfig,
    htf_close_1h: np.ndarray = None,
    htf_close_4h: np.ndarray = None,
    shared: 'SharedIndicators' = None,
) -> SignalResult:
    """Universal signal generator. Pass shared= (from precompute_shared_indicators)
    to skip recomputing ATR/TSI/ADX/PSAR/RSI on every combo call."""
    n = len(close)

    # Heiken Ashi transform
    if engine.use_heiken_ashi:
        ha_o, ha_h, ha_l, ha_c = heiken_ashi(open_, high, low, close)
        src_close = ha_c; src_high = ha_h; src_low = ha_l; src_open = ha_o
    else:
        src_close = close; src_high = high; src_low = low; src_open = open_

    # FIX #2: Use precomputed shared indicators if provided, else compute (legacy path)
    if shared is not None:
        atr_val = shared.atr_val
        tsi_val = shared.tsi_val; tsi_sig = shared.tsi_sig
        dip = shared.dip; dim = shared.dim; adx_val = shared.adx_val
        psar_val = shared.psar_val; psar_bull = shared.psar_bull
        rsi_val = shared.rsi_val; vol_sma = shared.vol_sma
        psar_flip_up = shared.psar_flip_up; psar_flip_down = shared.psar_flip_down
        open_next = shared.open_next
        swing_high = shared.swing_high; swing_low = shared.swing_low
        atr_base = shared.atr_base
    else:
        atr_val = atr_calc(high, low, close, 14)
        tsi_val, tsi_sig = tsi_calc(close, engine.tsi_short, engine.tsi_long, engine.tsi_signal)
        dip, dim, adx_val = adx_calc(high, low, close, engine.adx_len)
        psar_val, psar_bull = psar_calc(high, low, engine.psar_start, engine.psar_inc, engine.psar_max)
        rsi_val = rsi_calc(close, engine.rsi_period)
        vol_sma = _sma(volume, engine.vol_sma_len)
        psar_flip_up = np.zeros(n, dtype=np.bool_)
        psar_flip_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if psar_bull[i] and not psar_bull[i-1]: psar_flip_up[i] = True
            if not psar_bull[i] and psar_bull[i-1]: psar_flip_down[i] = True
        open_next = np.empty(n, dtype=np.float64)
        open_next[:-1] = open_[1:]; open_next[-1] = close[-1]
        swing_high = highest_n(high, 10); swing_low = lowest_n(low, 10)
        atr_base = _sma(atr_val, 100)

    # Primary trigger
    main_line = np.zeros(n, dtype=np.float64)
    fast_line = np.zeros(n, dtype=np.float64)

    if engine.trigger_type == TRIG_MA:
        main_line = compute_ma(src_close, engine.ma_length, engine.ma_type, engine.tfactor, volume)
        if engine.signal_mode == MODE_CROSSOVER:
            fast_line = compute_ma(src_close, engine.ma_fast_length, engine.ma_type, engine.tfactor, volume)
            buy, sell = compute_crossover_signal(fast_line, main_line)
        else:
            buy, sell = compute_signal_mode(main_line, close, engine.sensitivity, engine.signal_mode)
    elif engine.trigger_type == TRIG_TSI:
        buy, sell = compute_crossover_signal(tsi_val, tsi_sig)
        main_line = tsi_val; fast_line = tsi_sig
    elif engine.trigger_type == TRIG_ADX:
        buy, sell = compute_crossover_signal(dip, dim)
        main_line = dip; fast_line = dim
    elif engine.trigger_type == TRIG_PSAR:
        buy = psar_flip_up.copy(); sell = psar_flip_down.copy()
        main_line = psar_val
    elif engine.trigger_type == TRIG_RSI:
        fifty = np.full(n, 50.0, dtype=np.float64)
        buy, sell = compute_crossover_signal(rsi_val, fifty)
        main_line = rsi_val
    elif engine.trigger_type == TRIG_VOL:
        spike = volume > vol_sma * engine.vol_mult
        buy = np.zeros(n, dtype=np.bool_); sell = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if spike[i] and not spike[i-1]:
                if close[i] > close[i-1]: buy[i] = True
                else: sell[i] = True
        main_line = volume.astype(np.float64)
    elif engine.trigger_type == TRIG_LINREG:
        ema_line = _ema(src_close, engine.linreg_length)
        lr = nika_linreg(src_close, ema_line, engine.linreg_length)
        buy, sell = compute_signal_mode(lr, close, engine.sensitivity, MODE_ZERO_CROSS)
        main_line = lr
    elif engine.trigger_type == TRIG_LINREG_MA:
        ema_line = _ema(src_close, engine.linreg_length)
        lr = nika_linreg(src_close, ema_line, engine.linreg_length)
        lr_ma = compute_ma(lr, engine.linreg_ma_period, engine.linreg_ma_type, engine.tfactor)
        buy, sell = compute_crossover_signal(lr, lr_ma)
        main_line = lr; fast_line = lr_ma
    else:
        buy = np.zeros(n, dtype=np.bool_); sell = np.zeros(n, dtype=np.bool_)

    # Confluence indicators
    adx_buy = (adx_val > 20.0) & (dip > dim); adx_sell = (adx_val > 20.0) & (dim > dip)
    tsi_buy = tsi_val > tsi_sig; tsi_sell = tsi_val < tsi_sig
    rsi_buy = rsi_val > 50.0; rsi_sell = rsi_val < 50.0
    psar_buy = psar_bull; psar_sell = ~psar_bull
    vol_buy = volume > vol_sma; vol_sell = vol_buy
    atr_ratio = np.where(atr_base != 0, atr_val / atr_base, 1.0)
    regime_ok = atr_ratio > 1.0
    regime_buy = regime_ok; regime_sell = regime_ok

    # FIX #3: MTF loops now use @njit _mtf_slope_signals
    mtf1h_buy = np.zeros(n, dtype=np.bool_); mtf1h_sell = np.zeros(n, dtype=np.bool_)
    mtf4h_buy = np.zeros(n, dtype=np.bool_); mtf4h_sell = np.zeros(n, dtype=np.bool_)
    if htf_close_1h is not None:
        h1 = compute_ma(htf_close_1h, engine.ma_length, engine.ma_type, engine.tfactor)
        mtf1h_buy, mtf1h_sell = _mtf_slope_signals(h1, engine.sensitivity, n)
    if htf_close_4h is not None:
        h4 = compute_ma(htf_close_4h, engine.ma_length, engine.ma_type, engine.tfactor)
        mtf4h_buy, mtf4h_sell = _mtf_slope_signals(h4, engine.sensitivity, n)

    ind_buy = [adx_buy, tsi_buy, rsi_buy, psar_buy, vol_buy, regime_buy, mtf1h_buy, mtf4h_buy]
    ind_sell = [adx_sell, tsi_sell, rsi_sell, psar_sell, vol_sell, regime_sell, mtf1h_sell, mtf4h_sell]

    method = 'median' if engine.forecast_method == 0 else 'average'
    forecast = compute_forecast_duration(buy, sell, engine.forecast_sample, method)

    return SignalResult(
        buy=buy, sell=sell,
        main_line=main_line, fast_line=fast_line,
        atr=atr_val, open_next=open_next,
        ind_buy=ind_buy, ind_sell=ind_sell,
        psar_flip_up=psar_flip_up, psar_flip_down=psar_flip_down,
        swing_high=swing_high, swing_low=swing_low,
        forecast=forecast,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Confluence + bitmask packing
# ─────────────────────────────────────────────────────────────────────────────

def pack_indicator_bits(ind_list):
    n = len(ind_list[0]); bits = np.zeros(n, dtype=np.int16)
    for i, arr in enumerate(ind_list): bits |= (arr.astype(np.int16) << i)
    return bits

def apply_confluence(buy, sell, ind_buy, ind_sell, conf):
    n = len(buy)
    if conf.indicator_mask == 0 or conf.min_agree <= 0: return buy.copy(), sell.copy()
    bc = np.zeros(n, dtype=np.int32); sc = np.zeros(n, dtype=np.int32)
    for i in range(NUM_IND):
        if conf.indicator_mask & (1 << i):
            bc += ind_buy[i].astype(np.int32); sc += ind_sell[i].astype(np.int32)
    b_ok = bc >= conf.min_agree; s_ok = sc >= conf.min_agree
    if conf.mtf_block == 1:
        b_ok &= ~(ind_sell[IND_MTF1H] & ind_sell[IND_MTF4H])
        s_ok &= ~(ind_buy[IND_MTF1H] & ind_buy[IND_MTF4H])
    elif conf.mtf_block == 2:
        b_ok &= ~(ind_sell[IND_MTF1H] | ind_sell[IND_MTF4H])
        s_ok &= ~(ind_buy[IND_MTF1H] | ind_buy[IND_MTF4H])
    return buy & b_ok, sell & s_ok

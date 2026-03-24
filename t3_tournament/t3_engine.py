# T3 Tournament Engine
# All T3 combo generation + signal computation
# Structures: Single, Crossover, Triple, T3-of-Indicator
# Inputs: 12 price sources x 2 HA modes
# Parameters: length x vfactor x sensitivity x signal_mode

import numpy as np
from numba import njit
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Price source constants
# ---------------------------------------------------------------------------
SRC_CLOSE     = 0
SRC_OPEN      = 1
SRC_HIGH      = 2
SRC_LOW       = 3
SRC_HL2       = 4
SRC_HLC3      = 5
SRC_OHLC4     = 6
SRC_WEIGHTED  = 7   # (H+L+C+C)/4
SRC_HA_CLOSE  = 8
SRC_HA_HL2    = 9
SRC_HA_HLC3   = 10
SRC_HA_OHLC4  = 11
NUM_SOURCES   = 12
SOURCE_NAMES  = [
    'Close','Open','High','Low','HL2','HLC3','OHLC4','Weighted',
    'HA_Close','HA_HL2','HA_HLC3','HA_OHLC4'
]

# Structure constants
STRUCT_SINGLE     = 0
STRUCT_CROSS      = 1
STRUCT_TRIPLE     = 2
STRUCT_OF_IND     = 3
STRUCT_NAMES = ['Single','Crossover','Triple','T3ofIndicator']

# Signal mode constants
MODE_FLIP       = 0
MODE_SLOPE      = 1
MODE_PRICE_CROSS= 2
MODE_CROSSOVER  = 3
MODE_ZERO_CROSS = 4
MODE_ACCEL      = 5
MODE_NAMES = ['Flip','Slope','PriceCross','Crossover','ZeroCross','Accel']

# Indicator input for STRUCT_OF_IND
IND_RSI    = 0
IND_TSI    = 1
IND_ATR    = 2
IND_ADX    = 3
IND_DIDIFF = 4
IND_VOL    = 5
IND_MOM    = 6
IND_LINREG = 7
NUM_IND_INPUTS = 8
IND_INPUT_NAMES = ['RSI','TSI','ATR','ADX','DI_Diff','Volume','Momentum','LinReg']

# Modes where sensitivity has NO effect (always deduplicate to sens=1)
MODES_SENSITIVITY_INACTIVE = {MODE_ZERO_CROSS, MODE_PRICE_CROSS, MODE_CROSSOVER}


# ---------------------------------------------------------------------------
# Core Numba kernels
# ---------------------------------------------------------------------------

@njit
def _ema(data, period):
    alpha = 2.0 / (period + 1.0)
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i-1]
    return out

@njit
def _rma(data, period):
    alpha = 1.0 / period
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i-1]
    return out

@njit
def _sma(data, period):
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        s += data[i]
        if i >= period:
            s -= data[i - period]
        out[i] = s / min(i + 1, period)
    return out

@njit
def _gd(src, period, vf):
    e1 = _ema(src, period)
    e2 = _ema(e1, period)
    n = len(src)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = e1[i] * (1.0 + vf) - e2[i] * vf
    return out

@njit
def _t3(src, period, vf):
    return _gd(_gd(_gd(src, period, vf), period, vf), period, vf)

@njit
def heiken_ashi(o, h, l, c):
    n = len(c)
    ha_o = np.empty(n, dtype=np.float64)
    ha_h = np.empty(n, dtype=np.float64)
    ha_l = np.empty(n, dtype=np.float64)
    ha_c = np.empty(n, dtype=np.float64)
    ha_o[0] = o[0]
    ha_c[0] = (o[0] + h[0] + l[0] + c[0]) / 4.0
    ha_h[0] = h[0]
    ha_l[0] = l[0]
    for i in range(1, n):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2.0
        ha_c[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        ha_h[i] = max(h[i], max(ha_o[i], ha_c[i]))
        ha_l[i] = min(l[i], min(ha_o[i], ha_c[i]))
    return ha_o, ha_h, ha_l, ha_c

@njit
def atr_calc(h, l, c, period):
    n = len(c)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], max(abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    return _rma(tr, period)

@njit
def rsi_calc(c, period):
    n = len(c)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        d = c[i] - c[i-1]
        if d > 0: gain[i] = d
        else: loss[i] = -d
    ag = _rma(gain, period)
    al = _rma(loss, period)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = 100.0 - 100.0/(1.0 + ag[i]/al[i]) if al[i] != 0 else 100.0
    return out

@njit
def tsi_val(c, short_l, long_l, sig_l):
    n = len(c)
    mom = np.empty(n, dtype=np.float64)
    abm = np.empty(n, dtype=np.float64)
    mom[0] = 0.0; abm[0] = 0.0
    for i in range(1, n):
        m = c[i] - c[i-1]
        mom[i] = m
        abm[i] = abs(m)
    sm = _ema(_ema(mom, long_l), short_l)
    sa = _ema(_ema(abm, long_l), short_l)
    tv = np.empty(n, dtype=np.float64)
    for i in range(n):
        tv[i] = 100.0 * sm[i] / sa[i] if sa[i] != 0 else 0.0
    return tv

@njit
def adx_di(h, l, c, period):
    n = len(c)
    tr  = np.zeros(n, dtype=np.float64)
    pdm = np.zeros(n, dtype=np.float64)
    mdm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hd = h[i] - h[i-1]
        ld = l[i-1] - l[i]
        tr[i]  = max(h[i]-l[i], max(abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
        if hd > ld and hd > 0: pdm[i] = hd
        if ld > hd and ld > 0: mdm[i] = ld
    at = _rma(tr, period)
    ps = _rma(pdm, period)
    ms = _rma(mdm, period)
    dip = np.empty(n, dtype=np.float64)
    dim = np.empty(n, dtype=np.float64)
    adxv = np.empty(n, dtype=np.float64)
    dx  = np.empty(n, dtype=np.float64)
    for i in range(n):
        dip[i] = 100.0 * ps[i] / at[i] if at[i] != 0 else 0.0
        dim[i] = 100.0 * ms[i] / at[i] if at[i] != 0 else 0.0
        s = dip[i] + dim[i]
        dx[i]  = 100.0 * abs(dip[i] - dim[i]) / s if s != 0 else 0.0
    adxv = _rma(dx, period)
    return dip, dim, adxv

@njit
def nika_linreg(close, ema_line, length):
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
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
            d   = close[idx] - av2
            x   = float(k)
            sx += x; sy += d; sxx += x*x; sxy += x*d
        nn = float(length)
        denom = nn * sxx - sx * sx
        if denom != 0.0:
            slope = (nn * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / nn
            out[i] = intercept + slope * (nn - 1)
    return out

@njit
def compute_signal_mode(line, close_src, sensitivity, mode):
    n = len(line)
    buy  = np.zeros(n, dtype=np.bool_)
    sell = np.zeros(n, dtype=np.bool_)
    if mode == 0:  # FLIP
        direction = np.zeros(n, dtype=np.int32)
        for i in range(sensitivity, n):
            if   line[i] > line[i - sensitivity]: direction[i] =  1
            elif line[i] < line[i - sensitivity]: direction[i] = -1
            else: direction[i] = direction[i-1]
        for i in range(1, n):
            if direction[i] ==  1 and direction[i-1] !=  1: buy[i]  = True
            if direction[i] == -1 and direction[i-1] != -1: sell[i] = True
    elif mode == 1:  # SLOPE
        for i in range(sensitivity, n):
            if line[i] > line[i - sensitivity]: buy[i]  = True
            else:                                sell[i] = True
    elif mode == 2:  # PRICE_CROSS — sensitivity inactive
        for i in range(1, n):
            if close_src[i] > line[i] and close_src[i-1] <= line[i-1]: buy[i]  = True
            if close_src[i] < line[i] and close_src[i-1] >= line[i-1]: sell[i] = True
    elif mode == 4:  # ZERO_CROSS — sensitivity inactive
        for i in range(1, n):
            if line[i] > 0 and line[i-1] <= 0: buy[i]  = True
            if line[i] < 0 and line[i-1] >= 0: sell[i] = True
    elif mode == 5:  # ACCEL (slope of slope)
        slope = np.zeros(n, dtype=np.float64)
        for i in range(sensitivity, n):
            slope[i] = line[i] - line[i - sensitivity]
        for i in range(1, n):
            if slope[i] > slope[i-1] and slope[i] > 0: buy[i]  = True
            if slope[i] < slope[i-1] and slope[i] < 0: sell[i] = True
    return buy, sell

@njit
def compute_crossover(fast, slow):
    n = len(fast)
    buy  = np.zeros(n, dtype=np.bool_)
    sell = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if fast[i] > slow[i] and fast[i-1] <= slow[i-1]: buy[i]  = True
        if fast[i] < slow[i] and fast[i-1] >= slow[i-1]: sell[i] = True
    return buy, sell

@njit
def compute_crossover_slope_confirm(fast, slow, sensitivity):
    n = len(fast)
    buy  = np.zeros(n, dtype=np.bool_)
    sell = np.zeros(n, dtype=np.bool_)
    for i in range(sensitivity, n):
        cross_up   = fast[i] > slow[i] and fast[i-1] <= slow[i-1]
        cross_down = fast[i] < slow[i] and fast[i-1] >= slow[i-1]
        slow_up    = slow[i] > slow[i - sensitivity]
        slow_down  = slow[i] < slow[i - sensitivity]
        if cross_up   and slow_up:   buy[i]  = True
        if cross_down and slow_down: sell[i] = True
    return buy, sell

@njit
def compute_crossover_accel_confirm(fast, slow, sensitivity):
    n = len(fast)
    buy  = np.zeros(n, dtype=np.bool_)
    sell = np.zeros(n, dtype=np.bool_)
    for i in range(sensitivity + 1, n):
        cross_up   = fast[i] > slow[i] and fast[i-1] <= slow[i-1]
        cross_down = fast[i] < slow[i] and fast[i-1] >= slow[i-1]
        slow_accel_up   = (slow[i] - slow[i-1]) > (slow[i-1] - slow[i-1-sensitivity])
        slow_accel_down = (slow[i] - slow[i-1]) < (slow[i-1] - slow[i-1-sensitivity])
        if cross_up   and slow_accel_up:   buy[i]  = True
        if cross_down and slow_accel_down: sell[i] = True
    return buy, sell


# ---------------------------------------------------------------------------
# Price source builder
# ---------------------------------------------------------------------------

def get_price_source(open_, high, low, close, src_id):
    if src_id == SRC_CLOSE:    return close
    if src_id == SRC_OPEN:     return open_
    if src_id == SRC_HIGH:     return high
    if src_id == SRC_LOW:      return low
    if src_id == SRC_HL2:      return (high + low) / 2.0
    if src_id == SRC_HLC3:     return (high + low + close) / 3.0
    if src_id == SRC_OHLC4:    return (open_ + high + low + close) / 4.0
    if src_id == SRC_WEIGHTED: return (high + low + close * 2.0) / 4.0
    ha_o, ha_h, ha_l, ha_c = heiken_ashi(open_, high, low, close)
    if src_id == SRC_HA_CLOSE:  return ha_c
    if src_id == SRC_HA_HL2:    return (ha_h + ha_l) / 2.0
    if src_id == SRC_HA_HLC3:   return (ha_h + ha_l + ha_c) / 3.0
    if src_id == SRC_HA_OHLC4:  return (ha_o + ha_h + ha_l + ha_c) / 4.0
    return close


# ---------------------------------------------------------------------------
# Indicator input builder (for STRUCT_OF_IND)
# ---------------------------------------------------------------------------

def get_indicator_input(open_, high, low, close, volume, ind_id, period=14):
    if ind_id == IND_RSI:    return rsi_calc(close, period)
    if ind_id == IND_TSI:    return tsi_val(close, 5, 25, 14)
    if ind_id == IND_ATR:    return atr_calc(high, low, close, period)
    if ind_id == IND_ADX:
        _, _, adxv = adx_di(high, low, close, period)
        return adxv
    if ind_id == IND_DIDIFF:
        dip, dim, _ = adx_di(high, low, close, period)
        return dip - dim
    if ind_id == IND_VOL:    return volume.astype(np.float64)
    if ind_id == IND_MOM:
        n = len(close)
        mom = np.zeros(n, dtype=np.float64)
        for i in range(period, n):
            mom[i] = close[i] - close[i - period]
        return mom
    if ind_id == IND_LINREG:
        ema_l = _ema(close, period)
        return nika_linreg(close, ema_l, period)
    return close


# ---------------------------------------------------------------------------
# T3 Combo Config
# ---------------------------------------------------------------------------

class T3Config(NamedTuple):
    structure:   int    # STRUCT_*
    src_id:      int    # SRC_*
    slow_len:    int
    slow_vf:     float
    fast_len:    int    # only CROSS/TRIPLE
    fast_vf:     float  # only CROSS/TRIPLE
    mid_len:     int    # only TRIPLE
    mid_vf:      float  # only TRIPLE
    sensitivity: int
    signal_mode: int    # MODE_*
    cross_type:  int    # 0=standard 1=slope_confirm 2=accel_confirm
    ind_input:   int    # IND_* only for STRUCT_OF_IND
    ind_period:  int    # period for indicator


# ---------------------------------------------------------------------------
# Combo generators
# ---------------------------------------------------------------------------

SLOW_LENGTHS  = [5, 7, 10, 13, 15, 20, 25, 30, 34, 40, 50, 60, 75, 89, 100, 120, 144, 175, 200, 250]
FAST_LENGTHS  = [2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 20, 25, 30, 34, 40]
MID_LENGTHS   = [3, 5, 8, 13, 21, 34, 50, 75, 89, 120]
VFACTORS      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SENSITIVITIES = [1, 2, 3, 5, 8, 13]
SINGLE_MODES  = [MODE_FLIP, MODE_SLOPE, MODE_PRICE_CROSS, MODE_ZERO_CROSS, MODE_ACCEL]
CROSS_TYPES   = [0, 1, 2]
IND_PERIODS   = [7, 10, 14, 21]

PYRAMID_VF = [
    (0.1, 0.4, 0.7),
    (0.2, 0.5, 0.8),
    (0.3, 0.5, 0.7),
    (0.3, 0.6, 0.9),
    (0.5, 0.7, 0.9),
]


def generate_single_combos():
    combos = []
    for src in range(NUM_SOURCES):
        for sl in SLOW_LENGTHS:
            for vf in VFACTORS:
                for sens in SENSITIVITIES:
                    if sens >= sl: continue
                    for mode in SINGLE_MODES:
                        # ZeroCross and PriceCross: sensitivity inactive — only generate sens=1
                        if mode in MODES_SENSITIVITY_INACTIVE and sens != 1:
                            continue
                        combos.append(T3Config(
                            structure=STRUCT_SINGLE, src_id=src,
                            slow_len=sl, slow_vf=vf,
                            fast_len=0, fast_vf=0.0,
                            mid_len=0, mid_vf=0.0,
                            sensitivity=sens, signal_mode=mode,
                            cross_type=0, ind_input=0, ind_period=14,
                        ))
    return combos


def generate_crossover_combos():
    combos = []
    for src in range(NUM_SOURCES):
        for sl in SLOW_LENGTHS:
            for fl in FAST_LENGTHS:
                if fl >= sl: continue
                for svf in VFACTORS:
                    for fvf in VFACTORS:
                        for sens in SENSITIVITIES:
                            for ct in CROSS_TYPES:
                                # ct=0 (standard crossover): sensitivity inactive — only sens=1
                                if ct == 0 and sens != 1:
                                    continue
                                combos.append(T3Config(
                                    structure=STRUCT_CROSS, src_id=src,
                                    slow_len=sl, slow_vf=svf,
                                    fast_len=fl, fast_vf=fvf,
                                    mid_len=0, mid_vf=0.0,
                                    sensitivity=sens, signal_mode=MODE_CROSSOVER,
                                    cross_type=ct, ind_input=0, ind_period=14,
                                ))
    return combos


def generate_triple_combos():
    combos = []
    for src in range(NUM_SOURCES):
        for fl in [3, 5, 8, 13, 21]:
            for ml in MID_LENGTHS:
                if ml <= fl: continue
                for sl in SLOW_LENGTHS:
                    if sl <= ml: continue
                    for pvf in PYRAMID_VF:
                        for sens in SENSITIVITIES:
                            for mode in [MODE_FLIP, MODE_SLOPE, MODE_CROSSOVER]:
                                # CROSSOVER mode: sensitivity inactive — only sens=1
                                if mode == MODE_CROSSOVER and sens != 1:
                                    continue
                                combos.append(T3Config(
                                    structure=STRUCT_TRIPLE, src_id=src,
                                    slow_len=sl, slow_vf=pvf[2],
                                    fast_len=fl, fast_vf=pvf[0],
                                    mid_len=ml, mid_vf=pvf[1],
                                    sensitivity=sens, signal_mode=mode,
                                    cross_type=0, ind_input=0, ind_period=14,
                                ))
    return combos


def generate_t3_of_indicator_combos():
    combos = []
    for ind in range(NUM_IND_INPUTS):
        for ind_p in IND_PERIODS:
            for sl in SLOW_LENGTHS:
                for vf in VFACTORS:
                    for sens in SENSITIVITIES:
                        if sens >= sl: continue
                        for mode in SINGLE_MODES:
                            # ZeroCross and PriceCross: sensitivity inactive — only sens=1
                            if mode in MODES_SENSITIVITY_INACTIVE and sens != 1:
                                continue
                            combos.append(T3Config(
                                structure=STRUCT_OF_IND, src_id=SRC_CLOSE,
                                slow_len=sl, slow_vf=vf,
                                fast_len=0, fast_vf=0.0,
                                mid_len=0, mid_vf=0.0,
                                sensitivity=sens, signal_mode=mode,
                                cross_type=0, ind_input=ind, ind_period=ind_p,
                            ))
    return combos


def generate_all_combos():
    single = generate_single_combos()
    cross  = generate_crossover_combos()
    triple = generate_triple_combos()
    ofind  = generate_t3_of_indicator_combos()
    print(f"  T3 Single:         {len(single):>10,}")
    print(f"  T3 Crossover:      {len(cross):>10,}")
    print(f"  T3 Triple:         {len(triple):>10,}")
    print(f"  T3 of Indicator:   {len(ofind):>10,}")
    total = single + cross + triple + ofind
    print(f"  TOTAL:             {len(total):>10,}")
    return total


# ---------------------------------------------------------------------------
# Signal computer — one combo at a time
# ---------------------------------------------------------------------------

def compute_t3_signals(cfg, open_, high, low, close, volume):
    if cfg.structure == STRUCT_OF_IND:
        src = get_indicator_input(open_, high, low, close, volume, cfg.ind_input, cfg.ind_period)
        close_src = close
    else:
        src = get_price_source(open_, high, low, close, cfg.src_id)
        close_src = src if cfg.src_id in (SRC_CLOSE, SRC_HA_CLOSE) else close

    if cfg.structure == STRUCT_SINGLE or cfg.structure == STRUCT_OF_IND:
        main_line = _t3(src, cfg.slow_len, cfg.slow_vf)
        buy, sell = compute_signal_mode(main_line, close_src, cfg.sensitivity, cfg.signal_mode)
        return buy, sell, main_line

    elif cfg.structure == STRUCT_CROSS:
        slow_line = _t3(src, cfg.slow_len, cfg.slow_vf)
        fast_line = _t3(src, cfg.fast_len, cfg.fast_vf)
        if cfg.cross_type == 0:
            buy, sell = compute_crossover(fast_line, slow_line)
        elif cfg.cross_type == 1:
            buy, sell = compute_crossover_slope_confirm(fast_line, slow_line, cfg.sensitivity)
        else:
            buy, sell = compute_crossover_accel_confirm(fast_line, slow_line, cfg.sensitivity)
        return buy, sell, slow_line

    elif cfg.structure == STRUCT_TRIPLE:
        fast_t3 = _t3(src, cfg.fast_len, cfg.fast_vf)
        mid_t3  = _t3(src, cfg.mid_len,  cfg.mid_vf)
        slow_t3 = _t3(src, cfg.slow_len, cfg.slow_vf)
        if cfg.signal_mode == MODE_CROSSOVER:
            buy, sell = compute_crossover(fast_t3, slow_t3)
        elif cfg.signal_mode == MODE_SLOPE:
            buy, sell = compute_signal_mode(slow_t3, close_src, cfg.sensitivity, MODE_SLOPE)
        else:
            buy_s,  sell_s  = compute_signal_mode(slow_t3, close_src, cfg.sensitivity, MODE_FLIP)
            buy_fm, sell_fm = compute_crossover(fast_t3, mid_t3)
            buy  = buy_s  & (fast_t3 > mid_t3)
            sell = sell_s & (fast_t3 < mid_t3)
        return buy, sell, slow_t3

    n = len(close)
    return np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.float64)

"""
Module 1 — Data Loader
Loads pre-aggregated M10 OHLCV bars from MT5 CSV export.
Creates higher timeframe data for MTF alignment.
"""

import pandas as pd
import numpy as np
import time


def load_ohlcv(filepath: str, sep: str = '\t') -> pd.DataFrame:
    """
    Load MT5-exported OHLCV CSV.
    Handles MT5 format: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    Also handles generic CSV with date, open, high, low, close, volume columns.
    """
    print(f"[data_loader] Loading OHLCV from {filepath}...")
    t0 = time.time()

    df = pd.read_csv(filepath, sep=sep, dtype=str, low_memory=False)
    df.columns = [c.strip().upper().replace('<', '').replace('>', '') for c in df.columns]

    # Parse datetime
    if 'DATE' in df.columns and 'TIME' in df.columns:
        df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='mixed')
    elif 'DATETIME' in df.columns:
        df['datetime'] = pd.to_datetime(df['DATETIME'], format='mixed')
    else:
        df['datetime'] = pd.to_datetime(df.iloc[:, 0], format='mixed')

    df = df.set_index('datetime').sort_index()

    # Map columns
    col_map = {}
    for target, candidates in {
        'open':   ['OPEN', 'O'],
        'high':   ['HIGH', 'H'],
        'low':    ['LOW', 'L'],
        'close':  ['CLOSE', 'C'],
        'volume': ['TICKVOL', 'VOLUME', 'VOL', 'TICK_VOLUME', 'V'],
    }.items():
        for c in candidates:
            if c in df.columns:
                col_map[c] = target
                break

    df = df.rename(columns=col_map)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'volume' not in df.columns:
        df['volume'] = 0.0

    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

    elapsed = time.time() - t0
    print(f"[data_loader] Loaded {len(df):,} M10 bars in {elapsed:.1f}s")
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")
    return df


def prepare_multi_timeframe(
    base_ohlcv: pd.DataFrame,
    higher_tfs: list[str] = ['1h', '4h']
) -> dict[str, pd.DataFrame]:
    """
    Create higher timeframe OHLCV from M10 base for MTF alignment.
    Forward-fills HTF values to base index so arrays are same length.
    Returns dict: {'base': df, '1h': df_aligned, '4h': df_aligned}
    """
    result = {'base': base_ohlcv}

    for tf in higher_tfs:
        htf = base_ohlcv.resample(tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(subset=['open'])

        # Forward-fill to base timeframe index
        htf_aligned = htf.reindex(base_ohlcv.index, method='ffill')
        result[tf] = htf_aligned
        print(f"[data_loader] HTF {tf}: {len(htf):,} bars → aligned to {len(htf_aligned):,}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ohlcv = load_ohlcv(sys.argv[1])
        print(ohlcv.head(10))
        print(f"\nTotal bars: {len(ohlcv):,}")

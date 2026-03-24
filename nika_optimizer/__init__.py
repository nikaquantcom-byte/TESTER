"""
NikaQuant MEGA Optimizer V2
XAUUSD M10 Parameter Optimization for TradingView Pine Script

Architecture (V2):
  - signals_v2.py      — Vectorized indicators: T3 + T-Factor, voter/gate split
  - backtest_engine_v2.py — Numba @njit backtest with bitmask confluence
  - grid_search_v2.py   — 5-phase funnel: T3 core → indicators → confluence → management → validation
  - walk_forward_v2.py  — 9mo IS / 3mo OOS, embargo, frozen holdout, plateau scoring
  - visualize.py        — Heatmaps, equity curves, robustness analysis
  - data_loader.py      — M10 OHLCV CSV loader with MTF generation
  - run_mega.py         — Main orchestrator
"""

__version__ = '2.0.0'
__author__ = 'nikaquant'

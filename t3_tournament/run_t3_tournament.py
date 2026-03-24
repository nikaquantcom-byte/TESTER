# T3 Tournament Runner
# Run: python -m t3_tournament --data nika_optimizer/XAUUSD_M10.csv

import numpy as np
import argparse
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nika_optimizer.data_loader import load_ohlcv
from .t3_grid_search import (
    run_t3_tournament, save_results, load_results, print_top, save_csv,
)


def main():
    parser = argparse.ArgumentParser(description='T3 Tournament — isoleret T3 sweep')
    parser.add_argument('--data',    required=True,  help='Path to OHLCV CSV')
    parser.add_argument('--sep',     default='\t',   help='CSV separator')
    parser.add_argument('--output',  default='output_t3', help='Output folder')
    parser.add_argument('--cores',   type=int, default=None)
    parser.add_argument('--quick',   action='store_true', help='Sample 5K combos for quick test')
    parser.add_argument('--top',     type=int, default=50)
    parser.add_argument('--load',    default=None,   help='Load existing results pkl instead of running')
    parser.add_argument('--no-csv',  action='store_true', help='Skip CSV export')
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    n_cores = args.cores or os.cpu_count()

    print("\n" + "=" * 80)
    print("  T3 TOURNAMENT")
    print("  Isoleret T3 test — alle strukturer, inputs, modes, parametre")
    print(f"  Cores: {n_cores} | Output: {args.output}")
    print("=" * 80)

    print(f"\n[Data] Loading {args.data}...")
    from nika_optimizer.data_loader import load_ohlcv
    df = load_ohlcv(args.data, sep=args.sep)
    ohlcv = {
        'open':   df['open'].values.astype(np.float64),
        'high':   df['high'].values.astype(np.float64),
        'low':    df['low'].values.astype(np.float64),
        'close':  df['close'].values.astype(np.float64),
        'volume': df['volume'].values.astype(np.float64),
    }
    print(f"  {len(df):,} bars loaded")

    t0 = time.time()

    if args.load:
        print(f"\n[Load] {args.load}")
        results = load_results(args.load)
    else:
        results = run_t3_tournament(ohlcv, n_cores=n_cores, quick=args.quick)
        # Always save PKL
        pkl_path = f"{args.output}/t3_results.pkl"
        save_results(results, pkl_path)

    # Always save CSV (unless --no-csv)
    if not args.no_csv:
        csv_path = f"{args.output}/t3_results.csv"
        save_csv(results, csv_path)
        print(f"\n  Open in Excel: {os.path.abspath(csv_path)}")

    print_top(results, top_n=args.top)

    elapsed = time.time() - t0
    best_cfg, best_res, best_score = results[0]
    print(f"\n  Total time:  {elapsed/60:.1f} min")
    print(f"  Best score:  {best_score:.2f}")
    print(f"  Best PF:     {best_res[0]:.3f}")
    print(f"  Best WR:     {best_res[3]:.1f}%")
    print(f"  Best trades: {int(best_res[0])}")
    print()


if __name__ == '__main__':
    main()

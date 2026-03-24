"""
Entry point: python -m nika_optimizer --data "C:/path/to/XAUUSD_M10.csv"
Uses MEGA Optimizer V3 (universal) by default.
"""
import sys

if '--v1' in sys.argv:
    sys.argv.remove('--v1')
    from .run_optimizer import main
elif '--v2' in sys.argv:
    sys.argv.remove('--v2')
    from .run_mega import main
else:
    from .run_mega_v3 import main

if __name__ == '__main__':
    main()

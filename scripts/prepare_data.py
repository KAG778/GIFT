#!/usr/bin/env python3
"""End-to-end data preparation for GIFT.

Reads a wide CSV of daily SP500-style prices and converts it into the date-indexed
pickle that ``PortfolioEnv`` consumes. Runs three steps:

    1. Validate input CSV schema + ticker coverage (fail fast if missing).
    2. Convert CSV -> pickle via :func:`core.prepare_data.prepare_data`.
    3. Sanity-check the resulting pickle.

Default in/out paths match the configs in ``configs/``:

    data/sp500_prices.csv  -->  data/portfolio_5stocks.pkl

Usage:
    # Zero-arg form (expects CSV at the default location):
    python scripts/prepare_data.py

    # Custom CSV / tickers / date window:
    python scripts/prepare_data.py --csv path/to/prices.csv \\
        --output data/portfolio_5stocks.pkl \\
        --tickers TSLA NFLX AMZN MSFT JNJ \\
        --start 2018-01-01 --end 2024-12-31

Required CSV columns (case-insensitive, any column order):
    date | open | high | low | close | adjusted_close (or adj_close) | volume | symbol (or ticker)
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'core'))

from prepare_data import prepare_data  # noqa: E402

DEFAULT_TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']
DEFAULT_CSV = 'data/sp500_prices.csv'
DEFAULT_OUT = 'data/portfolio_5stocks.pkl'

REQUIRED = {'date', 'open', 'high', 'low', 'close', 'volume'}
PRICE_ALIASES = {'adjusted_close', 'adj_close', 'close'}
SYMBOL_ALIASES = {'symbol', 'ticker'}


def _norm(col: str) -> str:
    return col.lower().strip().replace(' ', '_')


def validate_csv(csv_path: Path, tickers):
    """Fail fast on schema/ticker problems before the slow conversion."""
    print(f"[1/3] Validating {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Place your file at '{DEFAULT_CSV}' or pass --csv <path>."
        )

    head = pd.read_csv(csv_path, nrows=1)
    by_norm = {_norm(c): c for c in head.columns}
    have = set(by_norm)

    missing = REQUIRED - have
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"  Saw: {sorted(have)}"
        )
    if not (PRICE_ALIASES & have):
        raise ValueError(f"CSV needs at least one of {sorted(PRICE_ALIASES)}; saw {sorted(have)}")
    if not (SYMBOL_ALIASES & have):
        raise ValueError(f"CSV needs at least one of {sorted(SYMBOL_ALIASES)}; saw {sorted(have)}")
    print(f"      columns OK: {sorted(have)}")

    sym_col_norm = next(iter(SYMBOL_ALIASES & have))
    sym_col = by_norm[sym_col_norm]
    syms = pd.read_csv(csv_path, usecols=[sym_col])
    counts = syms[sym_col].value_counts()
    print(f"      CSV has {len(counts)} unique symbols, {len(syms):,} rows")

    missing_tickers = [t for t in tickers if int(counts.get(t, 0)) == 0]
    if missing_tickers:
        raise ValueError(
            f"Tickers not present in CSV: {missing_tickers}\n"
            f"  Try --tickers <subset> or use a CSV that includes them."
        )
    for t in tickers:
        n = int(counts.get(t, 0))
        flag = " " if n >= 100 else "!"
        print(f"      {flag} {t}: {n:,} rows")


def verify_output(pkl_path: Path, tickers):
    """Confirm the produced pickle looks right."""
    print(f"[3/3] Verifying {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Output pickle is empty or not a dict")

    dates = sorted(data.keys())
    print(f"      dates: {len(dates):,} ({dates[0]} … {dates[-1]})")

    sample = data[dates[len(dates) // 2]]
    if 'price' not in sample:
        raise ValueError(f"Date entries missing 'price' key (e.g. {dates[0]})")

    seen = set()
    for d in dates:
        seen.update(data[d]['price'].keys())
    print(f"      tickers ever present: {sorted(seen)}")

    missing = [t for t in tickers if t not in seen]
    if missing:
        raise ValueError(f"Tickers never appear in output: {missing}")

    coverage = {t: sum(1 for d in dates if t in data[d]['price']) for t in tickers}
    for t in tickers:
        print(f"      {t}: present on {coverage[t]:,}/{len(dates):,} dates")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare SP500 price CSV into a GIFT-ready pickle.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:', 1)[1],
    )
    parser.add_argument('--csv', default=DEFAULT_CSV,
                        help=f'Input CSV (default: {DEFAULT_CSV})')
    parser.add_argument('--output', default=DEFAULT_OUT,
                        help=f'Output pickle (default: {DEFAULT_OUT})')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS,
                        help=f'Tickers to extract (default: {" ".join(DEFAULT_TICKERS)})')
    parser.add_argument('--start', default=None, help='Optional start date YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='Optional end date YYYY-MM-DD')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip the up-front CSV schema/ticker check')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.output)

    if not args.skip_validation:
        validate_csv(csv_path, args.tickers)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2/3] Converting {csv_path} -> {out_path}")
    prepare_data(str(csv_path), str(out_path), args.tickers, args.start, args.end)

    verify_output(out_path, args.tickers)
    print("Done.")


if __name__ == '__main__':
    main()

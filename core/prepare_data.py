"""
Data Preparation for Portfolio Optimization PPO

Loads SP500 CSV data, filters to 5 target tickers, saves as pickle.
Compatible with FinMemDataset / BacktestDataset format.
"""
import pandas as pd
import pickle
import argparse
from pathlib import Path

# Fallback only — callers pass --tickers / the tickers argument explicitly.
DEFAULT_TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']
DEFAULT_CSV = 'data/sp500_prices.csv'


def prepare_data(csv_path: str, output_path: str, tickers: list = None,
                 start_date: str = None, end_date: str = None):
    """Convert CSV to pickle format for BacktestDataset."""
    if tickers is None:
        tickers = DEFAULT_TICKERS

    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Standardize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ('date', 'time', 'timestamp', 'datetime'):
            col_map[col] = 'date'
        elif col_lower in ('close', 'adj close', 'adj_close', 'adjusted_close'):
            col_map[col] = col_lower.replace(' ', '_')
        elif col_lower in ('open', 'high', 'low', 'volume', 'ticker', 'symbol'):
            col_map[col] = col_lower
    df = df.rename(columns=col_map)

    if 'ticker' not in df.columns and 'symbol' in df.columns:
        df['ticker'] = df['symbol']

    df = df[df['ticker'].isin(tickers)]

    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]

    data = {}
    for _, row in df.iterrows():
        date_str = str(row['date'])
        if date_str not in data:
            data[date_str] = {'price': {}}

        ticker = row['ticker']
        close_val = row.get('close', 0)
        adj_close_val = row.get('adj_close', row.get('adjusted_close', close_val))

        data[date_str]['price'][ticker] = {
            'close': float(close_val) if pd.notna(close_val) else 0.0,
            'open': float(row.get('open', 0)) if pd.notna(row.get('open', 0)) else 0.0,
            'high': float(row.get('high', 0)) if pd.notna(row.get('high', 0)) else 0.0,
            'low': float(row.get('low', 0)) if pd.notna(row.get('low', 0)) else 0.0,
            'volume': float(row.get('volume', 0)) if pd.notna(row.get('volume', 0)) else 0.0,
            'adjusted_close': float(adj_close_val) if pd.notna(adj_close_val) else 0.0,
        }

    print(f"Processed {len(data)} dates for {len(tickers)} tickers")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {output_path}")
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--output', default='data/portfolio_5stocks.pkl')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS)
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    args = parser.parse_args()
    prepare_data(args.csv, args.output, args.tickers, args.start, args.end)

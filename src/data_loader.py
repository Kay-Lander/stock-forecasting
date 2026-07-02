import yfinance as yf
import pandas as pd 
import os 
from datetime import datetime 

# The 5 tickers qwe analyze
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

START_DATE = '2019-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
DATA_DIR = 'data/raw'

def _read_cache(path):
    '''
    Read a cached ticker CSV, tolerating both the clean single-header format
    (written by this module) and the legacy 3-row MultiIndex header that older
    yfinance downloads produced (Price / Ticker / Date rows).
    '''
    # Peek at the second line: legacy files have a 'Ticker' row there.
    with open(path) as f:
        f.readline()
        second = f.readline()

    if second.startswith('Ticker'):
        # Legacy format: header on row 0, junk 'Ticker' row 1, blank 'Date' row 2.
        df = pd.read_csv(path, skiprows=[1, 2], header=0)
        df = df.rename(columns={df.columns[0]: 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        df = pd.read_csv(path, index_col='Date', parse_dates=True)

    return df


def fetch_ticker(ticker ,start=START_DATE, end=END_DATE, force=False):
    '''
    Fetch stock data for one ticker.
    Caches to CSV - only re-downloads if force=True.
    Returns a clean DataFrame with DateTimeIndex.
    '''
    os.makedirs(DATA_DIR, exist_ok=True)
    path = f'{DATA_DIR}/{ticker}.csv'
    if os.path.exists(path) and not force:
        df = _read_cache(path)
        print(f'{ticker}: loaded from cache ({len(df)} rows)')
        return df

    print(f'{ticker}: downloading from yfinance...')
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f'No data returned for {ticker}')

    # yfinance returns MultiIndex columns like ('Close', 'AAPL') even for a
    # single ticker. Flatten to plain column names so df['Close'] is a Series
    # and the CSV has a normal single-row header.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = 'Date'

    df.to_csv(path)
    print(f'{ticker}:saved to {path} ({len(df)} rows)')
    return df

def fetch_all(force=False):
    '''Fetch all 5 ticker. Returns dict of DataFrames.'''
    data ={}
    for ticker in TICKERS:
        data[ticker] = fetch_ticker (ticker, force=force)
    print(f'\nAll tickers loaded. Date range: {START_DATE} to {END_DATE}')
    return data

def get_close_prices():
    '''Return DataFrame of closing prices for all tickers - one column per ticker.'''
    data = fetch_all()
    close_df = pd.DataFrame({
        ticker: df['Close'] for ticker, df in data.items()
    })
    return close_df

if __name__ == '__main__':
    data = fetch_all(force=True)
    for ticker, df in data.items():
        print(f'{ticker}: {df.shape}, {df.index.min().date()} to {df.index.max().date()}')

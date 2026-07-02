import pandas as pd 
import numpy as np


def add_technical_indicators(df):
    '''
    Add 12 technical indicator features to stock DataFrame.
    Input: DataFrame with columns [Open, High, Low, Close, Volume]
    Output:Same DataFrame with additional feature columns
    '''
    df = df.copy()

    # ── Moving Averages ──────────────────────────────────────────
    # Trend indicators - price above MA = bullish, below = bearish
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # MA crossover signal (1 = goldencross, -1 = death cross)
    df['MA_cross'] = np.where(df['MA_7'] > df['MA_21'], 1, -1)

    # ── Bollinger Bands ───────────────────────────────────────────
    # Volatility indicator — price outside bands = potential reversal
    df['BB_mid']   = df['Close'].rolling(20).mean()
    df['BB_std']   = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
 
    # ── RSI (Relative Strength Index) ─────────────────────────────
    # Momentum indicator — >70 = overbought, <30 = oversold
    delta  = df['Close'].diff()
    gain   = delta.where(delta > 0, 0).rolling(14).mean()
    loss   = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    # A zero-loss window means all gains → RSI is 100, not undefined.
    df.loc[loss == 0, 'RSI'] = 100
 
    # ── MACD (Moving Average Convergence Divergence) ──────────────
    # Trend + momentum — positive MACD = bullish momentum
    ema_12      = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26      = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']  = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']
 
    # ── Lag Features ──────────────────────────────────────────────
    # Yesterday's and last week's close — direct time-series signal
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag5'] = df['Close'].shift(5)
 
    # ── Returns ───────────────────────────────────────────────────
    df['Daily_return']  = df['Close'].pct_change()
    df['Weekly_return'] = df['Close'].pct_change(5)

    # ── Volume Features ───────────────────────────────────────────
    df['Volume_MA']    = df['Volume'].rolling(20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
 
    # Drop NaN rows created by rolling windows
    df = df.dropna()
 
    return df

def prepare_xgboost_data(df, target_col='Close', horizon=1):
    '''
    Prepare feature matrix X and target y for XGBoost.
    horizon: how many days ahead to predict (default=1)
    '''
    df = add_technical_indicators(df)
 
    # Target: next day's closing price
    df['Target'] = df[target_col].shift(-horizon)
    df = df.dropna()
 
    feature_cols = [
        'MA_7', 'MA_21', 'MA_50', 'MA_cross',
        'BB_width', 'RSI', 'MACD', 'MACD_hist',
        'Close_lag1', 'Close_lag5',
        'Daily_return', 'Weekly_return', 'Volume_ratio'
    ]
 
    X = df[feature_cols]
    y = df['Target']
    dates = df.index
 
    return X, y, dates




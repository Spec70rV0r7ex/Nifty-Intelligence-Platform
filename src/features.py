import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_stock(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
    df = df.sort_values('Date').reset_index(drop = True)
    return df

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha = 1/14, adjust = False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha = 1/14, adjust = False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    ema_12 = df['Close'].ewm(span = 12, adjust = False).mean()
    ema_26 = df['Close'].ewm(span = 26, adjust = False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span = 9, adjust = False).mean()
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Volatility_21d'] = df['Return_1d'].rolling(window = 21).std() * np.sqrt(252)
    sma_20 = df['Close'].rolling(window = 20).mean()
    std_20 = df['Close'].rolling(window = 20).std()
    df['BB_Upper'] = sma_20 + (2 * std_20)
    df['BB_Lower'] = sma_20 - (2 * std_20)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    vol_mean_50 = df['Volume'].rolling(window = 50).mean()
    vol_std_50 = df['Volume'].rolling(window = 50).std()
    df['Volume_Z_Score'] = (df['Volume'] - vol_mean_50) / (vol_std_50 + 1e-8)
    df['Price_to_VWAP'] = df['Close'] / df['VWAP']
    return df

def create_target_variables(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    df[f'Target_Return_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df[f'Target_Direction_{horizon}d'] = (df[f'Target_Return_{horizon}d'] > 0).astype(float)
    return df

def generate_features(file_path: str, prediction_horizon: int = 5) -> pd.DataFrame:
    df = load_and_preprocess_stock(file_path)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = create_target_variables(df, horizon = prediction_horizon)
    df = df.dropna().reset_index(drop = True)
    return df

if __name__ == "__main__":
    # Example usage:
    # file_path = "data/raw/RELIANCE.csv"
    # processed_df = generate_features(file_path)
    # processed_df.to_csv("data/processed/RELIANCE_features.csv", index = False)
    pass
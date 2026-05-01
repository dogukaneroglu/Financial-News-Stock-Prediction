"""
Feature engineering module for stock price data.
Calculates technical indicators and prepares features for modeling.
"""

import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
from typing import List, Tuple, Optional
import os


class FeatureEngineer:
    """Engineer features from stock price data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to stock price data.
        
        Args:
            df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        print("Calculating technical indicators...")
        
        # Moving Averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_14'] = df['Close'].rolling(window=14).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        df['RSI'] = momentum.rsi(df['Close'], window=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = volatility.BollingerBands(df['Close'])
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_mid'] = bollinger.bollinger_mavg()
        df['BB_width'] = df['BB_high'] - df['BB_low']
        
        # Volume indicators
        df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA_7']
        
        # On Balance Volume
        df['OBV'] = volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Average True Range (Volatility)
        df['ATR'] = volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=14
        )
        
        # Price changes
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_1d'] = df['Close'].pct_change(periods=1)
        df['Price_change_7d'] = df['Close'].pct_change(periods=7)
        
        # High-Low range
        df['HL_range'] = df['High'] - df['Low']
        df['HL_pct'] = (df['High'] - df['Low']) / df['Close']
        
        print(f"Added {len(df.columns)} features")
        
        return df
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Add lagged features (previous days' values).
        
        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: List of lag periods (e.g., [1, 3, 7])
            
        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found")
                continue
            
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Add rolling statistics features.
        
        Args:
            df: DataFrame
            columns: Columns to calculate rolling stats for
            windows: Window sizes (e.g., [7, 14, 30])
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found")
                continue
            
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
        
        return df
    
    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_column: str = 'Close',
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            df: DataFrame
            target_column: Column to predict
            horizon: Days ahead to predict
            
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        
        # Next day's price
        df['target_price'] = df[target_column].shift(-horizon)
        
        # Price change (regression target)
        df['target_change'] = df['target_price'] - df[target_column]
        df['target_change_pct'] = (df['target_price'] - df[target_column]) / df[target_column]
        
        # Direction (classification target)
        df['target_direction'] = (df['target_price'] > df[target_column]).astype(int)
        
        return df
    
    def merge_with_sentiment(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge price data with sentiment data.
        
        Args:
            price_df: Stock price DataFrame with technical indicators
            sentiment_df: Daily sentiment DataFrame
            
        Returns:
            Merged DataFrame
        """
        if sentiment_df.empty:
            print("Warning: Sentiment data is empty")
            return price_df
        
        # Ensure date columns are timezone-safe and aligned at day granularity.
        # Stock data may contain tz-aware timestamps (e.g., -04:00) while sentiment
        # data is usually tz-naive dates; normalize both before merging.
        price_df['Date'] = pd.to_datetime(price_df['Date'], utc=True).dt.tz_convert(None).dt.normalize()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True).dt.tz_convert(None).dt.normalize()
        
        # Merge on ticker and date
        merged_df = price_df.merge(
            sentiment_df,
            left_on=['Ticker', 'Date'],
            right_on=['ticker', 'date'],
            how='left'
        )
        
        # Fill missing sentiment values with neutral (0)
        sentiment_cols = [
            'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
            'positive_mean', 'negative_mean', 'neutral_mean', 'news_count'
        ]
        
        for col in sentiment_cols:
            if col in merged_df.columns:
                if col == 'news_count':
                    merged_df[col] = merged_df[col].fillna(0)
                elif col == 'neutral_mean':
                    merged_df[col] = merged_df[col].fillna(1.0)
                else:
                    merged_df[col] = merged_df[col].fillna(0.0)
        
        # Drop duplicate columns
        if 'ticker' in merged_df.columns:
            merged_df = merged_df.drop('ticker', axis=1)
        if 'date' in merged_df.columns:
            merged_df = merged_df.drop('date', axis=1)
        
        print(f"Merged data: {len(merged_df)} rows")
        print(f"Rows with news: {merged_df['news_count'].gt(0).sum()}")
        
        return merged_df
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM input.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            target_column: Target column name
            sequence_length: Number of time steps to look back
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Remove rows with NaN
        df_clean = df[feature_columns + [target_column]].dropna()
        
        if len(df_clean) < sequence_length:
            print(f"Warning: Not enough data points. Need {sequence_length}, have {len(df_clean)}")
            return np.array([]), np.array([])
        
        X, y = [], []
        
        for i in range(sequence_length, len(df_clean)):
            X.append(df_clean[feature_columns].iloc[i-sequence_length:i].values)
            y.append(df_clean[target_column].iloc[i])
        
        return np.array(X), np.array(y)
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str,
        data_dir: str = "data/processed"
    ):
        """
        Save processed data.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            data_dir: Directory to save file
        """
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")


def main():
    """Example usage of FeatureEngineer."""
    
    print("="*60)
    print("Feature Engineering Demo")
    print("="*60)
    
    # Load stock price data
    price_file = "data/raw/stock_prices.csv"
    
    if not os.path.exists(price_file):
        print(f"\nStock price file not found at {price_file}")
        print("Run stock_scraper.py first to collect price data.")
        return
    
    df = pd.read_csv(price_file, parse_dates=['Date'])
    print(f"\nLoaded {len(df)} price records")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Process each ticker separately
    processed_dfs = []
    
    for ticker in df['Ticker'].unique():
        print(f"\n{'='*60}")
        print(f"Processing {ticker}...")
        print('='*60)
        
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date')
        
        # Add technical indicators
        ticker_df = engineer.add_technical_indicators(ticker_df)
        
        # Add lag features for key indicators
        ticker_df = engineer.add_lag_features(
            ticker_df,
            columns=['Close', 'Volume', 'RSI'],
            lags=[1, 3, 7]
        )
        
        # Create target variable
        ticker_df = engineer.create_target_variable(ticker_df, horizon=1)
        
        processed_dfs.append(ticker_df)
    
    # Combine all tickers
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    
    print(f"\n{'='*60}")
    print("Feature Engineering Summary")
    print('='*60)
    print(f"\nTotal features created: {len(combined_df.columns)}")
    print(f"\nFeature columns:")
    for col in combined_df.columns:
        print(f"  - {col}")
    
    # Save processed data
    engineer.save_processed_data(combined_df, "stock_features.csv")
    
    # Try to merge with sentiment data
    sentiment_file = "data/processed/daily_sentiment_aggregated.csv"
    if os.path.exists(sentiment_file):
        print(f"\n{'='*60}")
        print("Merging with sentiment data...")
        print('='*60)
        
        sentiment_df = pd.read_csv(sentiment_file, parse_dates=['date'])
        merged_df = engineer.merge_with_sentiment(combined_df, sentiment_df)
        
        engineer.save_processed_data(merged_df, "stock_features_with_sentiment.csv")
        
        print("\nFinal dataset shape:", merged_df.shape)
        print("\nSample of merged data:")
        print(merged_df.head())
    else:
        print(f"\nSentiment file not found at {sentiment_file}")
        print("Run nlp_processor.py to generate sentiment data.")


if __name__ == "__main__":
    main()

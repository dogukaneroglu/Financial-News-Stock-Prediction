"""
Stock data scraper using yfinance API.
Fetches historical stock prices for specified tickers.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Optional
import time
import requests

# Configure yfinance with User-Agent to avoid rate limiting
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})


class StockScraper:
    """Scrapes stock price data using yfinance."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the stock scraper.
        
        Args:
            data_dir: Directory to save raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified (e.g., '2y', '5y')
            
        Returns:
            DataFrame with stock price data
        """
        try:
            print(f"Fetching {ticker}... (this may take a moment)")
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)
            
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                return pd.DataFrame()
            
            df['Ticker'] = ticker
            df.reset_index(inplace=True)
            
            print(f"[OK] Fetched {len(df)} days of data for {ticker}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        delay: float = 1.0
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified
            delay: Delay between requests in seconds
            
        Returns:
            Combined DataFrame with all stock data
        """
        all_data = []
        
        for ticker in tickers:
            print(f"\nFetching data for {ticker}...")
            df = self.fetch_stock_data(ticker, start_date, end_date, period)
            
            if not df.empty:
                all_data.append(df)
            
            time.sleep(delay)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal records fetched: {len(combined_df)}")
            return combined_df
        else:
            print("No data fetched for any ticker")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Save stock data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of the output file
        """
        if df.empty:
            print("Warning: DataFrame is empty, not saving")
            return
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load stock data from CSV file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame with stock data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def get_stock_info(self, ticker: str) -> dict:
        """
        Get basic information about a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            print(f"Error getting info for {ticker}: {str(e)}")
            return {}


def main():
    """Example usage of StockScraper."""
    
    # Recommended tickers for the project
    tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
    
    scraper = StockScraper()
    
    # Fetch data for multiple stocks
    print("="*60)
    print("Fetching stock data for US tech companies...")
    print("="*60)
    
    df = scraper.fetch_multiple_stocks(
        tickers=tickers,
        period="2y",
        delay=5.0  # Increased delay to avoid rate limiting
    )
    
    if not df.empty:
        # Save the data
        scraper.save_data(df, "stock_prices.csv")
        
        # Display basic statistics
        print("\n" + "="*60)
        print("Data Summary:")
        print("="*60)
        print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"\nStocks collected: {df['Ticker'].unique().tolist()}")
        print(f"\nRecords per stock:")
        print(df.groupby('Ticker').size())
        
        print("\nSample data:")
        print(df.head())
        
        # Get stock info
        print("\n" + "="*60)
        print("Stock Information:")
        print("="*60)
        for ticker in tickers:
            info = scraper.get_stock_info(ticker)
            if info:
                print(f"\n{ticker}: {info.get('name')}")
                print(f"  Sector: {info.get('sector')}")
                print(f"  Industry: {info.get('industry')}")


if __name__ == "__main__":
    main()

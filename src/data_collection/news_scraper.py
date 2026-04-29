"""
News scraper for financial news from Finviz.
Scrapes news headlines, timestamps, and sources for stock tickers.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Optional
import re


class NewsScraper:
    """Scrapes financial news from Finviz."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the news scraper.
        
        Args:
            data_dir: Directory to save raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        self.base_url = "https://finviz.com/quote.ashx?t="
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string from Finviz format.
        
        Args:
            date_str: Date string (e.g., "Jan-15-24 10:30AM", "Today 10:30AM")
            
        Returns:
            Datetime object or None if parsing fails
        """
        try:
            date_str = date_str.strip()
            
            # Handle "Today" and relative dates
            if date_str.startswith("Today"):
                time_part = date_str.split()[1]
                today = datetime.now().date()
                time_obj = datetime.strptime(time_part, "%I:%M%p").time()
                return datetime.combine(today, time_obj)
            
            # Handle specific date formats
            # Format: "Jan-15-24 10:30AM"
            if len(date_str.split()) == 2:
                date_part, time_part = date_str.split()
                
                # Parse date part
                month, day, year = date_part.split('-')
                year = '20' + year if len(year) == 2 else year
                
                # Parse time part
                time_obj = datetime.strptime(time_part, "%I:%M%p").time()
                
                # Combine
                month_num = datetime.strptime(month, "%b").month
                date_obj = datetime(int(year), month_num, int(day))
                return datetime.combine(date_obj.date(), time_obj)
            
            return None
            
        except Exception as e:
            print(f"Error parsing date '{date_str}': {str(e)}")
            return None
    
    def scrape_news(self, ticker: str) -> List[dict]:
        """
        Scrape news for a specific ticker from Finviz.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of news items with headline, date, source
        """
        url = f"{self.base_url}{ticker}"
        news_items = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the news table
            news_table = soup.find('table', {'id': 'news-table'})
            
            if not news_table:
                print(f"No news table found for {ticker}")
                return []
            
            # Parse news rows
            rows = news_table.find_all('tr')
            
            current_date = None
            
            for row in rows:
                try:
                    # Get date/time cell
                    td_date = row.find('td', {'align': 'right'})
                    # Get headline cell
                    td_headline = row.find('td', {'align': 'left'})
                    
                    if not td_headline:
                        continue
                    
                    # Extract headline and source
                    link = td_headline.find('a')
                    if not link:
                        continue
                    
                    headline = link.text.strip()
                    source = td_headline.find('span')
                    source_text = source.text.strip() if source else "Unknown"
                    
                    # Extract and parse date
                    if td_date:
                        date_str = td_date.text.strip()
                        parsed_date = self._parse_date(date_str)
                        if parsed_date:
                            current_date = parsed_date
                    
                    news_items.append({
                        'ticker': ticker,
                        'headline': headline,
                        'source': source_text,
                        'date': current_date,
                        'scraped_at': datetime.now()
                    })
                    
                except Exception as e:
                    print(f"Error parsing row: {str(e)}")
                    continue
            
            print(f"Scraped {len(news_items)} news items for {ticker}")
            return news_items
            
        except requests.RequestException as e:
            print(f"Error fetching news for {ticker}: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error for {ticker}: {str(e)}")
            return []
    
    def scrape_multiple_tickers(
        self,
        tickers: List[str],
        delay: float = 2.0
    ) -> pd.DataFrame:
        """
        Scrape news for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            delay: Delay between requests in seconds (respect rate limits)
            
        Returns:
            DataFrame with all news items
        """
        all_news = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Scraping news for {ticker}...")
            
            news_items = self.scrape_news(ticker)
            all_news.extend(news_items)
            
            # Respect rate limits
            if i < len(tickers):
                time.sleep(delay)
        
        if all_news:
            df = pd.DataFrame(all_news)
            print(f"\nTotal news items collected: {len(df)}")
            return df
        else:
            print("No news items collected")
            return pd.DataFrame()
    
    def save_news(self, df: pd.DataFrame, filename: str):
        """
        Save news data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of the output file
        """
        if df.empty:
            print("Warning: DataFrame is empty, not saving")
            return
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"News data saved to {filepath}")
    
    def load_news(self, filename: str) -> pd.DataFrame:
        """
        Load news data from CSV file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame with news data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, parse_dates=['date', 'scraped_at'])
        print(f"Loaded {len(df)} news items from {filepath}")
        return df
    
    def filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Filter news by date range.
        
        Args:
            df: News DataFrame
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        df['date'] = pd.to_datetime(df['date'])
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        filtered_df = df[(df['date'] >= start) & (df['date'] <= end)]
        print(f"Filtered to {len(filtered_df)} news items between {start_date} and {end_date}")
        
        return filtered_df


def main():
    """Example usage of NewsScraper."""
    
    tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
    
    scraper = NewsScraper()
    
    print("="*60)
    print("Scraping financial news from Finviz...")
    print("="*60)
    print("\nNote: This will take a few minutes. Please be patient.")
    print("Respecting rate limits with 2-second delays between requests.\n")
    
    # Scrape news
    df = scraper.scrape_multiple_tickers(tickers, delay=2.0)
    
    if not df.empty:
        # Save the data
        scraper.save_news(df, "financial_news.csv")
        
        # Display statistics
        print("\n" + "="*60)
        print("News Collection Summary:")
        print("="*60)
        
        print(f"\nTotal news items: {len(df)}")
        print(f"\nNews items per ticker:")
        print(df.groupby('ticker').size().sort_values(ascending=False))
        
        print(f"\nTop news sources:")
        print(df['source'].value_counts().head(10))
        
        if 'date' in df.columns and df['date'].notna().any():
            print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
        
        print("\nSample headlines:")
        print("-" * 60)
        for idx, row in df.head(10).iterrows():
            print(f"\n{row['ticker']}: {row['headline'][:80]}...")
            if pd.notna(row['date']):
                print(f"  Date: {row['date']}, Source: {row['source']}")
    else:
        print("\nNo news data was collected. This could be due to:")
        print("1. Network connectivity issues")
        print("2. Finviz blocking the requests")
        print("3. Changes in the Finviz website structure")
        print("\nConsider using alternative data sources or pre-collected datasets.")


if __name__ == "__main__":
    main()

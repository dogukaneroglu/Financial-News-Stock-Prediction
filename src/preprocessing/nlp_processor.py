"""
NLP processor for sentiment analysis of financial news.
Supports VADER and FinBERT models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

# VADER Sentiment
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available. Install with: pip install nltk")

# FinBERT (optional, for advanced users)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("FinBERT not available. Install with: pip install transformers torch")


class NLPProcessor:
    """Process financial news text for sentiment analysis."""
    
    def __init__(self, method: str = "finbert"):
        """
        Initialize NLP processor.
        
        Args:
            method: Sentiment analysis method ('finbert' or 'vader')
                   Default: 'finbert' - Financial domain-specific BERT model
        """
        self.method = method.lower()
        
        if self.method == "vader":
            self._init_vader()
        elif self.method == "finbert":
            self._init_finbert()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'vader' or 'finbert'")
    
    def _init_vader(self):
        """Initialize VADER sentiment analyzer."""
        if not VADER_AVAILABLE:
            raise ImportError("VADER not available. Install nltk first.")
        
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            print("VADER sentiment analyzer initialized")
        except LookupError:
            print("Downloading VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)
            self.analyzer = SentimentIntensityAnalyzer()
            print("VADER sentiment analyzer initialized")
    
    def _init_finbert(self):
        """Initialize FinBERT model for financial sentiment."""
        if not FINBERT_AVAILABLE:
            raise ImportError("FinBERT requires transformers and torch")
        
        print("Loading FinBERT model (this may take a while)...")
        
        model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"FinBERT model loaded on {self.device}")
    
    def analyze_sentiment_vader(self, text: str) -> dict:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        scores = self.analyzer.polarity_scores(text)
        
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def analyze_sentiment_finbert(self, text: str) -> dict:
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # FinBERT outputs: [positive, negative, neutral]
        positive = probs[0].item()
        negative = probs[1].item()
        neutral = probs[2].item()
        
        # Calculate compound score (similar to VADER)
        compound = positive - negative
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'compound': compound
        }
    
    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment using the selected method.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.method == "vader":
            return self.analyze_sentiment_vader(text)
        elif self.method == "finbert":
            return self.analyze_sentiment_finbert(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def process_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of news and add sentiment scores.
        
        Args:
            df: DataFrame with 'headline' column
            
        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty or 'headline' not in df.columns:
            print("Warning: Empty DataFrame or missing 'headline' column")
            return df
        
        print(f"Analyzing sentiment for {len(df)} news items...")
        
        sentiments = []
        for idx, text in enumerate(df['headline']):
            if idx % 100 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(df)} items...")
            
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        # Add sentiment columns
        df['sentiment_positive'] = [s['positive'] for s in sentiments]
        df['sentiment_negative'] = [s['negative'] for s in sentiments]
        df['sentiment_neutral'] = [s['neutral'] for s in sentiments]
        df['sentiment_compound'] = [s['compound'] for s in sentiments]
        
        # Add sentiment label
        df['sentiment_label'] = df['sentiment_compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        print(f"Sentiment analysis complete!")
        print(f"\nSentiment distribution:")
        print(df['sentiment_label'].value_counts())
        
        return df
    
    def aggregate_daily_sentiment(
        self,
        df: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment by date and ticker.
        
        Args:
            df: DataFrame with sentiment scores
            date_column: Name of the date column
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        if df.empty:
            return df
        
        # Convert date to date only (remove time)
        df['date_only'] = pd.to_datetime(df[date_column]).dt.date
        
        # Aggregate by ticker and date
        agg_df = df.groupby(['ticker', 'date_only']).agg({
            'sentiment_compound': ['mean', 'std', 'min', 'max'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'headline': 'count'
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = [
            'ticker', 'date',
            'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
            'positive_mean', 'negative_mean', 'neutral_mean',
            'news_count'
        ]
        
        # Fill NaN std with 0 (happens when only 1 news item)
        agg_df['sentiment_std'] = agg_df['sentiment_std'].fillna(0)
        
        print(f"\nAggregated sentiment for {len(agg_df)} ticker-date combinations")
        
        return agg_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, data_dir: str = "data/processed"):
        """
        Save processed data with sentiment scores.
        
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
    """Example usage of NLPProcessor."""
    
    print("="*60)
    print("FinBERT Sentiment Analysis Demo")
    print("="*60)
    
    # Initialize with FinBERT (financial domain-specific model)
    print("\nInitializing FinBERT model...")
    print("Note: First run will download the model (~440MB)")
    processor = NLPProcessor(method="finbert")
    
    # Test with sample headlines
    sample_headlines = [
        "Apple reports record-breaking quarterly earnings",
        "Tesla stock plummets after disappointing delivery numbers",
        "Google announces new AI features for search",
        "Microsoft faces regulatory scrutiny over acquisition",
        "Amazon workers strike over working conditions"
    ]
    
    print("\nAnalyzing sample headlines:\n")
    for headline in sample_headlines:
        sentiment = processor.analyze_sentiment(headline)
        print(f"Headline: {headline}")
        print(f"  Compound: {sentiment['compound']:.3f}")
        print(f"  Pos: {sentiment['positive']:.3f}, "
              f"Neg: {sentiment['negative']:.3f}, "
              f"Neu: {sentiment['neutral']:.3f}")
        print()
    
    # Try to load and process actual news data
    news_file = "data/raw/financial_news.csv"
    if os.path.exists(news_file):
        print("\n" + "="*60)
        print("Processing saved news data...")
        print("="*60)
        
        df = pd.read_csv(news_file)
        df = processor.process_news_dataframe(df)
        
        # Aggregate by date
        if 'date' in df.columns:
            agg_df = processor.aggregate_daily_sentiment(df)
            
            print("\nSample of aggregated daily sentiment:")
            print(agg_df.head(10))
            
            # Save processed data
            processor.save_processed_data(df, "news_with_sentiment.csv")
            processor.save_processed_data(agg_df, "daily_sentiment_aggregated.csv")
    else:
        print(f"\nNews file not found at {news_file}")
        print("Run news_scraper.py first to collect news data.")


if __name__ == "__main__":
    main()

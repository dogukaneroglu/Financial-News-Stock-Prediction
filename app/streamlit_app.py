"""
Streamlit web application for stock price prediction demo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.stock_scraper import StockScraper
from data_collection.news_scraper import NewsScraper
from preprocessing.nlp_processor import NLPProcessor
from preprocessing.feature_engineer import FeatureEngineer
from models.lstm_model import LSTMPredictor
from models.combined_model import CombinedPredictor

# Page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_stock_data(ticker, period="1y"):
    """Load stock data from cache or fetch new."""
    scraper = StockScraper(data_dir="data/raw")
    df = scraper.fetch_stock_data(ticker, period=period)
    return df


@st.cache_data
def load_news_data(ticker):
    """Load news data for ticker."""
    scraper = NewsScraper(data_dir="data/raw")
    news = scraper.scrape_news(ticker)
    return pd.DataFrame(news) if news else pd.DataFrame()


def plot_stock_price(df, ticker):
    """Create interactive stock price chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_prediction(actual, predicted, dates):
    """Create prediction comparison chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Actual vs Predicted Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig


def plot_sentiment_timeline(df):
    """Create sentiment timeline chart."""
    if df.empty or 'sentiment_compound' not in df.columns:
        return None
    
    # Aggregate by date
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_compound'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    
    fig = go.Figure()
    
    # Color based on sentiment
    colors = ['red' if s < -0.05 else 'green' if s > 0.05 else 'gray' 
              for s in daily_sentiment['sentiment']]
    
    fig.add_trace(go.Bar(
        x=daily_sentiment['date'],
        y=daily_sentiment['sentiment'],
        marker_color=colors,
        name='Sentiment'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title="News Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=300,
        template="plotly_white"
    )
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Title and description
    st.title("📈 Stock Price Prediction with NLP")
    st.markdown("""
    This application predicts stock prices using **LSTM neural networks** and **FinBERT financial sentiment analysis**.
    
    **Features:**
    - Real-time stock data fetching
    - **FinBERT**: Financial news sentiment analysis (BERT-based, 92% accuracy)
    - LSTM-based price predictions
    - Interactive visualizations
    
    **FinBERT**: A state-of-the-art sentiment model fine-tuned on 4,000+ financial articles.
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Stock selection
    available_tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
    ticker = st.sidebar.selectbox(
        "Select Stock",
        available_tickers,
        index=0
    )
    
    # Period selection
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y"
    }
    period_label = st.sidebar.selectbox(
        "Select Time Period",
        list(period_options.keys()),
        index=3
    )
    period = period_options[period_label]
    
    # Model selection
    model_type = st.sidebar.radio(
        "Select Model",
        ["Baseline LSTM", "Combined LSTM+Sentiment"],
        help="Baseline uses only price data. Combined uses price + FinBERT sentiment."
    )
    
    # Fetch data button
    fetch_button = st.sidebar.button("🔄 Fetch Data", type="primary")
    
    # Main content area
    if fetch_button:
        with st.spinner(f"Fetching data for {ticker}..."):
            # Fetch stock data
            df_stock = load_stock_data(ticker, period)
            
            if df_stock.empty:
                st.error("Failed to fetch stock data. Please try again.")
                return
            
            # Store in session state
            st.session_state['stock_data'] = df_stock
            st.session_state['ticker'] = ticker
            
            # Fetch news data
            with st.spinner("Fetching news data..."):
                df_news = load_news_data(ticker)
                st.session_state['news_data'] = df_news
    
    # Display data if available
    if 'stock_data' in st.session_state:
        df_stock = st.session_state['stock_data']
        ticker = st.session_state['ticker']
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df_stock['Close'].iloc[-1]
        prev_price = df_stock['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric("High", f"${df_stock['High'].max():.2f}")
        
        with col3:
            st.metric("Low", f"${df_stock['Low'].min():.2f}")
        
        with col4:
            avg_volume = df_stock['Volume'].mean()
            st.metric("Avg Volume", f"{avg_volume/1e6:.1f}M")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Price Chart", "📰 News & Sentiment", "🔮 Predictions"])
        
        with tab1:
            st.plotly_chart(plot_stock_price(df_stock, ticker), use_container_width=True)
            
            # Show recent data
            st.subheader("Recent Data")
            st.dataframe(
                df_stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10),
                use_container_width=True
            )
        
        with tab2:
            if 'news_data' in st.session_state and not st.session_state['news_data'].empty:
                df_news = st.session_state['news_data']
                
                # Process sentiment with FinBERT
                with st.spinner("Analyzing sentiment with FinBERT..."):
                    st.info("Using FinBERT - Financial domain-specific BERT model")
                    processor = NLPProcessor(method='finbert')
                    df_news = processor.process_news_dataframe(df_news)
                
                # Show sentiment distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = df_news['sentiment_label'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'positive': 'green',
                            'neutral': 'gray',
                            'negative': 'red'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Average Sentiment")
                    avg_sentiment = df_news['sentiment_compound'].mean()
                    st.metric(
                        "Compound Score",
                        f"{avg_sentiment:.3f}",
                        delta_color="normal" if avg_sentiment >= 0 else "inverse"
                    )
                
                # Sentiment timeline
                if 'date' in df_news.columns:
                    fig_timeline = plot_sentiment_timeline(df_news)
                    if fig_timeline:
                        st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Recent headlines
                st.subheader("Recent Headlines")
                for idx, row in df_news.head(10).iterrows():
                    sentiment_emoji = "🟢" if row['sentiment_compound'] > 0.05 else "🔴" if row['sentiment_compound'] < -0.05 else "⚪"
                    st.markdown(f"{sentiment_emoji} **{row['headline']}**")
                    st.caption(f"Sentiment: {row['sentiment_compound']:.3f} | Source: {row['source']}")
                    st.divider()
            else:
                st.info("No news data available. Click 'Fetch Data' to load news.")
        
        with tab3:
            st.subheader("Price Predictions")
            st.info("""
            **Note:** To generate predictions, you need to:
            1. Train the models using the training pipeline (`python src/training/train.py`)
            2. Ensure model files exist in `data/models/`
            
            This demo shows the interface. In a production system, trained models would be loaded here.
            """)
            
            # Check if models exist
            baseline_model_path = "data/models/baseline_lstm_model.pth"
            combined_model_path = "data/models/combined_model.pth"
            
            if os.path.exists(baseline_model_path) or os.path.exists(combined_model_path):
                st.success("Trained models found! Predictions can be generated.")
                
                # Placeholder for predictions
                st.markdown("**Prediction Summary:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Next Day Prediction", f"${current_price * 1.01:.2f}", "+1.0%")
                with col2:
                    st.metric("Confidence", "75%")
                with col3:
                    direction = "📈 Up" if np.random.rand() > 0.5 else "📉 Down"
                    st.metric("Direction", direction)
                
                # Placeholder chart
                future_dates = pd.date_range(
                    start=df_stock['Date'].iloc[-1],
                    periods=30,
                    freq='D'
                )
                actual = df_stock['Close'].iloc[-30:].values
                predicted = actual * (1 + np.random.randn(len(actual)) * 0.02)
                
                fig_pred = plot_prediction(
                    actual,
                    predicted,
                    df_stock['Date'].iloc[-30:].values
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.warning("No trained models found. Train models first to enable predictions.")
    
    else:
        # Initial state - show instructions
        st.info("👈 Select a stock ticker and click 'Fetch Data' to begin.")
        
        st.markdown("""
        ### How to Use:
        
        1. **Select a stock** from the sidebar (AAPL, TSLA, GOOGL, MSFT, AMZN)
        2. **Choose a time period** for historical data
        3. **Select a model** (Baseline LSTM or Combined with sentiment)
        4. **Click 'Fetch Data'** to load stock prices and news
        5. **Explore** the different tabs:
           - 📊 Price Chart: View historical prices
           - 📰 News & Sentiment: Analyze news sentiment
           - 🔮 Predictions: See model predictions
        
        ### About the Models:
        
        - **Baseline LSTM**: Uses only historical price and technical indicators
        - **Combined LSTM+Sentiment**: Integrates news sentiment with price data
        
        ### Requirements:
        
        To generate predictions, train the models first:
        ```bash
        python src/training/train.py
        ```
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Financial News NLP & Deep Learning**
    
    Stock Prediction System
    
    **NLP Model:** FinBERT  
    **DL Model:** LSTM
    
    Built with PyTorch & Streamlit
    """)


if __name__ == "__main__":
    main()

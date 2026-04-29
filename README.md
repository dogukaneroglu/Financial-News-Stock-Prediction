# Financial News NLP and Deep Learning for Stock Prediction

A comprehensive stock price prediction system that combines **Deep Learning (LSTM)** with **Natural Language Processing (NLP)** to predict stock prices using both historical price data and financial news sentiment analysis.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline that:
- Collects real-time stock price data and financial news
- Analyzes sentiment of financial news using NLP
- Predicts future stock prices using LSTM neural networks
- Provides an interactive web dashboard for visualization and predictions

## 🌟 Features

- **Data Collection**
  - Real-time stock price fetching using `yfinance`
  - Web scraping of financial news from Finviz
  - Automated data pipeline

- **NLP Sentiment Analysis**
  - **FinBERT**: State-of-the-art financial sentiment analysis (Primary method)
  - VADER: Fast rule-based sentiment (Alternative method)
  - Daily sentiment aggregation and feature extraction

- **Deep Learning Models**
  - Baseline LSTM model (price data only)
  - Combined LSTM+Sentiment model
  - Technical indicators integration (RSI, MACD, Bollinger Bands)

- **Interactive Dashboard**
  - Streamlit web application
  - Real-time data visualization
  - Model predictions display
  - News sentiment timeline

## 📁 Project Structure

```
Financial News NLP and Deep Learning for Stock Prediction/
├── data/
│   ├── raw/                      # Raw data (prices, news)
│   ├── processed/                # Processed features
│   └── models/                   # Trained model files
├── src/
│   ├── data_collection/
│   │   ├── stock_scraper.py      # Stock price fetching
│   │   └── news_scraper.py       # News scraping
│   ├── preprocessing/
│   │   ├── nlp_processor.py      # Sentiment analysis
│   │   └── feature_engineer.py   # Technical indicators
│   ├── models/
│   │   ├── lstm_model.py         # Baseline LSTM
│   │   └── combined_model.py     # Combined model
│   ├── training/
│   │   └── train.py              # Training pipeline
│   └── evaluation/
│       └── evaluate.py           # Model evaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_nlp_analysis.ipynb
│   └── 03_model_experiments.ipynb
├── app/
│   └── streamlit_app.py          # Web dashboard
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster FinBERT inference

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Financial News NLP and Deep Learning for Stock Prediction"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup FinBERT model (Recommended - Financial Domain-Specific):**
   ```bash
   python setup_finbert.py
   ```
   
   This will download the FinBERT model (~440MB) from HuggingFace.
   
   **Alternative - VADER (Faster but less accurate):**
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

## 📊 Usage

### Step 1: Collect Data

**Fetch stock prices:**
```bash
python src/data_collection/stock_scraper.py
```

**Scrape financial news:**
```bash
python src/data_collection/news_scraper.py
```

### Step 2: Process Data

**Analyze sentiment:**
```bash
python src/preprocessing/nlp_processor.py
```

**Engineer features:**
```bash
python src/preprocessing/feature_engineer.py
```

### Step 3: Train Models

**Train both baseline and combined models:**
```bash
python src/training/train.py --model-type both --epochs 100
```

**Train only baseline:**
```bash
python src/training/train.py --model-type baseline --epochs 100
```

### Step 4: Explore Results

**Open Jupyter notebooks:**
```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and explore:
- `01_data_exploration.ipynb` - Analyze collected data
- `02_nlp_analysis.ipynb` - Sentiment analysis insights
- `03_model_experiments.ipynb` - Model training and evaluation

### Step 5: Run Web Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🛠️ Technical Details

### Data Sources

- **Stock Prices**: Yahoo Finance via `yfinance` API
- **Financial News**: Finviz.com via web scraping
- **Target Stocks**: AAPL, TSLA, GOOGL, MSFT, AMZN

### Models

#### Baseline LSTM Model
- **Input**: Historical price data + technical indicators
- **Architecture**: 2-layer LSTM (64 hidden units) + Dense layers
- **Sequence Length**: 60 days
- **Features**: OHLCV, MA, RSI, MACD, Bollinger Bands, ATR

#### Combined LSTM+Sentiment Model
- **Input**: Price features + sentiment scores
- **Architecture**: LSTM for prices + Dense for sentiment → Combined
- **Sentiment Features**: Mean, std, min, max scores + news count

### Sentiment Analysis - FinBERT

FinBERT is a BERT-based model fine-tuned on financial news:
- **Model**: ProsusAI/finbert (HuggingFace)
- **Training**: 4,000+ financial news articles
- **Output**: Positive, Negative, Neutral probabilities
- **Advantage**: Domain-specific, understands financial terminology
- **Performance**: ~92% accuracy on financial sentiment tasks

### Technical Indicators

- Moving Averages (MA 7, 14, 30)
- Exponential Moving Averages (EMA 12, 26)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Average True Range (ATR)
- On-Balance Volume (OBV)

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R² Score**
- **Directional Accuracy** (% of correct up/down predictions)

## 📈 Expected Results

Based on similar financial prediction systems:

- **Baseline LSTM**: 50-55% directional accuracy
- **Combined Model**: 55-60% directional accuracy
- **RMSE**: 2-5% of stock price
- **MAPE**: 3-7%

**Note**: Stock prediction is inherently difficult. These results demonstrate the model's ability to learn patterns, but should not be used for actual trading decisions.

## 🔬 Model Training Tips

1. **Start Small**: Train on a single stock (e.g., AAPL) first
2. **Monitor Overfitting**: Use early stopping and dropout
3. **Feature Scaling**: StandardScaler is applied automatically
4. **Sequence Length**: Experiment with 30, 60, or 90 days
5. **Hyperparameters**: Adjust learning rate, hidden size, layers

## 📚 Project Components

### Data Collection
- Automated stock price fetching
- Rate-limited news scraping (respects robots.txt)
- CSV storage for reproducibility

### NLP Processing
- VADER sentiment analysis (rule-based, fast)
- Optional FinBERT (transformer-based, more accurate)
- Entity recognition with spaCy
- Daily sentiment aggregation

### Feature Engineering
- 20+ technical indicators
- Lag features for temporal patterns
- Rolling statistics
- Sentiment-price alignment

### Deep Learning
- PyTorch implementation
- GPU support (CUDA)
- Model checkpointing
- Early stopping

### Visualization
- Interactive Plotly charts
- Jupyter notebooks for exploration
- Streamlit dashboard for demos

## 🎓 Educational Use

This project is designed as a **thesis/capstone project** demonstrating:

1. **Data Engineering**: Collection, cleaning, preprocessing
2. **NLP**: Sentiment analysis, text processing
3. **Deep Learning**: LSTM, time series prediction
4. **Software Engineering**: Modular design, documentation
5. **Visualization**: Interactive dashboards, notebooks

## ⚠️ Limitations

1. **Market Complexity**: Stock prices are influenced by many factors beyond news and history
2. **Data Quality**: Web scraping may miss some news or have delays
3. **Sentiment Simplicity**: VADER is general-purpose, not finance-specific
4. **No Real Trading**: This is an educational project, not a trading system
5. **Computational Cost**: Training requires significant compute resources

## 🔮 Future Improvements

- [ ] Add more news sources (Bloomberg, Reuters)
- [ ] Implement attention mechanisms
- [ ] Try transformer models (GPT, BERT for time series)
- [ ] Add more stocks and forex/crypto support
- [ ] Real-time prediction API
- [ ] Backtesting framework
- [ ] Risk management features
- [ ] Social media sentiment (Twitter, Reddit)

## 📖 References

### FinBERT
- **Paper**: "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
  - Author: Dogu Araci (2019)
  - Link: https://arxiv.org/abs/1908.10063
- **Model**: ProsusAI/finbert on HuggingFace
  - Link: https://huggingface.co/ProsusAI/finbert
  - Trained on: 4,000+ financial news articles
  - Accuracy: ~92% on financial sentiment tasks

### Deep Learning
- **LSTM**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- **BERT**: Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"

### Technical Analysis
- Murphy, John J. (1999) - "Technical Analysis of the Financial Markets"

### Libraries & Frameworks
- PyTorch: https://pytorch.org/
- HuggingFace Transformers: https://huggingface.co/transformers/
- Streamlit: https://streamlit.io/

## 📝 License

This project is created for educational purposes. 

**Disclaimer**: This software is for educational and research purposes only. Do not use it for actual trading or investment decisions. The authors are not responsible for any financial losses.

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome!

## 💡 Tips for Presentation

### For Your Thesis Defense:

1. **Start with the Problem**: Why is stock prediction hard? Why combine NLP + LSTM?
2. **Show the Pipeline**: Walk through data → features → model → results
3. **Demo the App**: Live demonstration with Streamlit is impressive
4. **Discuss Results**: Be honest about limitations, show directional accuracy
5. **Future Work**: Discuss what you'd improve with more time/resources

### Key Talking Points:

- **Integration**: Combining multiple AI techniques (NLP + Deep Learning)
- **End-to-End**: Complete pipeline from data collection to deployment
- **Practical Application**: Real-world use case in finance
- **Challenges**: Dealing with noisy data, temporal dependencies, market volatility

## 📧 Support

For issues or questions about this project, please check:
1. Error messages in the console
2. Requirements.txt for missing packages
3. Data directory structure
4. Model file paths

## 🎉 Acknowledgments

- Yahoo Finance for stock data API
- Finviz for financial news
- NLTK and HuggingFace for NLP tools
- PyTorch and Streamlit communities

---

**Built with ❤️ for academic research in AI and Finance**

Good luck with your thesis! 🎓

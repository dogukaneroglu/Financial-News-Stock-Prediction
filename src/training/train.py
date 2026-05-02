"""
Training pipeline for stock prediction models.
Trains both baseline LSTM and combined LSTM+Sentiment models.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMPredictor
from models.combined_model import CombinedPredictor


def train_baseline_model(
    data_path: str,
    model_dir: str = "data/models",
    sequence_length: int = 60,
    epochs: int = 100,
    target_column: str = "target_change_pct"
):
    """
    Train baseline LSTM model (price data only).
    
    Args:
        data_path: Path to processed data CSV
        model_dir: Directory to save model
        sequence_length: Sequence length for LSTM
        epochs: Number of training epochs
        target_column: Target variable to predict
    """
    print("\n" + "="*80)
    print("TRAINING BASELINE LSTM MODEL")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    # Define price features (excluding sentiment)
    price_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_7', 'MA_14', 'MA_30',
        'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'BB_high', 'BB_low', 'BB_mid', 'BB_width',
        'Volume_MA_7', 'Volume_ratio',
        'ATR', 'Price_change', 'HL_range', 'HL_pct'
    ]
    
    # Filter to only available features
    available_features = [f for f in price_features if f in df.columns]
    print(f"Using {len(available_features)} price features")
    
    # Initialize model
    predictor = LSTMPredictor(
        sequence_length=sequence_length,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32
    )
    
    # Prepare data
    X, y = predictor.prepare_data(df, available_features, target_column=target_column)
    
    # Split and scale
    X_train, y_train, X_val, y_val, X_test, y_test = predictor.split_and_scale_data(X, y)
    
    # Build model
    predictor.build_model(input_size=len(available_features))
    
    # Train
    train_losses, val_losses = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        patience=15
    )
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "baseline_lstm_model.pth")
    predictor.save_model(model_path)
    
    # Save test data for evaluation
    test_data_path = os.path.join(model_dir, "baseline_test_data.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    print(f"Test data saved to {test_data_path}")
    
    print("\n" + "="*80)
    print("BASELINE MODEL TRAINING COMPLETE")
    print("="*80)
    
    return predictor, X_test, y_test


def train_combined_model(
    data_path: str,
    model_dir: str = "data/models",
    sequence_length: int = 60,
    epochs: int = 100,
    target_column: str = "target_change_pct",
    dropout: float = 0.3,
    weight_decay: float = 1e-4
):
    """
    Train combined LSTM+Sentiment model.
    
    Args:
        data_path: Path to processed data with sentiment CSV
        model_dir: Directory to save model
        sequence_length: Sequence length for LSTM
        epochs: Number of training epochs
        target_column: Target variable to predict
        dropout: Dropout rate for combined model
        weight_decay: L2 regularization strength
    """
    print("\n" + "="*80)
    print("TRAINING COMBINED LSTM+SENTIMENT MODEL")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    # Check if sentiment columns exist
    base_sentiment_cols = [
        'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
        'positive_mean', 'negative_mean', 'neutral_mean', 'news_count'
    ]
    lagged_sentiment_cols = []
    for lag in [1, 2, 3]:
        lagged_sentiment_cols.extend([
            f'sentiment_mean_lag_{lag}', f'sentiment_std_lag_{lag}',
            f'sentiment_min_lag_{lag}', f'sentiment_max_lag_{lag}',
            f'positive_mean_lag_{lag}', f'negative_mean_lag_{lag}',
            f'neutral_mean_lag_{lag}', f'news_count_lag_{lag}'
        ])
    sentiment_cols = base_sentiment_cols + lagged_sentiment_cols
    
    missing_cols = [col for col in sentiment_cols if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: Missing sentiment columns: {missing_cols}")
        print("Cannot train combined model without sentiment data.")
        return None, None, None, None
    
    # Define price features
    price_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_7', 'MA_14', 'MA_30',
        'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'BB_high', 'BB_low', 'BB_mid', 'BB_width',
        'Volume_MA_7', 'Volume_ratio',
        'ATR', 'Price_change', 'HL_range', 'HL_pct'
    ]
    
    # Filter to available features
    available_price_features = [f for f in price_features if f in df.columns]
    print(f"Using {len(available_price_features)} price features")
    print(f"Using {len(sentiment_cols)} sentiment features")
    
    # Initialize model
    predictor = CombinedPredictor(
        sequence_length=sequence_length,
        hidden_size=64,
        num_layers=2,
        dropout=dropout,
        learning_rate=0.001,
        batch_size=32,
        weight_decay=weight_decay
    )
    
    # Prepare data
    X_price, X_sentiment, y = predictor.prepare_data(
        df,
        price_columns=available_price_features,
        sentiment_columns=sentiment_cols,
        target_column=target_column
    )
    
    # Split and scale
    (X_price_train, X_sent_train, y_train,
     X_price_val, X_sent_val, y_val,
     X_price_test, X_sent_test, y_test) = predictor.split_and_scale_data(
        X_price, X_sentiment, y
    )
    
    # Build model
    predictor.build_model(
        price_input_size=len(available_price_features),
        sentiment_input_size=len(sentiment_cols)
    )
    
    # Train
    train_losses, val_losses = predictor.train(
        X_price_train, X_sent_train, y_train,
        X_price_val, X_sent_val, y_val,
        epochs=epochs,
        patience=15
    )
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "combined_model.pth")
    predictor.save_model(model_path)
    
    # Save test data
    test_data_path = os.path.join(model_dir, "combined_test_data.npz")
    np.savez(
        test_data_path,
        X_price_test=X_price_test,
        X_sent_test=X_sent_test,
        y_test=y_test
    )
    print(f"Test data saved to {test_data_path}")
    
    print("\n" + "="*80)
    print("COMBINED MODEL TRAINING COMPLETE")
    print("="*80)
    
    return predictor, X_price_test, X_sent_test, y_test


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    parser.add_argument(
        '--baseline-data',
        type=str,
        default='data/processed/stock_features.csv',
        help='Path to baseline data (price features only)'
    )
    parser.add_argument(
        '--combined-data',
        type=str,
        default='data/processed/stock_features_with_sentiment.csv',
        help='Path to combined data (price + sentiment features)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='data/models',
        help='Directory to save models'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Sequence length for LSTM'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['baseline', 'combined', 'both'],
        default='both',
        help='Which model(s) to train'
    )
    parser.add_argument(
        '--target-column',
        type=str,
        choices=['target_price', 'target_change', 'target_change_pct'],
        default='target_change_pct',
        help='Target variable to predict'
    )
    parser.add_argument(
        '--combined-dropout',
        type=float,
        default=0.3,
        help='Dropout rate for combined model regularization'
    )
    parser.add_argument(
        '--combined-weight-decay',
        type=float,
        default=1e-4,
        help='L2 regularization (weight decay) for combined model'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STOCK PREDICTION MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Target column: {args.target_column}")
    print(f"  Combined dropout: {args.combined_dropout}")
    print(f"  Combined weight decay: {args.combined_weight_decay}")
    print(f"  Model directory: {args.model_dir}")
    
    # Train baseline model
    if args.model_type in ['baseline', 'both']:
        if os.path.exists(args.baseline_data):
            try:
                train_baseline_model(
                    data_path=args.baseline_data,
                    model_dir=args.model_dir,
                    sequence_length=args.sequence_length,
                    epochs=args.epochs,
                    target_column=args.target_column
                )
            except Exception as e:
                print(f"\nError training baseline model: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nBaseline data not found at {args.baseline_data}")
            print("Run feature engineering first.")
    
    # Train combined model
    if args.model_type in ['combined', 'both']:
        if os.path.exists(args.combined_data):
            try:
                train_combined_model(
                    data_path=args.combined_data,
                    model_dir=args.model_dir,
                    sequence_length=args.sequence_length,
                    epochs=args.epochs,
                    target_column=args.target_column,
                    dropout=args.combined_dropout,
                    weight_decay=args.combined_weight_decay
                )
            except Exception as e:
                print(f"\nError training combined model: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nCombined data not found at {args.combined_data}")
            print("Run feature engineering with sentiment data first.")
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

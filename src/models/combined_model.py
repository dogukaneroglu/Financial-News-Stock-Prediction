"""
Combined model that integrates LSTM for price data with sentiment features.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import os


class CombinedStockDataset(Dataset):
    """PyTorch Dataset for stock sequences with sentiment."""
    
    def __init__(self, X_price: np.ndarray, X_sentiment: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X_price: Price feature sequences (samples, sequence_length, price_features)
            X_sentiment: Sentiment features (samples, sentiment_features)
            y: Target values (samples,)
        """
        self.X_price = torch.FloatTensor(X_price)
        self.X_sentiment = torch.FloatTensor(X_sentiment)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X_price)
    
    def __getitem__(self, idx):
        return self.X_price[idx], self.X_sentiment[idx], self.y[idx]


class CombinedModel(nn.Module):
    """Combined LSTM + Sentiment model for stock prediction."""
    
    def __init__(
        self,
        price_input_size: int,
        sentiment_input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize combined model.
        
        Args:
            price_input_size: Number of price features
            sentiment_input_size: Number of sentiment features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(CombinedModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for price sequences
        self.lstm = nn.LSTM(
            input_size=price_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dense layers for sentiment features
        self.sentiment_fc1 = nn.Linear(sentiment_input_size, 16)
        self.sentiment_relu1 = nn.ReLU()
        self.sentiment_dropout1 = nn.Dropout(dropout)
        self.sentiment_fc2 = nn.Linear(16, 8)
        self.sentiment_relu2 = nn.ReLU()
        
        # Combined layers
        self.combined_fc1 = nn.Linear(hidden_size + 8, 32)
        self.combined_relu = nn.ReLU()
        self.combined_dropout = nn.Dropout(dropout)
        self.combined_fc2 = nn.Linear(32, 16)
        self.combined_relu2 = nn.ReLU()
        self.output = nn.Linear(16, 1)
    
    def forward(self, x_price, x_sentiment):
        """
        Forward pass.
        
        Args:
            x_price: Price sequences (batch, sequence_length, price_features)
            x_sentiment: Sentiment features (batch, sentiment_features)
            
        Returns:
            Predictions (batch, 1)
        """
        # LSTM for price data
        lstm_out, _ = self.lstm(x_price)
        lstm_last = lstm_out[:, -1, :]
        
        # Dense network for sentiment
        sentiment_out = self.sentiment_fc1(x_sentiment)
        sentiment_out = self.sentiment_relu1(sentiment_out)
        sentiment_out = self.sentiment_dropout1(sentiment_out)
        sentiment_out = self.sentiment_fc2(sentiment_out)
        sentiment_out = self.sentiment_relu2(sentiment_out)
        
        # Combine
        combined = torch.cat([lstm_last, sentiment_out], dim=1)
        
        # Final layers
        out = self.combined_fc1(combined)
        out = self.combined_relu(out)
        out = self.combined_dropout(out)
        out = self.combined_fc2(out)
        out = self.combined_relu2(out)
        out = self.output(out)
        
        return out


class CombinedPredictor:
    """Wrapper class for combined model training and prediction."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize combined predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            device: Device to use ('cuda' or 'cpu')
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler_price = StandardScaler()
        self.scaler_sentiment = StandardScaler()
        self.scaler_y = StandardScaler()
        
        print(f"Using device: {self.device}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        price_columns: List[str],
        sentiment_columns: List[str],
        target_column: str = 'target_price'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for combined model.
        
        Args:
            df: DataFrame with features
            price_columns: List of price feature column names
            sentiment_columns: List of sentiment feature column names
            target_column: Target column name
            
        Returns:
            Tuple of (X_price, X_sentiment, y) arrays
        """
        all_columns = price_columns + sentiment_columns + [target_column]
        df_clean = df[all_columns].dropna().copy()
        
        if len(df_clean) < self.sequence_length:
            raise ValueError(f"Not enough data. Need {self.sequence_length}, have {len(df_clean)}")
        
        print(f"Preparing data from {len(df_clean)} samples...")
        
        # Create price sequences
        X_price = []
        X_sentiment = []
        y = []
        
        for i in range(self.sequence_length, len(df_clean)):
            # Price sequence (last N days)
            X_price.append(df_clean[price_columns].iloc[i-self.sequence_length:i].values)
            
            # Sentiment features (current day only)
            X_sentiment.append(df_clean[sentiment_columns].iloc[i].values)
            
            # Target
            y.append(df_clean[target_column].iloc[i])
        
        X_price = np.array(X_price)
        X_sentiment = np.array(X_sentiment)
        y = np.array(y).reshape(-1, 1)
        
        print(f"Created {len(X_price)} sequences")
        print(f"Price shape: {X_price.shape}, Sentiment shape: {X_sentiment.shape}, Target shape: {y.shape}")
        
        return X_price, X_sentiment, y
    
    def split_and_scale_data(
        self,
        X_price: np.ndarray,
        X_sentiment: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple:
        """
        Split data into train/val/test and scale.
        
        Args:
            X_price: Price feature sequences
            X_sentiment: Sentiment features
            y: Target values
            test_size: Proportion of test data
            val_size: Proportion of validation data
            
        Returns:
            Tuple of scaled train/val/test sets
        """
        # Split
        indices = np.arange(len(X_price))
        idx_temp, idx_test = train_test_split(indices, test_size=test_size, shuffle=False)
        val_size_adjusted = val_size / (1 - test_size)
        idx_train, idx_val = train_test_split(idx_temp, test_size=val_size_adjusted, shuffle=False)
        
        X_price_train, X_price_val, X_price_test = X_price[idx_train], X_price[idx_val], X_price[idx_test]
        X_sent_train, X_sent_val, X_sent_test = X_sentiment[idx_train], X_sentiment[idx_val], X_sentiment[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        
        print(f"\nData split:")
        print(f"  Train: {len(X_price_train)} samples")
        print(f"  Val:   {len(X_price_val)} samples")
        print(f"  Test:  {len(X_price_test)} samples")
        
        # Scale price features
        n_samples, n_steps, n_features = X_price_train.shape
        X_price_train_2d = X_price_train.reshape(-1, n_features)
        X_price_val_2d = X_price_val.reshape(-1, n_features)
        X_price_test_2d = X_price_test.reshape(-1, n_features)
        
        X_price_train_scaled = self.scaler_price.fit_transform(X_price_train_2d)
        X_price_val_scaled = self.scaler_price.transform(X_price_val_2d)
        X_price_test_scaled = self.scaler_price.transform(X_price_test_2d)
        
        X_price_train_scaled = X_price_train_scaled.reshape(len(X_price_train), n_steps, n_features)
        X_price_val_scaled = X_price_val_scaled.reshape(len(X_price_val), n_steps, n_features)
        X_price_test_scaled = X_price_test_scaled.reshape(len(X_price_test), n_steps, n_features)
        
        # Scale sentiment features
        X_sent_train_scaled = self.scaler_sentiment.fit_transform(X_sent_train)
        X_sent_val_scaled = self.scaler_sentiment.transform(X_sent_val)
        X_sent_test_scaled = self.scaler_sentiment.transform(X_sent_test)
        
        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        return (X_price_train_scaled, X_sent_train_scaled, y_train_scaled,
                X_price_val_scaled, X_sent_val_scaled, y_val_scaled,
                X_price_test_scaled, X_sent_test_scaled, y_test_scaled)
    
    def build_model(self, price_input_size: int, sentiment_input_size: int):
        """Build combined model."""
        self.model = CombinedModel(
            price_input_size=price_input_size,
            sentiment_input_size=sentiment_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"\nModel architecture:")
        print(self.model)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
    
    def train(
        self,
        X_price_train: np.ndarray,
        X_sent_train: np.ndarray,
        y_train: np.ndarray,
        X_price_val: np.ndarray,
        X_sent_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        patience: int = 10
    ):
        """Train the combined model."""
        train_dataset = CombinedStockDataset(X_price_train, X_sent_train, y_train.flatten())
        val_dataset = CombinedStockDataset(X_price_val, X_sent_val, y_val.flatten())
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"\nTraining for {epochs} epochs...")
        print("="*60)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_price, batch_sent, batch_y in train_loader:
                batch_price = batch_price.to(self.device)
                batch_sent = batch_sent.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_price, batch_sent)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_price, batch_sent, batch_y in val_loader:
                    batch_price = batch_price.to(self.device)
                    batch_sent = batch_sent.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_price, batch_sent)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, X_price: np.ndarray, X_sentiment: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        
        X_price_tensor = torch.FloatTensor(X_price).to(self.device)
        X_sent_tensor = torch.FloatTensor(X_sentiment).to(self.device)
        
        with torch.no_grad():
            predictions_scaled = self.model(X_price_tensor, X_sent_tensor).cpu().numpy()
        
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save model and scalers."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_price': self.scaler_price,
            'scaler_sentiment': self.scaler_sentiment,
            'scaler_y': self.scaler_y,
            'config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, price_input_size: int, sentiment_input_size: int):
        """Load model and scalers."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.sequence_length = checkpoint['config']['sequence_length']
        self.hidden_size = checkpoint['config']['hidden_size']
        self.num_layers = checkpoint['config']['num_layers']
        self.dropout = checkpoint['config']['dropout']
        
        self.build_model(price_input_size, sentiment_input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler_price = checkpoint['scaler_price']
        self.scaler_sentiment = checkpoint['scaler_sentiment']
        self.scaler_y = checkpoint['scaler_y']
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Combined Model module - import this in training scripts")

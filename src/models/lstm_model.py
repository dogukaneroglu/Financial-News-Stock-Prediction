"""
Baseline LSTM model for stock price prediction.
Uses only price and technical indicator features.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os
import pickle


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature sequences (samples, sequence_length, features)
            y: Target values (samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM model for stock price prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, sequence_length, input_size)
            
        Returns:
            Predictions (batch, 1)
        """
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step
        last_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


class LSTMPredictor:
    """Wrapper class for LSTM model training and prediction."""
    
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
        Initialize LSTM predictor.
        
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
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        print(f"Using device: {self.device}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        target_column: str = 'target_price'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Remove rows with NaN
        df_clean = df[feature_columns + [target_column]].dropna().copy()
        
        if len(df_clean) < self.sequence_length:
            raise ValueError(f"Not enough data. Need {self.sequence_length}, have {len(df_clean)}")
        
        print(f"Preparing data from {len(df_clean)} samples...")
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(df_clean)):
            X.append(df_clean[feature_columns].iloc[i-self.sequence_length:i].values)
            y.append(df_clean[target_column].iloc[i])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        print(f"Created {len(X)} sequences of length {self.sequence_length}")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        return X, y
    
    def split_and_scale_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple:
        """
        Split data into train/val/test and scale.
        
        Args:
            X: Feature sequences
            y: Target values
            test_size: Proportion of test data
            val_size: Proportion of validation data
            
        Returns:
            Tuple of scaled train/val/test sets
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, shuffle=False
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Scale features
        n_samples, n_steps, n_features = X_train.shape
        
        # Reshape for scaling
        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler_X.fit_transform(X_train_2d)
        X_val_scaled = self.scaler_X.transform(X_val_2d)
        X_test_scaled = self.scaler_X.transform(X_test_2d)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(len(X_train), n_steps, n_features)
        X_val_scaled = X_val_scaled.reshape(len(X_val), n_steps, n_features)
        X_test_scaled = X_test_scaled.reshape(len(X_test), n_steps, n_features)
        
        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        return (X_train_scaled, y_train_scaled,
                X_val_scaled, y_val_scaled,
                X_test_scaled, y_test_scaled)
    
    def build_model(self, input_size: int):
        """
        Build LSTM model.
        
        Args:
            input_size: Number of input features
        """
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"\nModel architecture:")
        print(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        patience: int = 10
    ):
        """
        Train the LSTM model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            patience: Early stopping patience
        """
        # Create datasets
        train_dataset = StockDataset(X_train, y_train.flatten())
        val_dataset = StockDataset(X_val, y_val.flatten())
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
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
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
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
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions (unscaled)
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save model and scalers."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, input_size: int):
        """Load model and scalers."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.sequence_length = checkpoint['config']['sequence_length']
        self.hidden_size = checkpoint['config']['hidden_size']
        self.num_layers = checkpoint['config']['num_layers']
        self.dropout = checkpoint['config']['dropout']
        
        self.build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("LSTM Model module - import this in training scripts")

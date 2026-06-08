"""
Ablation study: measure impact of sentiment features on classification performance.

Compares:
- Baseline (price + technical indicators only)
- Combined (price + sentiment)
- Sentiment-only (sentiment features alone, if feasible)
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import LSTMModel
from models.combined_model import CombinedModel


class BaselineDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CombinedDataset(Dataset):
    def __init__(self, X_price: np.ndarray, X_sent: np.ndarray, y: np.ndarray):
        self.X_price = torch.FloatTensor(X_price)
        self.X_sent = torch.FloatTensor(X_sent)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X_price)

    def __getitem__(self, idx):
        return self.X_price[idx], self.X_sent[idx], self.y[idx]


def temporal_split(n: int, test_size: float = 0.15, val_size: float = 0.15):
    indices = np.arange(n)
    idx_temp, idx_test = train_test_split(indices, test_size=test_size, shuffle=False)
    val_adj = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_adj, shuffle=False)
    return idx_train, idx_val, idx_test


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.flatten().astype(int)
    y_prob = y_prob.flatten()
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.35, 0.66, 0.01):
        y_pred = (y_prob >= t).astype(int)
        score = accuracy_score(y_true, y_pred)
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.flatten().astype(int)
    y_prob = y_prob.flatten()
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "Threshold": float(threshold),
    }


def train_baseline(
    Xp_train, y_train, Xp_val, y_val, Xp_test, y_test,
    n_feat: int, epochs: int, device: torch.device
) -> Dict[str, float]:
    model = LSTMModel(
        input_size=n_feat, hidden_size=64, num_layers=2, dropout=0.2
    ).to(device)
    
    pos = float(y_train.sum())
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    train_loader = DataLoader(BaselineDataset(Xp_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(BaselineDataset(Xp_val, y_val), batch_size=32, shuffle=False)
    
    best_val, wait, patience = float("inf"), 0, 12
    best_state = None
    
    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).squeeze(1)
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(1), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device).squeeze(1)
                val_loss += criterion(model(bx).squeeze(1), by).item()
        val_loss /= max(len(val_loader), 1)
        
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(torch.FloatTensor(Xp_val).to(device)).squeeze(1)).cpu().numpy()
        test_prob = torch.sigmoid(model(torch.FloatTensor(Xp_test).to(device)).squeeze(1)).cpu().numpy()
    
    thr = find_best_threshold(y_val, val_prob)
    return compute_metrics(y_test, test_prob, thr)


def train_combined(
    Xp_train, Xs_train, y_train, Xp_val, Xs_val, y_val, Xp_test, Xs_test, y_test,
    n_feat: int, n_sent: int, epochs: int, device: torch.device
) -> Dict[str, float]:
    model = CombinedModel(
        price_input_size=n_feat, sentiment_input_size=n_sent,
        hidden_size=64, num_layers=2, dropout=0.2
    ).to(device)
    
    pos = float(y_train.sum())
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    train_loader = DataLoader(CombinedDataset(Xp_train, Xs_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(CombinedDataset(Xp_val, Xs_val, y_val), batch_size=32, shuffle=False)
    
    best_val, wait, patience = float("inf"), 0, 12
    best_state = None
    
    for _ in range(epochs):
        model.train()
        for bxp, bxs, by in train_loader:
            bxp, bxs, by = bxp.to(device), bxs.to(device), by.to(device).squeeze(1)
            optimizer.zero_grad()
            loss = criterion(model(bxp, bxs).squeeze(1), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bxp, bxs, by in val_loader:
                bxp, bxs, by = bxp.to(device), bxs.to(device), by.to(device).squeeze(1)
                val_loss += criterion(model(bxp, bxs).squeeze(1), by).item()
        val_loss /= max(len(val_loader), 1)
        
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(
            model(torch.FloatTensor(Xp_val).to(device), torch.FloatTensor(Xs_val).to(device)).squeeze(1)
        ).cpu().numpy()
        test_prob = torch.sigmoid(
            model(torch.FloatTensor(Xp_test).to(device), torch.FloatTensor(Xs_test).to(device)).squeeze(1)
        ).cpu().numpy()
    
    thr = find_best_threshold(y_val, val_prob)
    return compute_metrics(y_test, test_prob, thr)


def main():
    parser = argparse.ArgumentParser(description="Ablation study: sentiment impact on classification")
    parser.add_argument("--data", type=str, default="data/processed/stock_features_with_sentiment.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--out-csv", type=str, default="data/models/classification/ablation_results.csv")
    args = parser.parse_args()
    
    df = pd.read_csv(args.data, parse_dates=["Date"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Dataset: {args.data}")
    print(f"Samples: {len(df)}, Tickers: {df['Ticker'].nunique()}")
    
    # Define features
    price_cols = [
        "Open", "High", "Low", "Close", "Volume", "MA_7", "MA_14", "MA_30", "EMA_12", "EMA_26",
        "RSI", "MACD", "MACD_signal", "MACD_diff", "BB_high", "BB_low", "BB_mid", "BB_width",
        "Volume_MA_7", "Volume_ratio", "ATR", "Price_change", "HL_range", "HL_pct",
    ]
    sentiment_cols = [
        "sentiment_mean", "news_count", "positive_mean", "negative_mean",
        "sentiment_mean_lag_1", "news_count_lag_1", "positive_mean_lag_1", "negative_mean_lag_1",
    ]
    
    price_cols = [c for c in price_cols if c in df.columns]
    sentiment_cols = [c for c in sentiment_cols if c in df.columns]
    
    required = price_cols + sentiment_cols + ["target_direction"]
    df_clean = df[required].dropna().copy()
    
    print(f"\nAfter dropna: {len(df_clean)} samples")
    print(f"Price features: {len(price_cols)}, Sentiment features: {len(sentiment_cols)}")
    
    # Build sequences
    Xp_list, Xs_list, y_list = [], [], []
    for i in range(args.sequence_length, len(df_clean)):
        Xp_list.append(df_clean[price_cols].iloc[i - args.sequence_length : i].values)
        Xs_list.append(df_clean[sentiment_cols].iloc[i].values)
        y_list.append(df_clean["target_direction"].iloc[i])
    
    Xp = np.array(Xp_list)
    Xs = np.array(Xs_list)
    y = np.array(y_list).reshape(-1, 1)
    
    print(f"Sequences: {len(Xp)}")
    
    # Temporal split
    idx_train, idx_val, idx_test = temporal_split(len(Xp))
    Xp_train, Xp_val, Xp_test = Xp[idx_train], Xp[idx_val], Xp[idx_test]
    Xs_train, Xs_val, Xs_test = Xs[idx_train], Xs[idx_val], Xs[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    
    # Scale
    sp = StandardScaler()
    ss = StandardScaler()
    n_steps, n_feat = Xp_train.shape[1], Xp_train.shape[2]
    
    Xp_train = sp.fit_transform(Xp_train.reshape(-1, n_feat)).reshape(len(Xp_train), n_steps, n_feat)
    Xp_val = sp.transform(Xp_val.reshape(-1, n_feat)).reshape(len(Xp_val), n_steps, n_feat)
    Xp_test = sp.transform(Xp_test.reshape(-1, n_feat)).reshape(len(Xp_test), n_steps, n_feat)
    
    Xs_train = ss.fit_transform(Xs_train)
    Xs_val = ss.transform(Xs_val)
    Xs_test = ss.transform(Xs_test)
    
    print("\n" + "="*60)
    print("ABLATION STUDY: Sentiment Feature Impact")
    print("="*60)
    
    # Experiment 1: Baseline (price only)
    print("\n[1/2] Training BASELINE (price + technical indicators only)...")
    baseline_metrics = train_baseline(
        Xp_train, y_train, Xp_val, y_val, Xp_test, y_test,
        n_feat=n_feat, epochs=args.epochs, device=device
    )
    print(f"Baseline: Accuracy={baseline_metrics['Accuracy']:.4f}, F1={baseline_metrics['F1']:.4f}, ROC-AUC={baseline_metrics['ROC_AUC']:.4f}")
    
    # Experiment 2: Combined (price + sentiment)
    print("\n[2/2] Training COMBINED (price + sentiment)...")
    combined_metrics = train_combined(
        Xp_train, Xs_train, y_train, Xp_val, Xs_val, y_val, Xp_test, Xs_test, y_test,
        n_feat=n_feat, n_sent=Xs_train.shape[1], epochs=args.epochs, device=device
    )
    print(f"Combined: Accuracy={combined_metrics['Accuracy']:.4f}, F1={combined_metrics['F1']:.4f}, ROC-AUC={combined_metrics['ROC_AUC']:.4f}")
    
    # Calculate improvement
    acc_diff = combined_metrics['Accuracy'] - baseline_metrics['Accuracy']
    f1_diff = combined_metrics['F1'] - baseline_metrics['F1']
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Sentiment impact on Accuracy: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
    print(f"Sentiment impact on F1:       {f1_diff:+.4f} ({f1_diff*100:+.2f}%)")
    
    # Save results
    results = pd.DataFrame([
        {"Experiment": "Baseline (price only)", **baseline_metrics},
        {"Experiment": "Combined (price + sentiment)", **combined_metrics},
        {
            "Experiment": "Delta (sentiment contribution)",
            "Accuracy": acc_diff,
            "Precision": combined_metrics['Precision'] - baseline_metrics['Precision'],
            "Recall": combined_metrics['Recall'] - baseline_metrics['Recall'],
            "F1": f1_diff,
            "ROC_AUC": combined_metrics['ROC_AUC'] - baseline_metrics['ROC_AUC'],
            "Threshold": 0.0,
        }
    ])
    
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results.to_csv(args.out_csv, index=False)
    
    print(f"\nResults saved: {args.out_csv}")
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()

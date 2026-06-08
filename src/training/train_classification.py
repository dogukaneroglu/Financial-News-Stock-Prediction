"""
Classification training pipeline for stock direction prediction.
Trains baseline (price-only) and combined (price+sentiment) models with BCE loss.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMModel
from models.combined_model import CombinedModel


class BaselineDirectionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CombinedDirectionDataset(Dataset):
    def __init__(self, X_price: np.ndarray, X_sent: np.ndarray, y: np.ndarray):
        self.X_price = torch.FloatTensor(X_price)
        self.X_sent = torch.FloatTensor(X_sent)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X_price)

    def __getitem__(self, idx):
        return self.X_price[idx], self.X_sent[idx], self.y[idx]


def make_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    df_clean = df[feature_columns + [target_column]].dropna().copy()
    X, y = [], []
    for i in range(sequence_length, len(df_clean)):
        X.append(df_clean[feature_columns].iloc[i - sequence_length : i].values)
        y.append(df_clean[target_column].iloc[i])
    return np.array(X), np.array(y).reshape(-1, 1)


def temporal_split_indices(n: int, test_size: float = 0.15, val_size: float = 0.15):
    indices = np.arange(n)
    idx_temp, idx_test = train_test_split(indices, test_size=test_size, shuffle=False)
    val_adj = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_adj, shuffle=False)
    return idx_train, idx_val, idx_test


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.flatten().astype(int)
    y_prob = y_prob.flatten()
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1") -> float:
    y_true = y_true.flatten().astype(int)
    y_prob = y_prob.flatten()
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.35, 0.66, 0.01):
        y_pred = (y_prob >= t).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def train_baseline_classification(
    data_path: str,
    model_dir: str,
    sequence_length: int,
    epochs: int,
    learning_rate: float,
    dropout: float,
    threshold_objective: str,
    hidden_size: int,
):
    print("\n" + "=" * 80)
    print("TRAINING BASELINE DIRECTION CLASSIFIER")
    print("=" * 80)

    df = pd.read_csv(data_path, parse_dates=["Date"])
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_7",
        "MA_14",
        "MA_30",
        "EMA_12",
        "EMA_26",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_diff",
        "BB_high",
        "BB_low",
        "BB_mid",
        "BB_width",
        "Volume_MA_7",
        "Volume_ratio",
        "ATR",
        "Price_change",
        "HL_range",
        "HL_pct",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X, y = make_sequences(df, feature_cols, "target_direction", sequence_length)
    idx_train, idx_val, idx_test = temporal_split_indices(len(X))
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    scaler = StandardScaler()
    n_steps, n_feat = X_train.shape[1], X_train.shape[2]
    X_train_2d = X_train.reshape(-1, n_feat)
    X_val_2d = X_val.reshape(-1, n_feat)
    X_test_2d = X_test.reshape(-1, n_feat)
    X_train = scaler.fit_transform(X_train_2d).reshape(len(X_train), n_steps, n_feat)
    X_val = scaler.transform(X_val_2d).reshape(len(X_val), n_steps, n_feat)
    X_test = scaler.transform(X_test_2d).reshape(len(X_test), n_steps, n_feat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=n_feat, hidden_size=hidden_size, num_layers=2, dropout=dropout).to(device)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([float(neg / max(pos, 1))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(BaselineDirectionDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(BaselineDirectionDataset(X_val, y_val), batch_size=32, shuffle=False)

    best_val = float("inf")
    patience = 15
    wait = 0
    best_state = None
    print(f"Training baseline classifier for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).squeeze(1)
            optimizer.zero_grad()
            logits = model(bx).squeeze(1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device).squeeze(1)
                logits = model(bx).squeeze(1)
                val_loss += criterion(logits, by).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.FloatTensor(X_val).to(device)).squeeze(1)).cpu().numpy()
        probs = torch.sigmoid(model(torch.FloatTensor(X_test).to(device)).squeeze(1)).cpu().numpy()

    best_threshold = find_best_threshold(y_val, val_probs, objective=threshold_objective)
    metrics = compute_classification_metrics(y_test, probs, threshold=best_threshold)
    metrics["Best_Threshold"] = best_threshold

    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_X": scaler,
            "config": {"sequence_length": sequence_length, "dropout": dropout, "task": "classification"},
        },
        os.path.join(model_dir, "baseline_direction_classifier.pth"),
    )
    np.savez(os.path.join(model_dir, "baseline_direction_test.npz"), y_test=y_test, y_prob=probs)
    return metrics


def train_combined_classification(
    data_path: str,
    model_dir: str,
    sequence_length: int,
    epochs: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    threshold_objective: str,
    hidden_size: int,
    sentiment_mode: str,
):
    print("\n" + "=" * 80)
    print("TRAINING COMBINED DIRECTION CLASSIFIER")
    print("=" * 80)

    df = pd.read_csv(data_path, parse_dates=["Date"])
    price_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_7",
        "MA_14",
        "MA_30",
        "EMA_12",
        "EMA_26",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_diff",
        "BB_high",
        "BB_low",
        "BB_mid",
        "BB_width",
        "Volume_MA_7",
        "Volume_ratio",
        "ATR",
        "Price_change",
        "HL_range",
        "HL_pct",
    ]
    price_cols = [c for c in price_cols if c in df.columns]

    if sentiment_mode == "full":
        sentiment_cols = [
            "sentiment_mean",
            "sentiment_std",
            "sentiment_min",
            "sentiment_max",
            "positive_mean",
            "negative_mean",
            "neutral_mean",
            "news_count",
            "sentiment_mean_lag_1",
            "sentiment_std_lag_1",
            "sentiment_min_lag_1",
            "sentiment_max_lag_1",
            "positive_mean_lag_1",
            "negative_mean_lag_1",
            "neutral_mean_lag_1",
            "news_count_lag_1",
        ]
    else:
        # Reduced-noise sentiment set: keep strongest signals only.
        sentiment_cols = [
            "sentiment_mean",
            "news_count",
            "positive_mean",
            "negative_mean",
            "sentiment_mean_lag_1",
            "news_count_lag_1",
            "positive_mean_lag_1",
            "negative_mean_lag_1",
        ]
    sentiment_cols = [c for c in sentiment_cols if c in df.columns]

    all_cols = price_cols + sentiment_cols + ["target_direction"]
    df_clean = df[all_cols].dropna().copy()
    X_price, X_sent, y = [], [], []
    for i in range(sequence_length, len(df_clean)):
        X_price.append(df_clean[price_cols].iloc[i - sequence_length : i].values)
        X_sent.append(df_clean[sentiment_cols].iloc[i].values)
        y.append(df_clean["target_direction"].iloc[i])
    X_price = np.array(X_price)
    X_sent = np.array(X_sent)
    y = np.array(y).reshape(-1, 1)

    idx_train, idx_val, idx_test = temporal_split_indices(len(X_price))
    Xp_train, Xp_val, Xp_test = X_price[idx_train], X_price[idx_val], X_price[idx_test]
    Xs_train, Xs_val, Xs_test = X_sent[idx_train], X_sent[idx_val], X_sent[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    sp = StandardScaler()
    ss = StandardScaler()
    n_steps, n_feat = Xp_train.shape[1], Xp_train.shape[2]
    Xp_train = sp.fit_transform(Xp_train.reshape(-1, n_feat)).reshape(len(Xp_train), n_steps, n_feat)
    Xp_val = sp.transform(Xp_val.reshape(-1, n_feat)).reshape(len(Xp_val), n_steps, n_feat)
    Xp_test = sp.transform(Xp_test.reshape(-1, n_feat)).reshape(len(Xp_test), n_steps, n_feat)
    Xs_train = ss.fit_transform(Xs_train)
    Xs_val = ss.transform(Xs_val)
    Xs_test = ss.transform(Xs_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel(
        price_input_size=n_feat,
        sentiment_input_size=Xs_train.shape[1],
        hidden_size=hidden_size,
        num_layers=2,
        dropout=dropout,
    ).to(device)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([float(neg / max(pos, 1))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(CombinedDirectionDataset(Xp_train, Xs_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(CombinedDirectionDataset(Xp_val, Xs_val, y_val), batch_size=32, shuffle=False)

    best_val = float("inf")
    patience = 15
    wait = 0
    best_state = None
    print(f"Training combined classifier for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bxp, bxs, by in train_loader:
            bxp, bxs, by = bxp.to(device), bxs.to(device), by.to(device).squeeze(1)
            optimizer.zero_grad()
            logits = model(bxp, bxs).squeeze(1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bxp, bxs, by in val_loader:
                bxp, bxs, by = bxp.to(device), bxs.to(device), by.to(device).squeeze(1)
                logits = model(bxp, bxs).squeeze(1)
                val_loss += criterion(logits, by).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = (
            torch.sigmoid(
                model(
                    torch.FloatTensor(Xp_val).to(device),
                    torch.FloatTensor(Xs_val).to(device),
                ).squeeze(1)
            )
            .cpu()
            .numpy()
        )
        probs = (
            torch.sigmoid(
                model(
                    torch.FloatTensor(Xp_test).to(device),
                    torch.FloatTensor(Xs_test).to(device),
                ).squeeze(1)
            )
            .cpu()
            .numpy()
        )

    best_threshold = find_best_threshold(y_val, val_probs, objective=threshold_objective)
    metrics = compute_classification_metrics(y_test, probs, threshold=best_threshold)
    metrics["Best_Threshold"] = best_threshold

    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_price": sp,
            "scaler_sentiment": ss,
            "config": {
                "sequence_length": sequence_length,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "task": "classification",
            },
        },
        os.path.join(model_dir, "combined_direction_classifier.pth"),
    )
    np.savez(os.path.join(model_dir, "combined_direction_test.npz"), y_test=y_test, y_prob=probs)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train stock direction classifiers")
    parser.add_argument("--baseline-data", type=str, default="data/processed/stock_features.csv")
    parser.add_argument("--combined-data", type=str, default="data/processed/stock_features_with_sentiment.csv")
    parser.add_argument("--model-dir", type=str, default="data/models/classification")
    parser.add_argument("--model-type", type=str, choices=["baseline", "combined", "both"], default="both")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threshold-objective", type=str, choices=["f1", "accuracy"], default="f1")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--sentiment-mode", type=str, choices=["reduced", "full"], default="reduced")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("DIRECTION CLASSIFICATION TRAINING PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"Config: model_type={args.model_type}, epochs={args.epochs}, "
        f"dropout={args.dropout}, wd={args.weight_decay}, threshold_objective={args.threshold_objective}, "
        f"hidden_size={args.hidden_size}, sentiment_mode={args.sentiment_mode}"
    )

    summary = []
    if args.model_type in ["baseline", "both"]:
        m = train_baseline_classification(
            data_path=args.baseline_data,
            model_dir=args.model_dir,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            threshold_objective=args.threshold_objective,
            hidden_size=args.hidden_size,
        )
        summary.append({"Model": "BaselineDirection", **m})

    if args.model_type in ["combined", "both"]:
        m = train_combined_classification(
            data_path=args.combined_data,
            model_dir=args.model_dir,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            threshold_objective=args.threshold_objective,
            hidden_size=args.hidden_size,
            sentiment_mode=args.sentiment_mode,
        )
        summary.append({"Model": "CombinedDirection", **m})

    if summary:
        df = pd.DataFrame(summary)
        print("\n" + "=" * 80)
        print("CLASSIFICATION METRICS")
        print("=" * 80)
        print(df.to_string(index=False))
        out = os.path.join(args.model_dir, "classification_metrics.csv")
        os.makedirs(args.model_dir, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()

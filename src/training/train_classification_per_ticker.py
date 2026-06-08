"""
Per-ticker direction classification training.
Trains independent classifiers for each ticker to reduce cross-asset noise.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.combined_model import CombinedModel

PRICE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume", "MA_7", "MA_14", "MA_30", "EMA_12", "EMA_26",
    "RSI", "MACD", "MACD_signal", "MACD_diff", "BB_high", "BB_low", "BB_mid", "BB_width",
    "Volume_MA_7", "Volume_ratio", "ATR", "Price_change", "HL_range", "HL_pct",
]


def sentiment_cols_for_mode(sentiment_mode: str) -> List[str]:
    if sentiment_mode == "full":
        return [
            "sentiment_mean", "sentiment_std", "sentiment_min", "sentiment_max",
            "positive_mean", "negative_mean", "neutral_mean", "news_count",
            "sentiment_mean_lag_1", "sentiment_std_lag_1", "sentiment_min_lag_1", "sentiment_max_lag_1",
            "positive_mean_lag_1", "negative_mean_lag_1", "neutral_mean_lag_1", "news_count_lag_1",
        ]
    if sentiment_mode == "minimal":
        return [
            "sentiment_mean", "news_count",
            "sentiment_mean_lag_1", "news_count_lag_1",
        ]
    return [
        "sentiment_mean", "news_count", "positive_mean", "negative_mean",
        "sentiment_mean_lag_1", "news_count_lag_1", "positive_mean_lag_1", "negative_mean_lag_1",
    ]


def _union_sentiment_columns() -> List[str]:
    order: List[str] = []
    for m in ("minimal", "reduced", "full"):
        for c in sentiment_cols_for_mode(m):
            if c not in order:
                order.append(c)
    return order


class CombinedDirectionDataset(Dataset):
    def __init__(self, X_price: np.ndarray, X_sent: np.ndarray, y: np.ndarray):
        self.X_price = torch.FloatTensor(X_price)
        self.X_sent = torch.FloatTensor(X_sent)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X_price)

    def __getitem__(self, idx):
        return self.X_price[idx], self.X_sent[idx], self.y[idx]


def temporal_split_indices(n: int, test_size: float = 0.15, val_size: float = 0.15):
    indices = np.arange(n)
    idx_temp, idx_test = train_test_split(indices, test_size=test_size, shuffle=False)
    val_adj = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_adj, shuffle=False)
    return idx_train, idx_val, idx_test


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str = "accuracy") -> float:
    y_true = y_true.flatten().astype(int)
    y_prob = y_prob.flatten()
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.35, 0.66, 0.01):
        y_pred = (y_prob >= t).astype(int)
        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
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
        "Best_Threshold": float(threshold),
    }


def _fit_scale_price(
    Xp_train: np.ndarray, Xp_val: np.ndarray, Xp_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    n_steps, n_feat = Xp_train.shape[1], Xp_train.shape[2]
    sp = StandardScaler()
    Xp_train_s = sp.fit_transform(Xp_train.reshape(-1, n_feat)).reshape(len(Xp_train), n_steps, n_feat)
    Xp_val_s = sp.transform(Xp_val.reshape(-1, n_feat)).reshape(len(Xp_val), n_steps, n_feat)
    Xp_test_s = sp.transform(Xp_test.reshape(-1, n_feat)).reshape(len(Xp_test), n_steps, n_feat)
    return Xp_train_s, Xp_val_s, Xp_test_s, n_steps, n_feat


def _fit_scale_sentiment(
    Xs_train: np.ndarray, Xs_val: np.ndarray, Xs_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ss = StandardScaler()
    return ss.fit_transform(Xs_train), ss.transform(Xs_val), ss.transform(Xs_test)


def train_direction_model(
    Xp_train: np.ndarray,
    Xs_train: np.ndarray,
    y_train: np.ndarray,
    Xp_val: np.ndarray,
    Xs_val: np.ndarray,
    y_val: np.ndarray,
    Xp_test: np.ndarray,
    Xs_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    hidden_size: int,
    threshold_objective: str,
    device: torch.device,
    n_feat: int,
    patience: int = 12,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    model = CombinedModel(
        price_input_size=n_feat,
        sentiment_input_size=Xs_train.shape[1],
        hidden_size=hidden_size,
        num_layers=2,
        dropout=dropout,
    ).to(device)

    pos = float(y_train.sum())
    neg = len(y_train) - pos
    pos_weight = torch.tensor([float(neg / max(pos, 1.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(CombinedDirectionDataset(Xp_train, Xs_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(CombinedDirectionDataset(Xp_val, Xs_val, y_val), batch_size=32, shuffle=False)

    best_val, wait = float("inf"), 0
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

    thr = find_best_threshold(y_val, val_prob, objective=threshold_objective)
    val_metrics = compute_metrics(y_val, val_prob, threshold=thr)
    test_metrics = compute_metrics(y_test, test_prob, threshold=thr)
    return val_metrics, test_metrics, thr


def _build_arrays_for_ticker(
    ticker_df: pd.DataFrame, sequence_length: int, sentiment_cols: List[str]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    price_cols = [c for c in PRICE_COLUMNS if c in ticker_df.columns]
    sentiment_cols = [c for c in sentiment_cols if c in ticker_df.columns]
    if len(sentiment_cols) == 0:
        return None
    required_cols = price_cols + sentiment_cols + ["target_direction"]
    df = ticker_df[required_cols].dropna().copy()
    if len(df) <= sequence_length + 50:
        return None
    Xp, Xs, y = [], [], []
    for i in range(sequence_length, len(df)):
        Xp.append(df[price_cols].iloc[i - sequence_length : i].values)
        Xs.append(df[sentiment_cols].iloc[i].values)
        y.append(df["target_direction"].iloc[i])
    return np.array(Xp), np.array(Xs), np.array(y).reshape(-1, 1)


def _prepare_auto_arrays(
    ticker_df: pd.DataFrame, sequence_length: int
) -> Optional[Dict]:
    price_cols = [c for c in PRICE_COLUMNS if c in ticker_df.columns]
    union = [c for c in _union_sentiment_columns() if c in ticker_df.columns]
    required_cols = price_cols + union + ["target_direction"]
    df = ticker_df[required_cols].dropna().copy()
    if len(df) <= sequence_length + 50:
        return None

    modes: List[str] = []
    sentiment_cols_ok: Dict[str, List[str]] = {}
    for mode in ("minimal", "reduced", "full"):
        need = sentiment_cols_for_mode(mode)
        if all(c in df.columns for c in need):
            modes.append(mode)
            sentiment_cols_ok[mode] = need
    if not modes:
        return None

    Xp_list, y_list = [], []
    Xs_lists: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    for i in range(sequence_length, len(df)):
        Xp_list.append(df[price_cols].iloc[i - sequence_length : i].values)
        y_list.append(df["target_direction"].iloc[i])
        for m in modes:
            sc = sentiment_cols_ok[m]
            Xs_lists[m].append(df[sc].iloc[i].values)
    Xp = np.array(Xp_list)
    y = np.array(y_list).reshape(-1, 1)
    Xs_by_mode = {m: np.array(Xs_lists[m]) for m in modes}
    return {
        "Xp": Xp,
        "y": y,
        "Xs_by_mode": Xs_by_mode,
        "modes": modes,
        "n_price_feat": Xp.shape[2],
    }


def _pick_mode_by_validation(
    val_scores: Dict[str, Dict[str, float]], modes_tried: List[str]
) -> str:
    def key(m: str) -> Tuple[float, float, float]:
        v = val_scores[m]
        return (v["Accuracy"], v["F1"], v["ROC_AUC"])

    return max(modes_tried, key=key)


def train_one_ticker(
    ticker_df: pd.DataFrame,
    sequence_length: int,
    epochs: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    hidden_size: int,
    threshold_objective: str,
    device: torch.device,
    sentiment_mode: str,
):
    built = _build_arrays_for_ticker(ticker_df, sequence_length, sentiment_cols_for_mode(sentiment_mode))
    if built is None:
        return None
    Xp, Xs, y = built

    idx_train, idx_val, idx_test = temporal_split_indices(len(Xp))
    Xp_train, Xp_val, Xp_test = Xp[idx_train], Xp[idx_val], Xp[idx_test]
    Xs_train, Xs_val, Xs_test = Xs[idx_train], Xs[idx_val], Xs[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    Xp_train, Xp_val, Xp_test, _, n_feat = _fit_scale_price(Xp_train, Xp_val, Xp_test)
    Xs_train, Xs_val, Xs_test = _fit_scale_sentiment(Xs_train, Xs_val, Xs_test)

    _, test_metrics, _ = train_direction_model(
        Xp_train,
        Xs_train,
        y_train,
        Xp_val,
        Xs_val,
        y_val,
        Xp_test,
        Xs_test,
        y_test,
        epochs=epochs,
        learning_rate=learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
        hidden_size=hidden_size,
        threshold_objective=threshold_objective,
        device=device,
        n_feat=n_feat,
    )
    return test_metrics


def train_one_ticker_auto_sentiment(
    ticker_df: pd.DataFrame,
    sequence_length: int,
    epochs: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    hidden_size: int,
    threshold_objective: str,
    device: torch.device,
) -> Optional[Tuple[Dict[str, float], str, Dict[str, Any]]]:
    """
    Train each sentiment mode with the SAME temporal split and overlapping rows.
    Pick the mode maximizing validation Accuracy (tie-break F1, then ROC-AUC).
    Report test metrics using the threshold tuned on validation for that mode.
    """
    prep = _prepare_auto_arrays(ticker_df, sequence_length)
    if prep is None:
        return None

    Xp = prep["Xp"]
    y = prep["y"]
    Xs_by_mode = prep["Xs_by_mode"]
    modes = prep["modes"]
    idx_train, idx_val, idx_test = temporal_split_indices(len(Xp))
    Xp_train, Xp_val, Xp_test = Xp[idx_train], Xp[idx_val], Xp[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    Xp_train, Xp_val, Xp_test, _, n_feat = _fit_scale_price(Xp_train, Xp_val, Xp_test)

    val_scores: Dict[str, Dict[str, float]] = {}
    test_by_mode: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        Xs = Xs_by_mode[mode]
        Xs_train, Xs_val, Xs_test = Xs[idx_train], Xs[idx_val], Xs[idx_test]
        Xs_train, Xs_val, Xs_test = _fit_scale_sentiment(Xs_train, Xs_val, Xs_test)
        val_m, test_m, _ = train_direction_model(
            Xp_train,
            Xs_train,
            y_train,
            Xp_val,
            Xs_val,
            y_val,
            Xp_test,
            Xs_test,
            y_test,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout=dropout,
            weight_decay=weight_decay,
            hidden_size=hidden_size,
            threshold_objective=threshold_objective,
            device=device,
            n_feat=n_feat,
        )
        val_scores[mode] = val_m
        test_by_mode[mode] = test_m

    best = _pick_mode_by_validation(val_scores, modes)
    out_metrics = dict(test_by_mode[best])
    # Keep downstream CSV schema consistent; annotate selection in extra keys via caller.
    meta: Dict[str, Any] = {
        "selected_mode": best,
        "val_accuracy_selected": val_scores[best]["Accuracy"],
        "val_f1_selected": val_scores[best]["F1"],
        "val_by_mode": val_scores,
        "test_by_mode": test_by_mode,
    }
    return out_metrics, best, meta


def main():
    parser = argparse.ArgumentParser(description="Per-ticker direction classification")
    parser.add_argument("--data", type=str, default="data/processed/stock_features_with_sentiment.csv")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--threshold-objective", type=str, choices=["accuracy", "f1"], default="accuracy")
    parser.add_argument("--sentiment-mode", type=str, choices=["minimal", "reduced", "full"], default="reduced")
    parser.add_argument(
        "--auto-sentiment",
        action="store_true",
        help="Per ticker: train minimal/reduced/full on the same dates; pick best by validation accuracy (tie-break F1, ROC-AUC). Writes Selected_Sentiment_Mode and validation scores used for selection.",
    )
    parser.add_argument("--out-csv", type=str, default="data/models/classification/per_ticker_metrics.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["Date"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.auto_sentiment:
        print("Sentiment: AUTO (minimal / reduced / full via validation)")
    else:
        print(f"Sentiment mode: {args.sentiment_mode}")
    print(f"Running per-ticker classification on {sorted(df['Ticker'].unique().tolist())}")

    rows = []
    for ticker in sorted(df["Ticker"].unique().tolist()):
        print(f"\nTraining ticker: {ticker}")
        tdf = df[df["Ticker"] == ticker].sort_values("Date").copy()
        if args.auto_sentiment:
            result = train_one_ticker_auto_sentiment(
                ticker_df=tdf,
                sequence_length=args.sequence_length,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                hidden_size=args.hidden_size,
                threshold_objective=args.threshold_objective,
                device=device,
            )
            if result is None:
                print(f"Skipped {ticker}: not enough usable samples")
                continue
            metrics, sel_mode, meta = result
            val_line = ", ".join(
                f"{m}: val_acc={meta['val_by_mode'][m]['Accuracy']:.4f}" for m in meta["val_by_mode"]
            )
            print(
                f"{ticker} selected={sel_mode} ({val_line}) | "
                f"test Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}, ROC-AUC: {metrics['ROC_AUC']:.4f}"
            )
            rows.append(
                {
                    "Ticker": ticker,
                    "Selected_Sentiment_Mode": sel_mode,
                    "Val_Accuracy_At_Select": meta["val_accuracy_selected"],
                    "Val_F1_At_Select": meta["val_f1_selected"],
                    **metrics,
                }
            )
        else:
            metrics = train_one_ticker(
                ticker_df=tdf,
                sequence_length=args.sequence_length,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                hidden_size=args.hidden_size,
                threshold_objective=args.threshold_objective,
                device=device,
                sentiment_mode=args.sentiment_mode,
            )
            if metrics is None:
                print(f"Skipped {ticker}: not enough usable samples")
                continue
            print(f"{ticker} Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}, ROC-AUC: {metrics['ROC_AUC']:.4f}")
            rows.append({"Ticker": ticker, **metrics})

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        avg_row: Dict[str, Any] = {
            "Ticker": "AVERAGE",
            "Accuracy": out_df["Accuracy"].mean(),
            "Precision": out_df["Precision"].mean(),
            "Recall": out_df["Recall"].mean(),
            "F1": out_df["F1"].mean(),
            "ROC_AUC": out_df["ROC_AUC"].mean(),
            "Best_Threshold": out_df["Best_Threshold"].mean(),
        }
        if args.auto_sentiment and "Val_Accuracy_At_Select" in out_df.columns:
            avg_row["Selected_Sentiment_Mode"] = "mixed_per_ticker"
            avg_row["Val_Accuracy_At_Select"] = out_df["Val_Accuracy_At_Select"].mean()
            avg_row["Val_F1_At_Select"] = out_df["Val_F1_At_Select"].mean()
        out_df.loc[len(out_df)] = avg_row

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("\nPer-ticker metrics:")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()

"""
Calculate naive baseline performance for comparison.
Naive strategies: always predict majority class, random guess, always up/down.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the processed data
df = pd.read_csv("data/processed/stock_features_with_sentiment.csv", parse_dates=["Date"])

# Same preprocessing as training
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

# Build sequences (same as ablation study)
sequence_length = 60
y_list = []
for i in range(sequence_length, len(df_clean)):
    y_list.append(df_clean["target_direction"].iloc[i])

y = np.array(y_list).reshape(-1, 1)

# Temporal split (same as training)
from sklearn.model_selection import train_test_split

def temporal_split(n, test_size=0.15, val_size=0.15):
    indices = np.arange(n)
    idx_temp, idx_test = train_test_split(indices, test_size=test_size, shuffle=False)
    val_adj = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_adj, shuffle=False)
    return idx_train, idx_val, idx_test

idx_train, idx_val, idx_test = temporal_split(len(y))
y_train = y[idx_train].flatten()
y_test = y[idx_test].flatten()

print("="*60)
print("NAIVE BASELINE EVALUATION")
print("="*60)
print(f"\nTrain set size: {len(y_train)}")
print(f"Test set size: {len(y_test)}")
print(f"\nTrain class distribution:")
print(f"  Up (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
print(f"  Down (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
print(f"\nTest class distribution:")
print(f"  Up (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.2f}%)")
print(f"  Down (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.2f}%)")

# Strategy 1: Always predict majority class (most frequent in training)
majority_class = 1 if (y_train == 1).sum() > (y_train == 0).sum() else 0
y_pred_majority = np.full_like(y_test, majority_class)

acc_majority = accuracy_score(y_test, y_pred_majority)
prec_majority = precision_score(y_test, y_pred_majority, zero_division=0)
rec_majority = recall_score(y_test, y_pred_majority, zero_division=0)
f1_majority = f1_score(y_test, y_pred_majority, zero_division=0)

print("\n" + "="*60)
print("STRATEGY 1: Always Predict Majority Class")
print("="*60)
print(f"Majority class (from training): {majority_class} ({'Up' if majority_class == 1 else 'Down'})")
print(f"Accuracy:  {acc_majority:.4f} ({acc_majority*100:.2f}%)")
print(f"Precision: {prec_majority:.4f}")
print(f"Recall:    {rec_majority:.4f}")
print(f"F1:        {f1_majority:.4f}")

# Strategy 2: Always predict UP
y_pred_up = np.ones_like(y_test)
acc_up = accuracy_score(y_test, y_pred_up)
prec_up = precision_score(y_test, y_pred_up, zero_division=0)
rec_up = recall_score(y_test, y_pred_up, zero_division=0)
f1_up = f1_score(y_test, y_pred_up, zero_division=0)

print("\n" + "="*60)
print("STRATEGY 2: Always Predict UP")
print("="*60)
print(f"Accuracy:  {acc_up:.4f} ({acc_up*100:.2f}%)")
print(f"Precision: {prec_up:.4f}")
print(f"Recall:    {rec_up:.4f}")
print(f"F1:        {f1_up:.4f}")

# Strategy 3: Random guess (50-50)
np.random.seed(42)
y_pred_random = np.random.randint(0, 2, size=len(y_test))
acc_random = accuracy_score(y_test, y_pred_random)
prec_random = precision_score(y_test, y_pred_random, zero_division=0)
rec_random = recall_score(y_test, y_pred_random, zero_division=0)
f1_random = f1_score(y_test, y_pred_random, zero_division=0)

print("\n" + "="*60)
print("STRATEGY 3: Random Guess (50-50)")
print("="*60)
print(f"Accuracy:  {acc_random:.4f} ({acc_random*100:.2f}%)")
print(f"Precision: {prec_random:.4f}")
print(f"Recall:    {rec_random:.4f}")
print(f"F1:        {f1_random:.4f}")

# Load existing ablation results
ablation_df = pd.read_csv("data/models/classification/ablation_results.csv")

# Add naive baseline row
naive_row = pd.DataFrame([{
    "Experiment": "Naive (always majority class)",
    "Accuracy": acc_majority,
    "Precision": prec_majority,
    "Recall": rec_majority,
    "F1": f1_majority,
    "ROC_AUC": 0.5,  # Random performance for ROC-AUC
    "Threshold": 0.5,
}])

# Insert naive baseline at the beginning
ablation_df_new = pd.concat([naive_row, ablation_df], ignore_index=True)

# Save updated results
ablation_df_new.to_csv("data/models/classification/ablation_with_baseline.csv", index=False)

print("\n" + "="*60)
print("COMPARISON TABLE (including Naive Baseline)")
print("="*60)
print(ablation_df_new.to_string(index=False))

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print(f"Naive baseline accuracy: {acc_majority*100:.2f}%")
print(f"Our Baseline model: {ablation_df.iloc[0]['Accuracy']*100:.2f}%")
print(f"Our Combined model: {ablation_df.iloc[1]['Accuracy']*100:.2f}%")
print(f"\nImprovement over naive:")
print(f"  Baseline: +{(ablation_df.iloc[0]['Accuracy'] - acc_majority)*100:.2f} percentage points")
print(f"  Combined: +{(ablation_df.iloc[1]['Accuracy'] - acc_majority)*100:.2f} percentage points")

print(f"\nSaved: data/models/classification/ablation_with_baseline.csv")

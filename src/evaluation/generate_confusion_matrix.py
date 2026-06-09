"""
Generate confusion matrices for baseline and combined models.
Shows where the models make correct/incorrect predictions.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import LSTMModel
from models.combined_model import CombinedModel

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("="*60)
print("CONFUSION MATRIX GENERATION")
print("="*60)

# Load data
df = pd.read_csv("data/processed/stock_features_with_sentiment.csv", parse_dates=["Date"])

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

print(f"\nData loaded: {len(df_clean)} samples")

# Build sequences
sequence_length = 60
Xp_list, Xs_list, y_list = [], [], []
for i in range(sequence_length, len(df_clean)):
    Xp_list.append(df_clean[price_cols].iloc[i - sequence_length : i].values)
    Xs_list.append(df_clean[sentiment_cols].iloc[i].values)
    y_list.append(df_clean["target_direction"].iloc[i])

Xp = np.array(Xp_list)
Xs = np.array(Xs_list)
y = np.array(y_list).reshape(-1, 1)

# Temporal split
def temporal_split(n, test_size=0.15, val_size=0.15):
    indices = np.arange(n)
    idx_temp, idx_test = train_test_split(indices, test_size=test_size, shuffle=False)
    val_adj = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_adj, shuffle=False)
    return idx_train, idx_val, idx_test

idx_train, idx_val, idx_test = temporal_split(len(Xp))
Xp_test = Xp[idx_test]
Xs_test = Xs[idx_test]
y_test = y[idx_test]

print(f"Test set size: {len(y_test)}")

# Scale
sp = StandardScaler()
ss = StandardScaler()
Xp_train = Xp[idx_train]
Xs_train = Xs[idx_train]

n_steps, n_feat = Xp_train.shape[1], Xp_train.shape[2]
sp.fit(Xp_train.reshape(-1, n_feat))
ss.fit(Xs_train)

Xp_test_scaled = sp.transform(Xp_test.reshape(-1, n_feat)).reshape(len(Xp_test), n_steps, n_feat)
Xs_test_scaled = ss.transform(Xs_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

output_dir = "data/evaluation/confusion"
os.makedirs(output_dir, exist_ok=True)

# Load models
baseline_path = "data/models/classification/baseline_direction_classifier.pth"
combined_path = "data/models/classification/combined_direction_classifier.pth"

if not os.path.exists(baseline_path):
    print(f"\nWARNING: Baseline model not found at {baseline_path}")
    print("Skipping baseline confusion matrix.")
    baseline_exists = False
else:
    baseline_exists = True

if not os.path.exists(combined_path):
    print(f"\nWARNING: Combined model not found at {combined_path}")
    print("Skipping combined confusion matrix.")
    combined_exists = False
else:
    combined_exists = True

results = {}

# Baseline model
if baseline_exists:
    print("\n" + "="*60)
    print("BASELINE MODEL (Price Only)")
    print("="*60)
    
    baseline_model = LSTMModel(input_size=n_feat, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    checkpoint = torch.load(baseline_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        baseline_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        baseline_model.load_state_dict(checkpoint)
    baseline_model.eval()
    
    with torch.no_grad():
        baseline_prob = torch.sigmoid(
            baseline_model(torch.FloatTensor(Xp_test_scaled).to(device)).squeeze(1)
        ).cpu().numpy()
    
    # Use 0.5 threshold for confusion matrix
    baseline_pred = (baseline_prob >= 0.5).astype(int)
    y_test_flat = y_test.flatten().astype(int)
    
    cm_baseline = confusion_matrix(y_test_flat, baseline_pred)
    results['baseline'] = {'cm': cm_baseline, 'pred': baseline_pred, 'prob': baseline_prob}
    
    print("\nConfusion Matrix:")
    print(cm_baseline)
    print("\nClassification Report:")
    print(classification_report(y_test_flat, baseline_pred, target_names=['Down', 'Up'], zero_division=0))

# Combined model - skip if dimension mismatch
if combined_exists:
    print("\n" + "="*60)
    print("COMBINED MODEL (Price + Sentiment)")
    print("="*60)
    
    try:
        checkpoint = torch.load(combined_path, map_location=device)
        
        # Get config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            sentiment_size = config.get('sentiment_input_size', Xs_test_scaled.shape[1])
        else:
            sentiment_size = Xs_test_scaled.shape[1]
        
        combined_model = CombinedModel(
            price_input_size=n_feat, sentiment_input_size=sentiment_size,
            hidden_size=64, num_layers=2, dropout=0.2
        ).to(device)
        
        if 'model_state_dict' in checkpoint:
            combined_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            combined_model.load_state_dict(checkpoint)
        combined_model.eval()
        
        with torch.no_grad():
            # Adjust sentiment features if needed
            if sentiment_size != Xs_test_scaled.shape[1]:
                print(f"\nWARNING: Sentiment size mismatch. Model expects {sentiment_size}, got {Xs_test_scaled.shape[1]}")
                print("Skipping combined model confusion matrix.")
                combined_exists = False
            else:
                combined_prob = torch.sigmoid(
                    combined_model(
                        torch.FloatTensor(Xp_test_scaled).to(device),
                        torch.FloatTensor(Xs_test_scaled).to(device)
                    ).squeeze(1)
                ).cpu().numpy()
        
                combined_pred = (combined_prob >= 0.5).astype(int)
                cm_combined = confusion_matrix(y_test_flat, combined_pred)
                results['combined'] = {'cm': cm_combined, 'pred': combined_pred, 'prob': combined_prob}
                
                print("\nConfusion Matrix:")
                print(cm_combined)
                print("\nClassification Report:")
                print(classification_report(y_test_flat, combined_pred, target_names=['Down', 'Up'], zero_division=0))
    except Exception as e:
        print(f"\nERROR loading combined model: {e}")
        print("Skipping combined model confusion matrix.")
        combined_exists = False

# Visualization
print("\n" + "="*60)
print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
print("="*60)

def plot_confusion_matrix(cm, title, filename, labels=['Down (0)', 'Up (1)']):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=1, linecolor='gray')
    
    # Add percentages
    for i in range(2):
        for j in range(2):
            text = ax.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]*100:.1f}%)',
                          ha="center", va="center", color="red", fontsize=10)
    
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", bbox_inches='tight')
    print(f"Saved: {output_dir}/{filename}")
    plt.close()

# Plot individual confusion matrices
if baseline_exists:
    plot_confusion_matrix(
        results['baseline']['cm'],
        'Confusion Matrix: Baseline Model (Price Only)',
        'confusion_matrix_baseline.png'
    )

if combined_exists:
    plot_confusion_matrix(
        results['combined']['cm'],
        'Confusion Matrix: Combined Model (Price + Sentiment)',
        'confusion_matrix_combined.png'
    )

# Side-by-side comparison
if baseline_exists and combined_exists:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (name, data), title in zip(
        axes,
        [('baseline', results['baseline']), ('combined', results['combined'])],
        ['Baseline (Price Only)', 'Combined (Price + Sentiment)']
    ):
        cm = data['cm']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                    ax=ax, linewidths=1, linecolor='gray')
        
        # Add percentages
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)',
                       ha="center", va="center", color="red", fontsize=9)
        
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_comparison.png", bbox_inches='tight')
    print(f"Saved: {output_dir}/confusion_comparison.png")
    plt.close()

print("\n" + "="*60)
print("CONFUSION MATRIX GENERATION COMPLETE")
print("="*60)
print(f"\nOutput directory: {output_dir}/")
if baseline_exists:
    print("- confusion_matrix_baseline.png")
if combined_exists:
    print("- confusion_matrix_combined.png")
if baseline_exists and combined_exists:
    print("- confusion_comparison.png")

"""
Visualize ablation study results for thesis.
Creates publication-quality charts showing sentiment feature impact.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Load results
results_path = "data/models/classification/ablation_results.csv"
df = pd.read_csv(results_path)

# Filter delta row for separate handling
df_models = df[df['Experiment'] != 'Delta (sentiment contribution)'].copy()
df_delta = df[df['Experiment'] == 'Delta (sentiment contribution)'].iloc[0]

output_dir = "data/evaluation/ablation"
os.makedirs(output_dir, exist_ok=True)

print("Creating ablation study visualizations...")

# 1. Bar chart: Accuracy, F1, ROC-AUC comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'F1', 'ROC_AUC']
x = range(len(metrics))
width = 0.35

baseline_vals = [df_models.iloc[0][m] for m in metrics]
combined_vals = [df_models.iloc[1][m] for m in metrics]

bars1 = ax.bar([i - width/2 for i in x], baseline_vals, width, label='Baseline (price only)', color='#3498db', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], combined_vals, width, label='Combined (price + sentiment)', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Ablation Study: Baseline vs Combined Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'F1 Score', 'ROC-AUC'])
ax.legend()
ax.set_ylim([0.4, 0.7])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"{output_dir}/ablation_comparison.png", bbox_inches='tight')
print(f"Saved: {output_dir}/ablation_comparison.png")
plt.close()

# 2. Precision-Recall comparison
fig, ax = plt.subplots(figsize=(8, 6))
models = ['Baseline\n(price only)', 'Combined\n(price + sentiment)']
precision = [df_models.iloc[0]['Precision'], df_models.iloc[1]['Precision']]
recall = [df_models.iloc[0]['Recall'], df_models.iloc[1]['Recall']]

x = range(len(models))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], precision, width, label='Precision', color='#2ecc71', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], recall, width, label='Recall', color='#f39c12', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Precision-Recall Trade-off: Sentiment Impact')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"{output_dir}/precision_recall_tradeoff.png", bbox_inches='tight')
print(f"Saved: {output_dir}/precision_recall_tradeoff.png")
plt.close()

# 3. Delta (sentiment contribution) chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics_delta = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
deltas = [df_delta[m] for m in metrics_delta]
colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in deltas]

bars = ax.bar(range(len(metrics_delta)), deltas, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Metrics')
ax.set_ylabel('Change (Combined - Baseline)')
ax.set_title('Sentiment Feature Contribution (Delta)')
ax.set_xticks(range(len(metrics_delta)))
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    label = f'{height:+.4f}\n({height*100:+.2f}%)'
    ax.text(bar.get_x() + bar.get_width()/2., 
            height + (0.005 if height > 0 else -0.015),
            label, ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig(f"{output_dir}/sentiment_contribution_delta.png", bbox_inches='tight')
print(f"Saved: {output_dir}/sentiment_contribution_delta.png")
plt.close()

# 4. Summary table image
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
for _, row in df.iterrows():
    table_data.append([
        row['Experiment'],
        f"{row['Accuracy']:.4f}",
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1']:.4f}",
        f"{row['ROC_AUC']:.4f}"
    ])

table = ax.table(cellText=table_data, 
                 colLabels=['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(6):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color delta row
for i in range(6):
    table[(3, i)].set_facecolor('#f39c12')
    table[(3, i)].set_text_props(weight='bold')

plt.title('Ablation Study Results Summary', fontsize=14, pad=20, weight='bold')
plt.savefig(f"{output_dir}/ablation_table.png", bbox_inches='tight')
print(f"Saved: {output_dir}/ablation_table.png")
plt.close()

print("\n" + "="*60)
print("ABLATION VISUALIZATIONS COMPLETE")
print("="*60)
print(f"\nGenerated 4 charts in: {output_dir}/")
print("\n1. ablation_comparison.png - Bar chart comparing Accuracy, F1, ROC-AUC")
print("2. precision_recall_tradeoff.png - Precision vs Recall comparison")
print("3. sentiment_contribution_delta.png - Delta (sentiment impact)")
print("4. ablation_table.png - Results summary table")
print("\nUse these in your thesis 'Results' section!")

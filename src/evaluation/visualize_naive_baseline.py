"""
Visualize naive baseline comparison for thesis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Load results
df = pd.read_csv("data/models/classification/ablation_with_baseline.csv")

# Filter out delta row
df_models = df[df['Experiment'] != 'Delta (sentiment contribution)'].copy()

output_dir = "data/evaluation/ablation"
os.makedirs(output_dir, exist_ok=True)

print("Creating naive baseline comparison visualizations...")

# 1. Bar chart with Naive baseline
fig, ax = plt.subplots(figsize=(12, 7))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
x = range(len(metrics))
width = 0.25

naive_vals = [df_models.iloc[0][m] for m in metrics]
baseline_vals = [df_models.iloc[1][m] for m in metrics]
combined_vals = [df_models.iloc[2][m] for m in metrics]

bars1 = ax.bar([i - width for i in x], naive_vals, width, label='Naive (always UP)', color='#95a5a6', alpha=0.8)
bars2 = ax.bar([i for i in x], baseline_vals, width, label='Baseline (price only)', color='#3498db', alpha=0.8)
bars3 = ax.bar([i + width for i in x], combined_vals, width, label='Combined (price + sentiment)', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison including Naive Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f"{output_dir}/naive_baseline_comparison.png", bbox_inches='tight')
print(f"Saved: {output_dir}/naive_baseline_comparison.png")
plt.close()

# 2. Accuracy comparison (focus on key metric)
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Naive\n(always UP)', 'Baseline\n(price only)', 'Combined\n(price + sentiment)']
accuracies = [df_models.iloc[0]['Accuracy'], df_models.iloc[1]['Accuracy'], df_models.iloc[2]['Accuracy']]
colors = ['#95a5a6', '#3498db', '#e74c3c']

bars = ax.bar(range(len(models)), accuracies, color=colors, alpha=0.8, width=0.6)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random guess (50%)')
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Test Accuracy: Is Our Model Better Than Naive Baseline?', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim([0.4, 0.6])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels and percentage
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.4f}\n({height*100:.2f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/accuracy_vs_naive.png", bbox_inches='tight')
print(f"Saved: {output_dir}/accuracy_vs_naive.png")
plt.close()

# 3. Updated complete table
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('tight')
ax.axis('off')

table_data = []
for _, row in df_models.iterrows():
    table_data.append([
        row['Experiment'],
        f"{row['Accuracy']:.4f}",
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1']:.4f}",
        f"{row['ROC_AUC']:.4f}"
    ])

table = ax.table(cellText=table_data,
                 colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(6):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color naive row (light gray)
for i in range(6):
    table[(1, i)].set_facecolor('#ecf0f1')

plt.title('Complete Model Comparison with Naive Baseline', fontsize=14, pad=20, weight='bold')
plt.savefig(f"{output_dir}/complete_comparison_table.png", bbox_inches='tight')
print(f"Saved: {output_dir}/complete_comparison_table.png")
plt.close()

print("\n" + "="*60)
print("NAIVE BASELINE VISUALIZATIONS COMPLETE")
print("="*60)
print(f"\nGenerated 3 charts in: {output_dir}/")
print("\n1. naive_baseline_comparison.png - All metrics comparison")
print("2. accuracy_vs_naive.png - Focus on accuracy improvement")
print("3. complete_comparison_table.png - Updated results table")
print("\nKey finding: Combined model matches naive baseline accuracy")
print("but with different precision-recall trade-off!")

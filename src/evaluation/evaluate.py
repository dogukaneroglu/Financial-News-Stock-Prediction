"""
Model evaluation module.
Calculates metrics and generates visualizations for model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelEvaluator:
    """Evaluate stock prediction models."""
    
    def __init__(self, output_dir: str = "data/evaluation"):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R² Score
        r2 = r2_score(y_true, y_pred)
        
        # Directional Accuracy
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], model_name: str = "Model"):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print(f"\n{'='*60}")
        print(f"{model_name} Performance Metrics")
        print('='*60)
        
        print(f"\nRegression Metrics:")
        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAE:  ${metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.4f}")
        
        print(f"\nDirectional Accuracy:")
        print(f"  {metrics['Directional_Accuracy']:.2f}%")
        
        print('='*60)
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_name: Optional[str] = None
    ):
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_name: Filename to save plot
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series plot
        ax1 = axes[0]
        ax1.plot(y_true, label='Actual', color='blue', alpha=0.7, linewidth=2)
        ax1.plot(y_pred, label='Predicted', color='red', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Stock Price ($)')
        ax1.set_title(f'{title} - Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2 = axes[1]
        ax2.scatter(y_true, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Price ($)')
        ax2.set_ylabel('Predicted Price ($)')
        ax2.set_title(f'{title} - Scatter Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Analysis",
        save_name: Optional[str] = None
    ):
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_name: Filename to save plot
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals over time
        ax1 = axes[0, 0]
        ax1.plot(residuals, color='purple', alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Residuals ($)')
        ax1.set_title('Residuals Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2 = axes[0, 1]
        ax2.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Residuals vs predicted
        ax3 = axes[1, 0]
        ax3.scatter(y_pred, residuals, alpha=0.5, s=30, color='purple')
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Price ($)')
        ax3.set_ylabel('Residuals ($)')
        ax3.set_title('Residuals vs Predicted')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax4 = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_directional_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Directional Accuracy",
        save_name: Optional[str] = None
    ):
        """
        Plot directional accuracy analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_name: Filename to save plot
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Calculate directions
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        
        true_direction = (true_diff > 0).astype(int)
        pred_direction = (pred_diff > 0).astype(int)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_direction, pred_direction)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion matrix heatmap
        ax1 = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax1.set_xlabel('Predicted Direction')
        ax1.set_ylabel('Actual Direction')
        ax1.set_title('Direction Confusion Matrix')
        
        # Directional accuracy over time
        ax2 = axes[1]
        correct = (true_direction == pred_direction).astype(int)
        
        # Rolling accuracy
        window = 20
        rolling_acc = pd.Series(correct).rolling(window=window).mean() * 100
        
        ax2.plot(rolling_acc, color='green', linewidth=2, label=f'{window}-day Rolling Accuracy')
        ax2.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Random Baseline (50%)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Rolling Directional Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def compare_models(
        self,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        save_name: Optional[str] = None
    ):
        """
        Compare multiple models.
        
        Args:
            results: Dictionary {model_name: (y_true, y_pred)}
            save_name: Filename to save plot
        """
        metrics_df = []
        
        for model_name, (y_true, y_pred) in results.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            metrics_df.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_df)
        metrics_df = metrics_df.set_index('Model')
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(metrics_df.to_string())
        print("="*80)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            metrics_df[metric].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(metric)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {filepath}")
        
        plt.show()
        plt.close()
        
        return metrics_df
    
    def save_results(self, metrics: Dict[str, float], filename: str):
        """
        Save metrics to CSV.
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        df = pd.DataFrame([metrics])
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Metrics saved to {filepath}")


def main():
    """Example usage of ModelEvaluator."""
    
    print("="*60)
    print("Model Evaluation Module")
    print("="*60)
    print("\nThis module provides evaluation tools for stock prediction models.")
    print("Import this in your evaluation scripts or notebooks.")
    
    # Example with dummy data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randn(n_samples).cumsum() + 100
    y_pred = y_true + np.random.randn(n_samples) * 2
    
    evaluator = ModelEvaluator()
    
    print("\nExample with dummy data:")
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics, "Example Model")


if __name__ == "__main__":
    main()

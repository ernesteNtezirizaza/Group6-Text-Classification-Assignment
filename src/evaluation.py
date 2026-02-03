"""
Evaluation utilities for text classification project.

This module provides:
- Metrics calculation (accuracy, precision, recall, F1)
- Confusion matrix generation
- Visualization functions
- Results comparison tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import time


class ModelEvaluator:
    """
    Comprehensive evaluation for classification models.
    
    Usage:
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, model_name="Logistic Regression")
        evaluator.plot_confusion_matrix(y_true, y_pred)
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: Names of classes (e.g., ['ham', 'spam'])
        """
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.results_history = []
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        embedding_name: str = "Embedding",
        training_time: Optional[float] = None
    ) -> Dict:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            embedding_name: Name of the embedding
            training_time: Training time in seconds
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'model': model_name,
            'embedding': embedding_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'training_time': training_time
        }
        
        # Add to history
        self.results_history.append(metrics)
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in readable format."""
        print(f"\n{'='*60}")
        print(f"Model: {metrics['model']} | Embedding: {metrics['embedding']}")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        if metrics['training_time']:
            print(f"Training Time: {metrics['training_time']:.2f}s")
        print(f"{'='*60}\n")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Classification Report",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot classification report as heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.iloc[:-3, :-1]  # Remove last 3 rows and support column
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            df_report,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Score'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification report saved to {save_path}")
        
        plt.show()
    
    def save_results_table(
        self,
        filepath: str = 'results/tables/model_results.csv'
    ):
        """
        Save results history to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        df = pd.DataFrame(self.results_history)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return df


class ResultsComparator:
    """
    Compare results across different models and embeddings.
    
    Usage:
        comparator = ResultsComparator()
        comparator.add_results(results_df)
        comparator.plot_comparison()
    """
    
    def __init__(self):
        """Initialize comparator."""
        self.results = None
    
    def load_results(self, filepath: str) -> pd.DataFrame:
        """
        Load results from CSV.
        
        Args:
            filepath: Path to results CSV
            
        Returns:
            DataFrame with results
        """
        self.results = pd.read_csv(filepath)
        return self.results
    
    def create_comparison_table(
        self,
        metric: str = 'f1_score',
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create pivot table comparing models and embeddings.
        
        Args:
            metric: Metric to compare
            save_path: Path to save table
            
        Returns:
            Pivot table DataFrame
        """
        if self.results is None:
            raise ValueError("No results loaded. Use load_results() first.")
        
        pivot = self.results.pivot_table(
            values=metric,
            index='model',
            columns='embedding',
            aggfunc='mean'
        )
        
        if save_path:
            pivot.to_csv(save_path)
            print(f"Comparison table saved to {save_path}")
        
        return pivot
    
    def plot_comparison_bar(
        self,
        metric: str = 'f1_score',
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart comparing models across embeddings.
        
        Args:
            metric: Metric to plot
            figsize: Figure size
            save_path: Path to save figure
        """
        if self.results is None:
            raise ValueError("No results loaded. Use load_results() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by model and embedding
        grouped = self.results.groupby(['model', 'embedding'])[metric].mean().unstack()
        grouped.plot(kind='bar', ax=ax)
        
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.legend(title='Embedding', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_comparison_heatmap(
        self,
        metric: str = 'f1_score',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap comparing models and embeddings.
        
        Args:
            metric: Metric to plot
            figsize: Figure size
            save_path: Path to save figure
        """
        pivot = self.create_comparison_table(metric)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn',
            center=pivot.values.mean(),
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        plt.title(f'{metric.replace("_", " ").title()} - Model vs Embedding',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Embedding', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()


def time_training(func):
    """
    Decorator to time model training.
    
    Usage:
        @time_training
        def train_model():
            # training code
            pass
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        training_time = end - start
        print(f"Training completed in {training_time:.2f} seconds")
        return result, training_time
    return wrapper


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression()
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    evaluator = ModelEvaluator(class_names=['Class 0', 'Class 1'])
    metrics = evaluator.evaluate(
        y_test, y_pred,
        model_name="Logistic Regression",
        embedding_name="TF-IDF",
        training_time=training_time
    )
    evaluator.print_metrics(metrics)
    evaluator.plot_confusion_matrix(y_test, y_pred, title="Example Confusion Matrix")

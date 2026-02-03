"""
Utility functions for text classification project.

Common helper functions used across the project.
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional
from sklearn.model_selection import train_test_split
import logging


def setup_logging(log_file: str = 'logs/training.log', level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_directories(directories: list):
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion of test set
        val_size: Proportion of validation set (from training data)
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val vs test
    stratify_param = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    # Second split: train vs val
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        stratify_param = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_param
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_temp, X_test, y_temp, y_test


def save_model(model: Any, filepath: str, use_joblib: bool = True):
    """
    Save trained model to file.
    
    Args:
        model: Trained model
        filepath: Path to save model
        use_joblib: Use joblib (True) or pickle (False)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if use_joblib:
        joblib.dump(model, filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath: str, use_joblib: bool = True) -> Any:
    """
    Load trained model from file.
    
    Args:
        filepath: Path to model file
        use_joblib: Use joblib (True) or pickle (False)
        
    Returns:
        Loaded model
    """
    if use_joblib:
        model = joblib.load(filepath)
    else:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


def save_results(results: dict, filepath: str):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary with results
        filepath: Path to save JSON
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> dict:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def get_class_distribution(y: np.ndarray) -> pd.DataFrame:
    """
    Get class distribution statistics.
    
    Args:
        y: Label array
        
    Returns:
        DataFrame with class counts and percentages
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    df = pd.DataFrame({
        'Class': unique,
        'Count': counts,
        'Percentage': (counts / total) * 100
    })
    
    return df


def print_data_info(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
):
    """
    Print information about data splits.
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        X_val, y_val: Optional validation split
    """
    print("\n" + "="*60)
    print("DATA SPLIT INFORMATION")
    print("="*60)
    
    print(f"\nTraining set:   {X_train.shape[0]:>6} samples")
    if X_val is not None:
        print(f"Validation set: {X_val.shape[0]:>6} samples")
    print(f"Test set:       {X_test.shape[0]:>6} samples")
    print(f"Feature dim:    {X_train.shape[1]:>6}")
    
    print("\nClass distribution (training):")
    train_dist = get_class_distribution(y_train)
    print(train_dist.to_string(index=False))
    
    print("\nClass distribution (test):")
    test_dist = get_class_distribution(y_test)
    print(test_dist.to_string(index=False))
    
    print("="*60 + "\n")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size string (e.g., "15.2 MB")
    """
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


def combine_results_csv(
    input_dir: str = 'results/tables',
    output_file: str = 'results/comparison_results.csv'
):
    """
    Combine multiple result CSV files into one.
    
    Args:
        input_dir: Directory containing CSV files
        output_file: Path to combined output file
    """
    all_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.csv') and f != os.path.basename(output_file)
    ]
    
    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined {len(all_files)} files into {output_file}")
    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    # Example usage
    print("Utils module loaded successfully!")
    
    # Test directory creation
    create_directories(['test_dir/subdir1', 'test_dir/subdir2'])
    
    # Test class distribution
    y_sample = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    dist = get_class_distribution(y_sample)
    print("\nSample class distribution:")
    print(dist)

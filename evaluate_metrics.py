#!/usr/bin/env python3
"""
DIVER-QA Evaluation Script

This script evaluates the performance of user-defined metrics against human evaluations
on the DIVER-QA dataset.

Usage:
    python evaluate_metrics.py --dataset path/to/diver_qa.csv --include-question
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from typing import Callable, List, Dict, Union, Optional
import sys
import importlib.util
from pathlib import Path


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the DIVER-QA dataset from CSV."""
    try:
        df = pd.read_csv(dataset_path)
        required_columns = ["questions", "answers", "dataset", "prediction", "model", "eval"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {', '.join(missing_columns)}")
            
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def load_metric_function(metric_path: str) -> Callable:
    """
    Load the user-defined metric function from a Python file.
    
    The file should contain a function called 'metric_function' that accepts:
    - reference_answer: str
    - candidate_answer: str
    - question: str (optional)
    
    And returns a score (float) or binary prediction (0 or 1).
    """
    try:
        # Get absolute path
        metric_path = Path(metric_path).resolve()
        
        # Load the module
        spec = importlib.util.spec_from_file_location("user_metric", metric_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load file at {metric_path}")
            
        user_metric = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_metric)
        
        # Check if metric_function exists
        if not hasattr(user_metric, "metric_function"):
            raise AttributeError(
                "The provided Python file must define a function named 'metric_function'"
            )
            
        return user_metric.metric_function
    except Exception as e:
        print(f"Error loading metric function: {e}")
        sys.exit(1)


def evaluate_metric(
    df: pd.DataFrame, 
    metric_func: Callable, 
    include_question: bool = False,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluate a metric function against human evaluations.
    
    Args:
        df: DataFrame containing the dataset
        metric_func: The metric function to evaluate
        include_question: Whether to include the question as input to the metric function
        threshold: If the metric returns continuous scores, this threshold converts them to binary
                  predictions. If None, assumes the metric returns binary predictions.
    
    Returns:
        Dictionary containing MCC, F1, and Accuracy scores
    """
    # Apply the metric function to each row
    predictions = []
    
    for _, row in df.iterrows():
        try:
            if include_question:
                score = metric_func(row["answers"], row["prediction"], row["questions"])
            else:
                score = metric_func(row["answers"], row["prediction"])
                
            # If threshold is provided, convert continuous scores to binary predictions
            if threshold is not None:
                prediction = 1 if score >= threshold else 0
            else:
                prediction = score
                
            predictions.append(prediction)
        except Exception as e:
            print(f"Error applying metric function to row: {e}")
            print(f"Row data: {row}")
            sys.exit(1)
    
    # Get ground truth (human evaluations)
    ground_truth = df["eval"].tolist()
    
    # Calculate metrics
    try:
        mcc = matthews_corrcoef(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions)
        accuracy = accuracy_score(ground_truth, predictions)
        
        return {
            "MCC": mcc,
            "F1": f1,
            "Accuracy": accuracy,
            "Total examples": len(ground_truth),
            "Positive examples": sum(ground_truth),
            "Negative examples": len(ground_truth) - sum(ground_truth)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        sys.exit(1)


def analyze_by_dataset(
    df: pd.DataFrame, 
    predictions: List[int]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze performance broken down by source dataset.
    
    Args:
        df: DataFrame containing the dataset
        predictions: List of predictions from the metric function
        
    Returns:
        Dictionary mapping dataset names to performance metrics
    """
    results = {}
    
    # Add predictions to dataframe for analysis
    df = df.copy()
    df["metric_prediction"] = predictions
    
    # Analyze by dataset
    for dataset_name, group in df.groupby("dataset"):
        ground_truth = group["eval"].tolist()
        group_predictions = group["metric_prediction"].tolist()
        
        results[dataset_name] = {
            "MCC": matthews_corrcoef(ground_truth, group_predictions),
            "F1": f1_score(ground_truth, group_predictions),
            "Accuracy": accuracy_score(ground_truth, group_predictions),
            "Count": len(group)
        }
        
    return results


def analyze_by_model(
    df: pd.DataFrame, 
    predictions: List[int]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze performance broken down by model.
    
    Args:
        df: DataFrame containing the dataset
        predictions: List of predictions from the metric function
        
    Returns:
        Dictionary mapping model names to performance metrics
    """
    results = {}
    
    # Add predictions to dataframe for analysis
    df = df.copy()
    df["metric_prediction"] = predictions
    
    # Analyze by model
    for model_name, group in df.groupby("model"):
        ground_truth = group["eval"].tolist()
        group_predictions = group["metric_prediction"].tolist()
        
        results[model_name] = {
            "MCC": matthews_corrcoef(ground_truth, group_predictions),
            "F1": f1_score(ground_truth, group_predictions),
            "Accuracy": accuracy_score(ground_truth, group_predictions),
            "Count": len(group)
        }
        
    return results


def print_results(
    overall_results: Dict[str, float],
    dataset_results: Dict[str, Dict[str, float]] = None,
    model_results: Dict[str, Dict[str, float]] = None
):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*50)
    print("OVERALL RESULTS")
    print("="*50)
    for metric, value in overall_results.items():
        print(f"{metric:<20}: {value:.4f}" if isinstance(value, float) else f"{metric:<20}: {value}")
    
    if dataset_results:
        print("\n" + "="*50)
        print("RESULTS BY DATASET")
        print("="*50)
        for dataset, metrics in dataset_results.items():
            print(f"\n{dataset} (n={metrics['Count']})")
            print("-"*30)
            for metric, value in metrics.items():
                if metric != "Count":
                    print(f"{metric:<10}: {value:.4f}")
    
    if model_results:
        print("\n" + "="*50)
        print("RESULTS BY MODEL")
        print("="*50)
        for model, metrics in model_results.items():
            print(f"\n{model} (n={metrics['Count']})")
            print("-"*30)
            for metric, value in metrics.items():
                if metric != "Count":
                    print(f"{metric:<10}: {value:.4f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate metric function on DIVER-QA dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to DIVER-QA CSV file")
    parser.add_argument("--metric", type=str, required=True, help="Path to Python file containing metric function")
    parser.add_argument("--include-question", action="store_true", help="Include question in metric function input")
    parser.add_argument("--threshold", type=float, help="Threshold for converting continuous scores to binary predictions")
    parser.add_argument("--by-dataset", action="store_true", help="Show performance breakdown by dataset")
    parser.add_argument("--by-model", action="store_true", help="Show performance breakdown by model")
    
    args = parser.parse_args()
    
    # Load dataset and metric function
    df = load_dataset(args.dataset)
    metric_func = load_metric_function(args.metric)
    
    print(f"Loaded dataset with {len(df)} examples")
    print(f"Using metric function from {args.metric}")
    if args.include_question:
        print("Including question as input to metric function")
    
    # Apply metric function and evaluate
    predictions = []
    for _, row in df.iterrows():
        try:
            if args.include_question:
                score = metric_func(row["answers"], row["prediction"], row["questions"])
            else:
                score = metric_func(row["answers"], row["prediction"])
                
            # Convert to binary prediction if threshold is provided
            if args.threshold is not None:
                prediction = 1 if score >= threshold else 0
            else:
                prediction = score
                
            predictions.append(prediction)
        except Exception as e:
            print(f"Error applying metric function: {e}")
            print(f"Row data: {row}")
            sys.exit(1)
    
    # Get ground truth
    ground_truth = df["eval"].tolist()
    
    # Calculate overall metrics
    overall_results = {
        "MCC": matthews_corrcoef(ground_truth, predictions),
        "F1": f1_score(ground_truth, predictions),
        "Accuracy": accuracy_score(ground_truth, predictions),
        "Total examples": len(ground_truth),
        "Positive examples": sum(ground_truth),
        "Negative examples": len(ground_truth) - sum(ground_truth)
    }
    
    # Calculate breakdowns if requested
    dataset_results = analyze_by_dataset(df, predictions) if args.by_dataset else None
    model_results = analyze_by_model(df, predictions) if args.by_model else None
    
    # Print results
    print_results(overall_results, dataset_results, model_results)


if __name__ == "__main__":
    main()
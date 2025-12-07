"""
Evaluation Module for ECG Scanpath Classification

Computes classification metrics:
- Accuracy: Percentage of correct predictions
- Confusion Matrix: TP, TN, FP, FN breakdown
TP: True Positives
TN: True Negatives
FP: False Positives
FN: False Negatives
"""

import numpy as np


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    
    Accuracy = (Correct Predictions) / (Total Predictions)
             = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: List of actual labels ('EXPERT' or 'NOVICE')
        y_pred: List of predicted labels
    
    Returns:
        accuracy: Float between 0 and 1
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for binary classification.
    
    Returns a 2x2 matrix:
                        Predicted
                    EXPERT    NOVICE
    Actual EXPERT [   TP        FN   ]
           NOVICE [   FP        TN   ]
    
    Where:
    - TP (True Positive): Expert correctly classified as Expert
    - TN (True Negative): Novice correctly classified as Novice
    - FP (False Positive): Novice incorrectly classified as Expert
    - FN (False Negative): Expert incorrectly classified as Novice
    
    Args:
        y_true: List of actual labels
        y_pred: List of predicted labels
    
    Returns:
        confusion_matrix: 2x2 numpy array [[TP, FN], [FP, TN]]
        counts: Dictionary with TP, TN, FP, FN counts
    """
    # Initialize counts
    tp = tn = fp = fn = 0
    
    for true, pred in zip(y_true, y_pred):
        if true == 'EXPERT' and pred == 'EXPERT':
            tp += 1  # True Positive
        elif true == 'NOVICE' and pred == 'NOVICE':
            tn += 1  # True Negative
        elif true == 'NOVICE' and pred == 'EXPERT':
            fp += 1  # False Positive
        elif true == 'EXPERT' and pred == 'NOVICE':
            fn += 1  # False Negative
    
    # Create confusion matrix
    matrix = np.array([[tp, fn], 
                       [fp, tn]])
    
    counts = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    
    return matrix, counts


def print_confusion_matrix(matrix, counts):
    """
    Print confusion matrix in a readable format.
    
    Args:
        matrix: 2x2 numpy array
        counts: Dictionary with TP, TN, FP, FN
    """
    print("\nConfusion Matrix:")
    print("                    Predicted")
    print("                 EXPERT    NOVICE")
    print(f"Actual  EXPERT [   {matrix[0,0]:3d}       {matrix[0,1]:3d}   ]")
    print(f"        NOVICE [   {matrix[1,0]:3d}       {matrix[1,1]:3d}   ]")
    print()
    print(f"TP={counts['TP']}, TN={counts['TN']}, FP={counts['FP']}, FN={counts['FN']}")
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                threshold: float = 0.5) -> Dict:
        """
        Comprehensive model evaluation with focus on imbalanced classification
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary containing various performance metrics
        """
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Basic metrics
        self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        self.metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix based metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        self.metrics['sensitivity'] = tp / (tp + fn)  # True Positive Rate
        self.metrics['specificity'] = tn / (tn + fp)  # True Negative Rate
        self.metrics['precision'] = tp / (tp + fp)
        self.metrics['f1_score'] = f1_score(y_true, y_pred)
        
        # Class-wise metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        self.metrics['class_wise'] = class_report
        
        # Additional metrics for imbalanced data
        self.metrics['balanced_accuracy'] = (self.metrics['sensitivity'] + 
                                           self.metrics['specificity']) / 2
        self.metrics['g_mean'] = np.sqrt(self.metrics['sensitivity'] * 
                                       self.metrics['specificity'])
        
        return self.metrics
    
    def plot_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Plot ROC and Precision-Recall curves
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        """
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        ax1.plot(fpr, tpr, label=f'ROC (AUC = {self.metrics["roc_auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ax2.plot(recall, precision, 
                label=f'PR (AP = {self.metrics["average_precision"]:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def get_optimal_threshold(self, y_true: np.ndarray, 
                            y_pred_proba: np.ndarray) -> Tuple[float, Dict]:
        """
        Find optimal classification threshold using F1 score
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (optimal threshold, metrics at optimal threshold)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Calculate metrics at optimal threshold
        optimal_metrics = self.evaluate(y_true, y_pred_proba, best_threshold)
        
        return best_threshold, optimal_metrics 
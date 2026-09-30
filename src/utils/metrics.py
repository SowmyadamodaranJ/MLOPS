"""
metrics.py
----------
Shared metric computation utilities used by both the ModelEvaluator and
MLflowTracker, eliminating the previous DRY violation where identical
metric logic was duplicated in two modules.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.pipeline import Pipeline

from src.utils.constants import CLASS_LABELS


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all classification evaluation metrics.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1_score, roc_auc
    """
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "roc_auc":   round(
            roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan"), 4
        ),
    }
    return metrics


def compute_pipeline_metrics(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute classification metrics and confusion matrix for a trained pipeline.

    Parameters
    ----------
    pipe : Pipeline
        Fitted sklearn Pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True test labels.

    Returns
    -------
    tuple
        (metrics_dict, confusion_matrix_array)
    """
    y_pred = pipe.predict(X_test)
    y_proba = (
        pipe.predict_proba(X_test)[:, 1]
        if hasattr(pipe, "predict_proba")
        else None
    )
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def get_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_names: List[str] = None,
) -> str:
    """
    Generate a formatted classification report string.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    target_names : list, optional
        Class label names. Defaults to CLASS_LABELS.

    Returns
    -------
    str
        Formatted classification report.
    """
    if target_names is None:
        target_names = CLASS_LABELS
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )


def compute_roc_data(
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve data points and AUC score.

    Returns
    -------
    tuple
        (fpr_array, tpr_array, auc_score)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def compute_pr_data(
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Precision-Recall curve data points and AUC score.

    Returns
    -------
    tuple
        (precision_array, recall_array, pr_auc_score)
    """
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec, prec)
    return prec, rec, pr_auc

"""Evaluation for the jailbreak classifier.

Reports the metrics that matter in a Trust and Safety setting:
precision, recall, F1, and ROC-AUC. In abuse detection the trade-off
between precision (not flagging benign users) and recall (catching real
attacks) is a deliberate policy choice, so we report both plus the full
confusion matrix rather than a single accuracy number, which would be
misleading on imbalanced data.

Includes stratified k-fold cross-validation to give an honest estimate
of generalization rather than a single lucky train/test split.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .classifier import JailbreakClassifier


def evaluate_split(clf: JailbreakClassifier, prompts: List[str],
                   labels: List[int]) -> Dict[str, float]:
    """Evaluate a fitted classifier on a held-out set."""
    probs = clf.predict_proba(prompts)
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probs)), 4) if len(set(labels)) > 1 else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def cross_validate(prompts: List[str], labels: List[int],
                   n_splits: int = 5, seed: int = 42) -> Dict[str, float]:
    """Stratified k-fold cross-validation. Returns mean and std of each metric."""
    prompts_arr = np.array(prompts, dtype=object)
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    collected: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": [], "roc_auc": []}
    for train_idx, test_idx in skf.split(prompts_arr, labels_arr):
        clf = JailbreakClassifier()
        clf.fit(list(prompts_arr[train_idx]), list(labels_arr[train_idx]))
        metrics = evaluate_split(clf, list(prompts_arr[test_idx]), list(labels_arr[test_idx]))
        for key in collected:
            collected[key].append(metrics[key])

    summary: Dict[str, float] = {}
    for key, values in collected.items():
        summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
        summary[f"{key}_std"] = round(float(np.std(values)), 4)
    return summary

from typing import Dict

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def explain_accuracy_limitations() -> str:
    """Explain why accuracy alone is insufficient for this problem."""
    return (
        "Accuracy can be misleading when classes are imbalanced. A model that "
        "predicts the majority class most of the time can still score high accuracy, "
        "but it will miss congestion or incident cases. Precision, recall, F1-score, "
        "and ROC-AUC give a clearer picture of how well we detect the positive class."
    )


def evaluate_model(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute key evaluation metrics for binary classification."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }

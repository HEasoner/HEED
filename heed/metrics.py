import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve


def fpr_at_recall(labels: np.ndarray, scores: np.ndarray, target: float = 0.95) -> float:
    """FPR at the first threshold whose recall of HALL (label 1) reaches `target`."""
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = np.where(tpr >= target)[0]
    return float(fpr[ok].min()) if len(ok) else float("nan")


def detection_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """HALL is the positive class; higher score = more likely a false recall."""
    return dict(auroc=float(roc_auc_score(labels, scores)),
                aupr=float(average_precision_score(labels, scores)),
                fpr95=fpr_at_recall(labels, scores))


def monitor_value(labels: np.ndarray, probs: np.ndarray, monitor: str) -> float:
    """Early-stopping criterion on the validation split. probs: N x 2 expected class probabilities."""
    auroc = float(roc_auc_score(labels, probs[:, 1]))
    if monitor == "auroc":
        return auroc
    if monitor == "composite":
        return 0.5 * float(balanced_accuracy_score(labels, probs.argmax(1))) + 0.5 * auroc
    raise ValueError(monitor)

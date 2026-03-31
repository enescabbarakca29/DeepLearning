from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification(y_true, y_pred, y_score=None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "classification_report_dict": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if y_score is not None:
        metrics["score_mean"] = float(pd.Series(y_score).mean())
    return metrics


def history_to_frame(history: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(history)


def build_comparison_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records)


def assess_fit_from_history(history: dict[str, list[float]], gap_threshold: float = 0.08) -> str:
    train_acc = history["train_accuracy"][-1]
    val_acc = history["val_accuracy"][-1]
    train_loss = history["train_loss"][-1]
    val_loss = history["val_loss"][-1]
    accuracy_gap = train_acc - val_acc
    loss_gap = val_loss - train_loss

    if accuracy_gap > gap_threshold and loss_gap > 0.05:
        return "Overfitting eğilimi gözleniyor: eğitim performansı doğrulamadan belirgin biçimde daha yüksek."
    if train_acc < 0.80 and val_acc < 0.80:
        return "Underfitting eğilimi gözleniyor: hem eğitim hem doğrulama performansı sınırlı kalıyor."
    return "Belirgin bir overfitting görülmüyor; modelin genellemesi kabul edilebilir düzeyde."

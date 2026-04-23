from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.plots import plot_confusion_matrix
from src.utils import save_dataframe, save_json, save_text


def evaluate_model(model, dataloader, class_names: list[str], device, model_name: str, output_dir: Path):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    return evaluate_predictions(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        class_names=class_names,
        model_name=model_name,
        output_dir=output_dir,
    )


def evaluate_predictions(y_true, y_pred, class_names: list[str], model_name: str, output_dir: Path):
    labels = list(range(len(class_names)))
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    plot_confusion_matrix(
        matrix=matrix,
        class_names=class_names,
        title=f"{model_name} Confusion Matrix",
        output_path=output_dir / f"{model_name}_confusion_matrix.png",
    )
    save_dataframe(
        pd.DataFrame(matrix, index=class_names, columns=class_names),
        output_dir / f"{model_name}_confusion_matrix.csv",
    )
    save_json(report, output_dir.parent / "reports" / f"{model_name}_classification_report.json")
    save_text(report_text, output_dir.parent / "reports" / f"{model_name}_classification_report.txt")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

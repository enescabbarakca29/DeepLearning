from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

sns.set_theme(style="whitegrid", palette="deep")


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_class_distribution(series, output_path: str | Path, title: str = "Target Class Distribution") -> None:
    output_path = Path(output_path)
    ensure_output_dir(output_path.parent)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=series.astype(str))
    plt.title(title)
    plt.xlabel("DEATH_EVENT")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(correlation_matrix, output_path: str | Path, title: str = "Correlation Heatmap") -> None:
    output_path = Path(output_path)
    ensure_output_dir(output_path.parent)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_history(
    history: dict[str, list[float]],
    output_prefix: str | Path,
    model_name: str,
) -> None:
    output_prefix = Path(output_prefix)
    ensure_output_dir(output_prefix.parent)

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_accuracy.png"), dpi=300)
    plt.close()


def plot_multi_model_history(
    histories: dict[str, dict[str, list[float]]],
    output_path: str | Path,
    metric: str,
    title: str,
) -> None:
    output_path = Path(output_path)
    ensure_output_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    for model_name, history in histories.items():
        plt.plot(history[metric], label=model_name)
    plt.title(title)
    plt.xlabel("Epoch")
    ylabel = metric.replace("_", " ").title()
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix_figure(
    y_true,
    y_pred,
    output_path: str | Path,
    title: str,
    labels: Iterable[str] = ("Survived", "Death"),
) -> None:
    output_path = Path(output_path)
    ensure_output_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=list(labels),
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

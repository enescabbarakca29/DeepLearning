from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


sns.set_theme(style="whitegrid")


def plot_class_distribution(counts: dict[str, int], title: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    classes = list(counts.keys())
    values = list(counts.values())
    sns.barplot(x=classes, y=values, palette="viridis")
    plt.title(title)
    plt.xlabel("Sınıf")
    plt.ylabel("Örnek Sayısı")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    return image.clamp(0, 1)


def plot_sample_images(dataset, class_names: list[str], output_path: Path, n_samples: int = 6) -> None:
    plt.figure(figsize=(14, 8))
    sample_count = min(n_samples, len(dataset))
    for idx in range(sample_count):
        image, label = dataset[idx]
        image = denormalize_image(image)
        plt.subplot(2, 3, idx + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(class_names[label])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_curves(history: list[dict], model_name: str, output_dir: Path) -> None:
    history_df = pd.DataFrame(history)

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Eğrisi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy Eğrisi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_accuracy.png", dpi=300)
    plt.close()


def plot_confusion_matrix(matrix, class_names: list[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


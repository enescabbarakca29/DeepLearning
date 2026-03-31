from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class HeartFailureTorchDataset(TensorDataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        X_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(targets.reshape(-1, 1), dtype=torch.float32)
        super().__init__(X_tensor, y_tensor)


class HeartFailureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.network = nn.Sequential(
            self.hidden,
            nn.Tanh(),
            self.output,
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The lab notebook initializes weights with a small Gaussian distribution and zero biases.
        nn.init.normal_(self.hidden.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class TorchTrainingResult:
    history: dict[str, list[float]]
    model: nn.Module


def create_dataloader(features: np.ndarray, targets: np.ndarray, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    dataset = HeartFailureTorchDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predicted_labels = (predictions >= 0.5).float()
    return float((predicted_labels == targets).float().mean().item())


def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dim: int = 8,
    learning_rate: float = 0.03,
    epochs: int = 250,
    batch_size: int = 32,
    seed: int = 42,
) -> TorchTrainingResult:
    torch.manual_seed(seed)
    model = HeartFailureMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for _ in range(epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accuracies.append(_compute_accuracy(outputs.detach(), batch_targets))

        model.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_losses.append(loss.item())
                val_accuracies.append(_compute_accuracy(outputs, batch_targets))

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["train_accuracy"].append(float(np.mean(train_accuracies)))
        history["val_accuracy"].append(float(np.mean(val_accuracies)))

    return TorchTrainingResult(history=history, model=model)


def predict_torch(model: nn.Module, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32)
        probabilities = model(inputs).cpu().numpy().reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities

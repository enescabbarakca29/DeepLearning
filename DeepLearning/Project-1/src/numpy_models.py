from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


@dataclass
class NumpyTrainingResult:
    model_name: str
    architecture: str
    epochs: int
    history: dict[str, list[float]]
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


class NumpyMLPClassifier:
    def __init__(
        self,
        layer_dims: list[int],
        learning_rate: float = 0.05,
        epochs: int = 500,
        l2_lambda: float = 0.0,
        batch_size: int | None = None,
        seed: int = 42,
        initialization: str = "lab",
    ) -> None:
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.seed = seed
        self.initialization = initialization
        self.parameters = self.initialize_parameters()
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    def initialize_parameters(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        parameters: dict[str, np.ndarray] = {}
        for layer in range(1, len(self.layer_dims)):
            input_dim = self.layer_dims[layer - 1]
            output_dim = self.layer_dims[layer]
            if self.initialization == "lab":
                scale = 0.01
            elif self.initialization == "xavier":
                scale = np.sqrt(1.0 / input_dim)
            else:
                raise ValueError(f"Unsupported initialization strategy: {self.initialization}")
            parameters[f"W{layer}"] = rng.normal(0.0, scale, size=(output_dim, input_dim))
            parameters[f"b{layer}"] = np.zeros((output_dim, 1))
        return parameters

    def forward_propagation(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        A = X.T
        cache: dict[str, np.ndarray] = {"A0": A}
        number_of_layers = len(self.layer_dims) - 1

        for layer in range(1, number_of_layers):
            Z = self.parameters[f"W{layer}"] @ A + self.parameters[f"b{layer}"]
            A = tanh(Z)
            cache[f"Z{layer}"] = Z
            cache[f"A{layer}"] = A

        output_layer = number_of_layers
        ZL = self.parameters[f"W{output_layer}"] @ A + self.parameters[f"b{output_layer}"]
        AL = sigmoid(ZL)
        cache[f"Z{output_layer}"] = ZL
        cache[f"A{output_layer}"] = AL
        return AL, cache

    def compute_cost(self, A_last: np.ndarray, y: np.ndarray) -> float:
        y = y.reshape(1, -1)
        m = y.shape[1]
        epsilon = 1e-12
        clipped_output = np.clip(A_last, epsilon, 1.0 - epsilon)
        data_loss = -np.sum(y * np.log(clipped_output) + (1.0 - y) * np.log(1.0 - clipped_output)) / m
        l2_penalty = 0.0
        if self.l2_lambda > 0:
            l2_penalty = sum(
                np.sum(np.square(self.parameters[f"W{layer}"])) for layer in range(1, len(self.layer_dims))
            )
            l2_penalty = (self.l2_lambda / (2.0 * m)) * l2_penalty
        return float(data_loss + l2_penalty)

    def backward_propagation(self, X: np.ndarray, y: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}
        y = y.reshape(1, -1)
        m = y.shape[1]
        number_of_layers = len(self.layer_dims) - 1
        dZ = cache[f"A{number_of_layers}"] - y

        for layer in reversed(range(1, number_of_layers + 1)):
            A_prev = cache[f"A{layer - 1}"]
            grads[f"dW{layer}"] = (dZ @ A_prev.T) / m
            if self.l2_lambda > 0:
                grads[f"dW{layer}"] += (self.l2_lambda / m) * self.parameters[f"W{layer}"]
            grads[f"db{layer}"] = np.sum(dZ, axis=1, keepdims=True) / m

            if layer > 1:
                dA_prev = self.parameters[f"W{layer}"].T @ dZ
                dZ = dA_prev * (1.0 - np.square(cache[f"A{layer - 1}"]))
        return grads

    def update_parameters(self, grads: dict[str, np.ndarray]) -> None:
        for layer in range(1, len(self.layer_dims)):
            self.parameters[f"W{layer}"] -= self.learning_rate * grads[f"dW{layer}"]
            self.parameters[f"b{layer}"] -= self.learning_rate * grads[f"db{layer}"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities, _ = self.forward_propagation(X)
        return probabilities.reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true.reshape(-1) == y_pred.reshape(-1)))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, list[float]]:
        rng = np.random.default_rng(self.seed)
        sample_count = X_train.shape[0]
        batch_size = self.batch_size or sample_count

        for _ in range(self.epochs):
            permutation = rng.permutation(sample_count)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for start in range(0, sample_count, batch_size):
                stop = start + batch_size
                X_batch = X_shuffled[start:stop]
                y_batch = y_shuffled[start:stop]
                train_output, train_cache = self.forward_propagation(X_batch)
                grads = self.backward_propagation(X_batch, y_batch, train_cache)
                self.update_parameters(grads)

            train_probs = self.predict_proba(X_train)
            val_probs = self.predict_proba(X_val)
            train_preds = (train_probs >= 0.5).astype(int)
            val_preds = (val_probs >= 0.5).astype(int)

            self.history["train_loss"].append(self.compute_cost(train_probs.reshape(1, -1), y_train))
            self.history["val_loss"].append(self.compute_cost(val_probs.reshape(1, -1), y_val))
            self.history["train_accuracy"].append(self._accuracy(y_train, train_preds))
            self.history["val_accuracy"].append(self._accuracy(y_val, val_preds))

        return self.history

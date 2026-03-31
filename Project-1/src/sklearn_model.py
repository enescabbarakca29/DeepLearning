from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier


def train_sklearn_baseline(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    hidden_layer_sizes=(8,),
    activation="tanh",
    solver="sgd",
    random_state=42,
    max_iter=400,
    learning_rate_init=0.05,
    initial_parameters=None,
):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        random_state=random_state,
        max_iter=1,
        learning_rate_init=learning_rate_init,
        learning_rate="constant",
        alpha=0.0,
        momentum=0.0,
        n_iter_no_change=max_iter,
        shuffle=False,
        warm_start=True,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    classes = np.array([0, 1])
    model.partial_fit(X_train, y_train, classes=classes)

    if initial_parameters is not None:
        model.coefs_ = [
            initial_parameters["W1"].T.copy(),
            initial_parameters["W2"].T.copy(),
        ]
        model.intercepts_ = [
            initial_parameters["b1"].reshape(-1).copy(),
            initial_parameters["b2"].reshape(-1).copy(),
        ]

    for _ in range(max_iter):
        model.partial_fit(X_train, y_train)
        history["train_loss"].append(float(model.loss_))
        history["train_accuracy"].append(float(model.score(X_train, y_train)))
        if X_val is not None and y_val is not None:
            val_probabilities = model.predict_proba(X_val)[:, 1]
            epsilon = 1e-12
            clipped = np.clip(val_probabilities, epsilon, 1.0 - epsilon)
            val_loss = -np.mean(y_val * np.log(clipped) + (1 - y_val) * np.log(1 - clipped))
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(model.score(X_val, y_val)))

    return model, history

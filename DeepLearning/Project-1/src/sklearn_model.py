from __future__ import annotations

from sklearn.neural_network import MLPClassifier


def train_sklearn_baseline(
    X_train,
    y_train,
    hidden_layer_sizes=(8,),
    activation="tanh",
    solver="sgd",
    random_state=42,
    max_iter=800,
    learning_rate_init=0.03,
):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        random_state=random_state,
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
        learning_rate="constant",
        alpha=0.0,
        momentum=0.0,
        n_iter_no_change=100,
        shuffle=True,
    )
    model.fit(X_train, y_train)
    return model

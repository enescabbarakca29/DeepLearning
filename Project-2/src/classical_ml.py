import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_predictions
from src.utils import save_dataframe, save_json


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def evaluate_classical_model(model, X_train, y_train, X_val, y_val, X_test, y_test, class_names, output_dir, model_name):
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    train_metrics = evaluate_predictions(y_train, train_predictions, class_names, f"{model_name}_train", output_dir)
    val_metrics = evaluate_predictions(y_val, val_predictions, class_names, f"{model_name}_val", output_dir)
    test_metrics = evaluate_predictions(y_test, test_predictions, class_names, model_name, output_dir)

    coefficient_df = pd.DataFrame(model.coef_)
    save_dataframe(coefficient_df, output_dir.parent / "metrics" / f"{model_name}_coefficients.csv")
    save_json(
        {
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
        },
        output_dir.parent / "metrics" / f"{model_name}_metrics.json",
    )

    return {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

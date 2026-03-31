from __future__ import annotations

from pathlib import Path

from src.data_utils import RANDOM_SEED, prepare_datasets, set_global_seed
from src.metrics import evaluate_classification
from src.numpy_models import NumpyMLPClassifier


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "heart_failure_clinical_records_dataset.csv"

    set_global_seed(RANDOM_SEED)
    split = prepare_datasets(data_path, random_state=RANDOM_SEED)

    baseline_model = NumpyMLPClassifier(
        layer_dims=[split.X_train.shape[1], 8, 1],
        learning_rate=0.05,
        epochs=400,
        seed=RANDOM_SEED,
        initialization="lab",
    )
    baseline_model.fit(split.X_train, split.y_train, split.X_val, split.y_val)
    test_predictions = baseline_model.predict(split.X_test)
    metrics = evaluate_classification(split.y_test, test_predictions, baseline_model.predict_proba(split.X_test))

    print("Heart Failure NumPy Baseline")
    print(f"Input dimension: {split.X_train.shape[1]}")
    print(f"Train/Val/Test sizes: {split.X_train.shape[0]}/{split.X_val.shape[0]}/{split.X_test.shape[0]}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()

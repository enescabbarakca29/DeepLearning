import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn as torch_nn

from src import config
from src.classical_ml import evaluate_classical_model, train_logistic_regression
from src.dataset import build_data_bundle
from src.evaluate import evaluate_model
from src.feature_extraction import extract_features, save_feature_arrays
from src.models import ImprovedCNN, LeNetLikeCNN, build_transfer_model
from src.plots import plot_class_distribution, plot_sample_images
from src.train import train_model
from src.utils import ensure_directories, save_dataframe, save_json, save_text, seconds_to_readable, set_seed


def prepare_output_dirs() -> None:
    ensure_directories(
        [
            config.OUTPUT_DIR,
            config.FIGURES_DIR,
            config.METRICS_DIR,
            config.CONFUSION_MATRIX_DIR,
            config.SAVED_MODELS_DIR,
            config.FEATURES_DIR,
            config.REPORTS_DIR,
        ]
    )


def save_dataset_analysis(bundle) -> None:
    train_distribution_path = config.FIGURES_DIR / "train_class_distribution.png"
    test_distribution_path = config.FIGURES_DIR / "test_class_distribution.png"
    sample_images_path = config.FIGURES_DIR / "sample_images.png"

    plot_class_distribution(bundle.train_counts, "Eğitim Veri Seti Sınıf Dağılımı", train_distribution_path)
    plot_class_distribution(bundle.test_counts, "Test Veri Seti Sınıf Dağılımı", test_distribution_path)
    plot_sample_images(bundle.train_dataset, bundle.class_names, sample_images_path)

    dataset_summary = {
        "class_names": bundle.class_names,
        "num_classes": len(bundle.class_names),
        "train_counts": bundle.train_counts,
        "test_counts": bundle.test_counts,
        "sample_image_shape_hwc": list(bundle.sample_shape),
        "split_sizes": bundle.split_sizes,
        "image_size_after_transform": list(config.IMAGE_SIZE),
        "device": str(config.DEVICE),
    }

    print("Veri seti analizi:")
    for class_name, count in bundle.train_counts.items():
        print(f"  Train {class_name}: {count}")
    for class_name, count in bundle.test_counts.items():
        print(f"  Test {class_name}: {count}")
    print(f"  Sınıf sayısı: {len(bundle.class_names)}")
    print(f"  Örnek görüntü şekli (H, W, C): {bundle.sample_shape}")
    print(f"  Split boyutları: {bundle.split_sizes}")

    save_json(dataset_summary, config.REPORTS_DIR / "dataset_summary.json")
    save_text(pd.DataFrame(dataset_summary.items(), columns=["field", "value"]).to_string(index=False), config.REPORTS_DIR / "dataset_summary.txt")


def train_and_evaluate_cnn(model, model_name: str, train_loader, val_loader, test_loader, learning_rate: float):
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_outputs = train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS,
        save_dir=config.SAVED_MODELS_DIR,
    )

    test_metrics = evaluate_model(
        model=train_outputs["model"],
        dataloader=test_loader,
        class_names=config.CLASS_NAMES,
        device=config.DEVICE,
        model_name=model_name,
        output_dir=config.CONFUSION_MATRIX_DIR,
    )

    metrics_record = {
        "model": model_name,
        "train_accuracy": train_outputs["history"][-1]["train_accuracy"],
        "validation_accuracy": train_outputs["best_val_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1_score"],
        "training_time_seconds": train_outputs["training_time"],
    }
    save_json(metrics_record, config.METRICS_DIR / f"{model_name}_metrics.json")
    return train_outputs, test_metrics, metrics_record


def run_hybrid_pipeline(train_eval_loader, val_loader, test_loader):
    start_time = time.perf_counter()
    trained_resnet = build_transfer_model(num_classes=config.NUM_CLASSES, pretrained=config.USE_PRETRAINED)
    checkpoint_path = config.SAVED_MODELS_DIR / "model3_transfer_best.pth"
    trained_resnet.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    feature_model = torch_nn.Sequential(*list(trained_resnet.children())[:-1])
    feature_model = feature_model.to(config.DEVICE)

    X_train, y_train = extract_features(feature_model, train_eval_loader, config.DEVICE)
    X_val, y_val = extract_features(feature_model, val_loader, config.DEVICE)
    X_test, y_test = extract_features(feature_model, test_loader, config.DEVICE)
    save_feature_arrays(X_train, y_train, X_test, y_test, X_val, y_val, config.FEATURES_DIR)

    classical_model = train_logistic_regression(X_train, y_train)
    classical_metrics = evaluate_classical_model(
        model=classical_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=config.CLASS_NAMES,
        output_dir=config.CONFUSION_MATRIX_DIR,
        model_name="hybrid_logistic_regression",
    )

    return {
        "model": "hybrid_logistic_regression",
        "train_accuracy": classical_metrics["train"]["accuracy"],
        "validation_accuracy": classical_metrics["validation"]["accuracy"],
        "test_accuracy": classical_metrics["test"]["accuracy"],
        "precision": classical_metrics["test"]["precision"],
        "recall": classical_metrics["test"]["recall"],
        "f1_score": classical_metrics["test"]["f1_score"],
        "training_time_seconds": time.perf_counter() - start_time,
    }


def save_predictions_for_seg_pred(model, pred_loader) -> None:
    if pred_loader is None:
        return

    model.eval()
    predictions = []
    with torch.no_grad():
        for images, filenames in pred_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            predicted_indices = outputs.argmax(dim=1).cpu().numpy().tolist()
            predicted_labels = [config.CLASS_NAMES[index] for index in predicted_indices]
            for filename, label in zip(filenames, predicted_labels):
                predictions.append({"filename": filename, "predicted_label": label})

    prediction_df = pd.DataFrame(predictions)
    save_dataframe(prediction_df, config.REPORTS_DIR / "seg_pred_predictions.csv")


def main():
    prepare_output_dirs()
    set_seed(config.RANDOM_SEED)
    print(f"Cihaz: {config.DEVICE}")

    bundle = build_data_bundle()
    save_dataset_analysis(bundle)

    model1_outputs, model1_test_metrics, model1_record = train_and_evaluate_cnn(
        model=LeNetLikeCNN(num_classes=config.NUM_CLASSES),
        model_name="model1_lenet_like",
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
        learning_rate=config.LEARNING_RATE,
    )

    model2_outputs, model2_test_metrics, model2_record = train_and_evaluate_cnn(
        model=ImprovedCNN(num_classes=config.NUM_CLASSES),
        model_name="model2_improved_cnn",
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
        learning_rate=config.LEARNING_RATE,
    )

    model3_outputs, model3_test_metrics, model3_record = train_and_evaluate_cnn(
        model=build_transfer_model(num_classes=config.NUM_CLASSES, pretrained=config.USE_PRETRAINED),
        model_name="model3_transfer",
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
        learning_rate=1e-4 if config.USE_PRETRAINED else config.LEARNING_RATE,
    )

    save_predictions_for_seg_pred(model3_outputs["model"], bundle.pred_loader)

    hybrid_record = run_hybrid_pipeline(
        train_eval_loader=bundle.train_eval_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
    )

    comparison_df = pd.DataFrame([model1_record, model2_record, model3_record, hybrid_record])
    comparison_df["training_time"] = comparison_df["training_time_seconds"].apply(seconds_to_readable)
    comparison_df = comparison_df[
        [
            "model",
            "train_accuracy",
            "validation_accuracy",
            "test_accuracy",
            "precision",
            "recall",
            "f1_score",
            "training_time",
            "training_time_seconds",
        ]
    ]
    best_model_row = comparison_df.sort_values("test_accuracy", ascending=False).iloc[0]

    save_dataframe(comparison_df, config.METRICS_DIR / "comparison_metrics.csv")
    save_json(
        {
            "best_model": best_model_row["model"],
            "comparison": comparison_df.to_dict(orient="records"),
        },
        config.METRICS_DIR / "comparison_metrics.json",
    )
    save_text(
        comparison_df.to_string(index=False),
        config.REPORTS_DIR / "comparison_table.txt",
    )

    print("\nKarşılaştırma tablosu:")
    print(comparison_df.to_string(index=False))
    print(f"\nEn iyi model: {best_model_row['model']}")


if __name__ == "__main__":
    main()

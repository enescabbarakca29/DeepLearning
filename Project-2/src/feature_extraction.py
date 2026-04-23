from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils import save_json, save_text


def extract_features(model, dataloader, device):
    model.eval()
    feature_batches = []
    label_batches = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Feature Extraction", leave=False):
            images = images.to(device)
            features = model(images)
            features = torch.flatten(features, start_dim=1)
            feature_batches.append(features.cpu().numpy())
            label_batches.append(labels.numpy())

    X = np.concatenate(feature_batches, axis=0)
    y = np.concatenate(label_batches, axis=0)
    return X, y


def save_feature_arrays(
    X_train,
    y_train,
    X_test,
    y_test,
    X_val,
    y_val,
    output_dir: Path,
):
    np.save(output_dir / "X_train_features.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test_features.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    shape_info = {
        "X_train_features_shape": list(X_train.shape),
        "y_train_shape": list(y_train.shape),
        "X_val_features_shape": list(X_val.shape),
        "y_val_shape": list(y_val.shape),
        "X_test_features_shape": list(X_test.shape),
        "y_test_shape": list(y_test.shape),
    }
    save_json(shape_info, output_dir / "feature_shapes.json")
    save_text(
        "\n".join([f"{key}: {value}" for key, value in shape_info.items()]),
        output_dir / "feature_shapes.txt",
    )
    print("Feature shape bilgileri:")
    for key, value in shape_info.items():
        print(f"  {key}: {value}")


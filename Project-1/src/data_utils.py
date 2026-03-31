from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
TARGET_COLUMN = "DEATH_EVENT"


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_heart_failure_data(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def summarize_dataframe(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df[target_column].value_counts().to_dict(),
        "descriptive_statistics": df.describe().round(3).to_dict(),
        "correlation_matrix": df[numeric_columns].corr().round(3),
    }


@dataclass
class DataSplit:
    feature_names: list[str]
    target_name: str
    scaler: StandardScaler
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    raw_train: pd.DataFrame
    raw_val: pd.DataFrame
    raw_test: pd.DataFrame

    def to_numpy_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.y_train.reshape(-1, 1), self.y_val.reshape(-1, 1), self.y_test.reshape(-1, 1)

    def metadata(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "train_size": int(self.X_train.shape[0]),
            "val_size": int(self.X_val.shape[0]),
            "test_size": int(self.X_test.shape[0]),
            "input_dim": int(self.X_train.shape[1]),
        }


def prepare_datasets(
    csv_path: str | Path,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = RANDOM_SEED,
) -> DataSplit:
    df = load_heart_failure_data(csv_path)

    feature_columns = [column for column in df.columns if column != target_column]
    X = df[feature_columns]
    y = df[target_column]
    indices = np.arange(len(df))

    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    val_ratio_from_train_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val,
        y_train_val,
        idx_train_val,
        test_size=val_ratio_from_train_val,
        stratify=y_train_val,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return DataSplit(
        feature_names=feature_columns,
        target_name=target_column,
        scaler=scaler,
        X_train=X_train_scaled.astype(np.float64),
        X_val=X_val_scaled.astype(np.float64),
        X_test=X_test_scaled.astype(np.float64),
        y_train=y_train.to_numpy(dtype=np.int64),
        y_val=y_val.to_numpy(dtype=np.int64),
        y_test=y_test.to_numpy(dtype=np.int64),
        train_indices=np.asarray(idx_train),
        val_indices=np.asarray(idx_val),
        test_indices=np.asarray(idx_test),
        raw_train=df.iloc[idx_train].reset_index(drop=True),
        raw_val=df.iloc[idx_val].reset_index(drop=True),
        raw_test=df.iloc[idx_test].reset_index(drop=True),
    )


def save_preprocessing_artifacts(split: DataSplit, directory: str | Path) -> None:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(split.scaler, output_dir / "scaler.joblib")
    indices_payload = {
        "train_indices": split.train_indices.tolist(),
        "val_indices": split.val_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
        "metadata": split.metadata(),
    }
    (output_dir / "split_indices.json").write_text(json.dumps(indices_payload, indent=2), encoding="utf-8")


def save_json_report(payload: dict[str, Any], path: str | Path) -> None:
    def default_converter(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.DataFrame):
            return value.to_dict()
        if isinstance(value, pd.Series):
            return value.to_dict()
        return value

    Path(path).write_text(json.dumps(payload, indent=2, default=default_converter), encoding="utf-8")

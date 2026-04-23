from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from src import config
from src.transforms import get_eval_transforms, get_train_transforms


@dataclass
class DataBundle:
    train_dataset: Dataset
    train_eval_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    pred_dataset: Dataset | None
    train_loader: DataLoader
    train_eval_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    pred_loader: DataLoader | None
    class_names: list[str]
    train_counts: dict[str, int]
    test_counts: dict[str, int]
    sample_shape: tuple[int, int, int]
    split_sizes: dict[str, int]


class TransformSubset(Dataset):
    def __init__(self, dataset: ImageFolder, indices: list[int], transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.classes = dataset.classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample_idx = self.indices[index]
        image_path, label = self.dataset.samples[sample_idx]
        image = self.dataset.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class PredictionFolderDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = config.resolve_image_root(root_dir)
        self.transform = transform
        self.image_paths = sorted(
            [
                path
                for path in self.root_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path.name


def count_images_per_class(root_dir: Path) -> dict[str, int]:
    resolved_root = config.resolve_image_root(root_dir)
    return {
        class_dir.name: len(
            [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        for class_dir in sorted(resolved_root.iterdir())
        if class_dir.is_dir()
    }


def get_image_properties(root_dir: Path) -> tuple[int, int, int]:
    resolved_root = config.resolve_image_root(root_dir)
    first_class = next(class_dir for class_dir in resolved_root.iterdir() if class_dir.is_dir())
    first_image = next(path for path in first_class.iterdir() if path.is_file())
    with Image.open(first_image) as image:
        width, height = image.size
        channels = len(image.getbands())
    return height, width, channels


def build_data_bundle() -> DataBundle:
    train_root = config.resolve_image_root(config.TRAIN_DIR)
    test_root = config.resolve_image_root(config.TEST_DIR)

    base_train_dataset = ImageFolder(train_root)
    base_test_dataset = ImageFolder(test_root)

    all_indices = np.arange(len(base_train_dataset))
    all_targets = np.array(base_train_dataset.targets)

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=all_targets,
    )

    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()

    train_dataset = TransformSubset(
        dataset=base_train_dataset,
        indices=train_indices,
        transform=get_train_transforms(config.IMAGE_SIZE),
    )
    train_eval_dataset = TransformSubset(
        dataset=base_train_dataset,
        indices=train_indices,
        transform=get_eval_transforms(config.IMAGE_SIZE),
    )
    val_dataset = TransformSubset(
        dataset=base_train_dataset,
        indices=val_indices,
        transform=get_eval_transforms(config.IMAGE_SIZE),
    )
    test_dataset = TransformSubset(
        dataset=base_test_dataset,
        indices=list(range(len(base_test_dataset))),
        transform=get_eval_transforms(config.IMAGE_SIZE),
    )

    pred_dataset = None
    pred_loader = None
    resolved_pred_dir = config.resolve_image_root(config.PRED_DIR)
    if resolved_pred_dir.exists():
        pred_dataset = PredictionFolderDataset(resolved_pred_dir, transform=get_eval_transforms(config.IMAGE_SIZE))
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return DataBundle(
        train_dataset=train_dataset,
        train_eval_dataset=train_eval_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        pred_dataset=pred_dataset,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pred_loader=pred_loader,
        class_names=base_train_dataset.classes,
        train_counts=count_images_per_class(config.TRAIN_DIR),
        test_counts=count_images_per_class(config.TEST_DIR),
        sample_shape=get_image_properties(config.TRAIN_DIR),
        split_sizes={
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
        },
    )


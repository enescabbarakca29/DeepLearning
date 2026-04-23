from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "seg_train"
TEST_DIR = DATA_DIR / "seg_test"
PRED_DIR = DATA_DIR / "seg_pred"
OUTPUT_DIR = BASE_DIR / "outputs"

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
VAL_SIZE = 0.2
NUM_WORKERS = 0
PIN_MEMORY = False
EARLY_STOPPING_PATIENCE = 3
USE_PRETRAINED = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_image_root(path: Path) -> Path:
    """Resolve Intel dataset folders that may contain an extra nested directory."""
    if not path.exists():
        return path
    children = [item for item in path.iterdir() if item.is_dir()]
    if len(children) == 1 and children[0].name == path.name:
        return children[0]
    return path


def infer_class_names(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        return ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    resolved = resolve_image_root(train_dir)
    class_names = sorted([item.name for item in resolved.iterdir() if item.is_dir()])
    if not class_names:
        return ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    return class_names


CLASS_NAMES = infer_class_names(TRAIN_DIR)
NUM_CLASSES = len(CLASS_NAMES)

FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrices"
SAVED_MODELS_DIR = OUTPUT_DIR / "saved_models"
FEATURES_DIR = OUTPUT_DIR / "features"
REPORTS_DIR = OUTPUT_DIR / "reports"

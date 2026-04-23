import copy
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src import config
from src.plots import plot_training_curves
from src.utils import save_dataframe, save_json


def run_epoch(model, dataloader, criterion, optimizer=None, device=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        progress = tqdm(dataloader, leave=False, desc="Train" if is_train else "Eval")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size
            progress.set_postfix(loss=loss.item())

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    return epoch_loss, epoch_accuracy


def train_model(
    model,
    model_name: str,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs: int,
    save_dir: Path,
):
    history = []
    best_val_accuracy = 0.0
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    checkpoint_path = save_dir / f"{model_name}_best.pth"
    patience_counter = 0

    start_time = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_record)

        print(
            f"[{model_name}] Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"[{model_name}] Early stopping tetiklendi.")
            break

    training_time = time.perf_counter() - start_time
    model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history)
    save_dataframe(history_df, config.METRICS_DIR / f"{model_name}_history.csv")
    save_json(
        {
            "model_name": model_name,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "training_time_seconds": training_time,
            "history": history,
        },
        config.METRICS_DIR / f"{model_name}_history.json",
    )
    plot_training_curves(history, model_name, config.FIGURES_DIR)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "training_time": training_time,
        "checkpoint_path": checkpoint_path,
    }


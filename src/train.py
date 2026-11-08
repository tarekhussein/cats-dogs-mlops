
import subprocess
import yaml
from pathlib import Path
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn

from dataset import create_dataloaders, BATCH_SIZE
from model import create_model


SEED = 42
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = Path("models/best_model.pth")

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "cats-dogs-classification"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train():
    set_seed(SEED)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    train_loader, val_loader, _ = create_dataloaders()

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = -1.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="mobilenetv2_baseline") as run:
        

        # Record the source-code version
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()

        # Record the DVC-tracked dataset version
        dvc_metadata = yaml.safe_load(
            Path("data/processed.dvc").read_text()
        )
        dvc_hash = dvc_metadata["outs"][0]["md5"]

        # Check for uncommitted source-code changes
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()

        mlflow.set_tags({
            "git_commit": git_commit,
            "dvc_dataset_hash": dvc_hash,
            "git_dirty": str(bool(git_status)).lower(),
            "dataset": "Microsoft Cats vs Dogs",
            "framework": "PyTorch",
        })

        print("Git commit:", git_commit)
        print("DVC dataset hash:", dvc_hash)
        print("Uncommitted tracked changes:", bool(git_status))


        print("Git commit:", git_commit)
        print("DVC dataset hash:", dvc_hash)

        mlflow.log_params({
            "model": "MobileNetV2",
            "pretrained": True,
            "freeze_features": True,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "seed": SEED,
            "device": str(device),
        })

        for epoch in range(EPOCHS):
            model.train()
            model.features.eval()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)

                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_accuracy = correct / total

            val_loss, val_accuracy = evaluate(
                model, val_loader, criterion, device
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch + 1,
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2%} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2%}",
                flush=True,
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_accuracy": val_accuracy,
                        "class_to_idx": train_loader.dataset.class_to_idx,
                    },
                    MODEL_PATH,
                )

                print(f"Best model saved to {MODEL_PATH}")

        mlflow.log_metric(
            "best_val_accuracy", best_val_accuracy
        )

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="checkpoints",
        )

        print("\nMLflow Run ID:", run.info.run_id)
        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.2%}"
        )


if __name__ == "__main__":
    train()

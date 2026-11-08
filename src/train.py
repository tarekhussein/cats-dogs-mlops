
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from dataset import create_dataloaders
from model import create_model


# Training configuration
SEED = 42
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = Path("models/best_model.pth")


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

    best_val_accuracy = 0.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()

        # Keep frozen feature extractor's BatchNorm
        # statistics unchanged.
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

    print(f"\nBest validation accuracy: {best_val_accuracy:.2%}")


if __name__ == "__main__":
    train()

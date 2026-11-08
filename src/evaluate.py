
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset import create_dataloaders
from model import create_model


MODEL_PATH = Path("models/best_model.pth")


def evaluate_model():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    _, _, test_loader = create_dataloaders()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = create_model(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation accuracy:",
        f"{checkpoint['val_accuracy']:.2%}",
    )

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(
                predictions.cpu().tolist()
            )

    print("\nTest accuracy:",
          f"{accuracy_score(all_labels, all_predictions):.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Cat", "Dog"],
            digits=4,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            all_labels,
            all_predictions,
        )
    )


if __name__ == "__main__":
    evaluate_model()

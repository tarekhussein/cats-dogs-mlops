
from pathlib import Path

import torch

from src.dataset import create_dataloaders
from src.model import create_model


PRODUCTION_PATH = Path("models/best_model.pth")
CANDIDATE_PATH = Path("models/candidate_model.pth")

MIN_IMPROVEMENT = 0.0


def evaluate_checkpoint(checkpoint_path, loader, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    expected_classes = {"Cat": 0, "Dog": 1}
    if checkpoint["class_to_idx"] != expected_classes:
        raise ValueError(
            f"Unexpected class mapping in {checkpoint_path}"
        )

    model = create_model(
        num_classes=2,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Validation dataset is empty")

    return correct / total


def main():
    if not PRODUCTION_PATH.is_file():
        raise FileNotFoundError(PRODUCTION_PATH)

    if not CANDIDATE_PATH.is_file():
        raise FileNotFoundError(
            f"{CANDIDATE_PATH} does not exist. "
            "Train a candidate model first."
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Evaluation device:", device)

    _, val_loader, _ = create_dataloaders()

    production_accuracy = evaluate_checkpoint(
        PRODUCTION_PATH, val_loader, device
    )

    candidate_accuracy = evaluate_checkpoint(
        CANDIDATE_PATH, val_loader, device
    )

    improvement = candidate_accuracy - production_accuracy

    print(f"Production accuracy: {production_accuracy:.2%}")
    print(f"Candidate accuracy:  {candidate_accuracy:.2%}")
    print(f"Difference:          {improvement:+.2%}")

    if improvement > MIN_IMPROVEMENT:
        print("Decision: CANDIDATE ELIGIBLE FOR PROMOTION")
    else:
        print("Decision: KEEP PRODUCTION MODEL")


if __name__ == "__main__":
    main()

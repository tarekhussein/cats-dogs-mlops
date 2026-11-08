
import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)


def create_model(num_classes=2):
    # Load pretrained ImageNet weights
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Freeze the feature extractor
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace the original classifier
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = create_model()
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # Test forward pass
    dummy_input = torch.randn(
        1, 3, 224, 224,
        device=device
    )

    with torch.no_grad():
        output = model(dummy_input)

    print("Output shape:", output.shape)
    print("Forward pass successful!")

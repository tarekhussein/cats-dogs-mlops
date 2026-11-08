
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = Path("data/processed")
IMAGE_SIZE = 224
BATCH_SIZE = 32

# ImageNet normalization for pretrained MobileNetV2
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize,
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])


def create_dataloaders():
    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    print("Class mapping:", train_loader.dataset.class_to_idx)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("\nBatch shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

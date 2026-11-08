
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import random
import shutil

# Configuration
SEED = 42
IMAGES_PER_CLASS = 1000

RAW_DIR = Path("data/raw/microsoft_cats_dogs/PetImages")
OUTPUT_DIR = Path("data/processed")

SPLITS = {
    "train": 700,
    "val": 150,
    "test": 150,
}

random.seed(SEED)


def is_valid_image(image_path):
    """Check whether an image can be decoded."""
    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            img.convert("RGB").load()

        return True

    except (OSError, ValueError, UnidentifiedImageError):
        return False


def prepare_dataset():
    for class_name in ["Cat", "Dog"]:

        class_dir = RAW_DIR / class_name

        valid_images = [
            path
            for path in sorted(class_dir.glob("*.jpg"))
            if is_valid_image(path)
        ]

        if len(valid_images) < IMAGES_PER_CLASS:
            raise ValueError(
                f"Not enough valid images for {class_name}"
            )

        # Reproducible random selection
        random.shuffle(valid_images)
        selected_images = valid_images[:IMAGES_PER_CLASS]

        start = 0

        for split_name, count in SPLITS.items():

            destination = OUTPUT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            split_images = selected_images[start:start + count]

            for image_path in split_images:
                shutil.copy2(
                    image_path,
                    destination / image_path.name
                )

            print(
                f"{split_name} | {class_name}: "
                f"{len(split_images)} images"
            )

            start += count

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    prepare_dataset()

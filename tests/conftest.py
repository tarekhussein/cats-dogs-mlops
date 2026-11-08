
from io import BytesIO

import pytest
import torch
from PIL import Image

import app.main as api_module
from src.model import create_model


@pytest.fixture(scope="session")
def test_checkpoint(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.mktemp("model")
    checkpoint_path = checkpoint_dir / "test_model.pth"

    model = create_model(pretrained=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": {"Cat": 0, "Dog": 1},
        },
        checkpoint_path,
    )

    return checkpoint_path


@pytest.fixture(autouse=True)
def use_test_checkpoint(monkeypatch, test_checkpoint):
    monkeypatch.setattr(
        api_module,
        "MODEL_PATH",
        test_checkpoint,
    )


@pytest.fixture
def test_image():
    image = Image.new(
        "RGB",
        (224, 224),
        color=(180, 120, 80),
    )

    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    return buffer.getvalue()

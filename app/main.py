
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.model import create_model


MODEL_PATH = Path("models/best_model.pth")
DEVICE = torch.device("cpu")

model = None
class_names = []

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    class_names = [
        name for name, idx
        in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    print(f"Model loaded from {MODEL_PATH}")
    print(f"Classes: {class_names}")
    print(f"Device: {DEVICE}")

    yield

    model = None


app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="Image classification using PyTorch MobileNetV2",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported",
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds 10 MB limit",
            )

        image = Image.open(BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probabilities).item())

    return {
        "prediction": class_names[predicted_idx],
        "confidence": round(
            float(probabilities[predicted_idx].item()), 4
        ),
        "probabilities": {
            name: round(float(probabilities[idx].item()), 4)
            for idx, name in enumerate(class_names)
        },
    }


from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"



def test_predict_cat(test_image):
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "cat.jpg",
                    test_image,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["Cat", "Dog"]
    assert 0 <= data["confidence"] <= 1

    assert set(data["probabilities"]) == {"Cat", "Dog"}

    assert abs(
        sum(data["probabilities"].values()) - 1.0
    ) < 0.001

def test_reject_invalid_file_type():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "example.txt",
                    b"This is not an image",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415


def test_reject_corrupted_image():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "corrupted.jpg",
                    b"This is not a real JPEG image",
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"


def test_reject_oversized_image():
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "large.jpg",
                    oversized_content,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image exceeds 10 MB limit"

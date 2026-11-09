
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

API_URL = "http://127.0.0.1:8080"
MODEL_PATH = Path("models/best_model.pth")


def verify_inference():
    import uuid
    import urllib.error

    image_dir = Path("data/processed/test/Cat")
    images = sorted(image_dir.glob("*.jpg"))

    if not images:
        sys.exit("No cat test images found.")

    image_path = images[0]
    boundary = f"----MLOPS{uuid.uuid4().hex}"

    # Read monitoring counters before prediction
    with urllib.request.urlopen(
        f"{API_URL}/metrics", timeout=15
    ) as response:
        metrics_before = json.load(response)

    image_bytes = image_path.read_bytes()

    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_bytes + (
        f"\r\n--{boundary}--\r\n"
    ).encode()

    request = urllib.request.Request(
        f"{API_URL}/predict",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        prediction = json.load(response)

    assert prediction["prediction"] in ("Cat", "Dog")
    assert 0 <= prediction["confidence"] <= 1
    assert abs(sum(prediction["probabilities"].values()) - 1) < 0.01

    print(
        f"PASS: Real inference "
        f"({prediction['prediction']}, "
        f"confidence={prediction['confidence']:.2%})"
    )

    with urllib.request.urlopen(
        f"{API_URL}/metrics", timeout=15
    ) as response:
        metrics_after = json.load(response)

    assert (
        metrics_after["total_predictions"]
        == metrics_before["total_predictions"] + 1
    )

    print("PASS: Monitoring counter incremented")

def main():
    print("Verifying local production deployment...")

    # 1. Calculate the expected model fingerprint
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}")

    expected_hash = hashlib.sha256(
        MODEL_PATH.read_bytes()
    ).hexdigest()

    # 2. Verify API health and model identity
    try:
        with urllib.request.urlopen(
            f"{API_URL}/health", timeout=15
        ) as response:
            health = json.load(response)
    except Exception as exc:
        sys.exit(f"Health check failed: {exc}")

    assert health["status"] == "healthy"
    assert health["model_loaded"] is True
    assert health["device"] == "cpu"
    assert health["model_version"] == expected_hash

    print("PASS: API health")
    print("PASS: Production model fingerprint")
    print("PASS: CPU inference configuration")

    # 3. Verify that the prediction endpoint is available
    with urllib.request.urlopen(
        f"{API_URL}/openapi.json", timeout=15
    ) as response:
        schema = json.load(response)

    assert "/predict" in schema["paths"]

    print("PASS: Prediction endpoint registered")

    # Verify real image inference and monitoring
    verify_inference()

    print("Deployment verification successful!")


if __name__ == "__main__":
    main()

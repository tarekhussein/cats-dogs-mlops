
import io
import json
import sys
import uuid
import urllib.request

from PIL import Image

API_URL = "http://127.0.0.1:8002"


def get_json(path):
    with urllib.request.urlopen(
        f"{API_URL}{path}", timeout=30
    ) as response:
        return json.load(response)


def main():
    print("Starting CI deployment smoke test...")

    health = get_json("/health")
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True
    assert health["device"] == "cpu"
    assert len(health["model_version"]) == 64
    print("PASS: Container health and model loaded")

    metrics_before = get_json("/metrics")

    # Generate a synthetic image for API testing.
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    boundary = f"----MLOPS{uuid.uuid4().hex}"

    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + buffer.getvalue() + (
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
    assert set(prediction["probabilities"]) == {"Cat", "Dog"}
    assert abs(sum(prediction["probabilities"].values()) - 1) < 0.01
    assert prediction["inference_time_ms"] >= 0

    print("PASS: Synthetic-image inference request")

    metrics_after = get_json("/metrics")

    assert (
        metrics_after["total_predictions"]
        == metrics_before["total_predictions"] + 1
    )

    print("PASS: Monitoring counter incremented")
    print("CI deployment smoke test successful!")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

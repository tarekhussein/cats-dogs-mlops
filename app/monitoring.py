
from threading import Lock

_lock = Lock()

_metrics = {
    "total_predictions": 0,
    "cat_predictions": 0,
    "dog_predictions": 0,
    "total_inference_time_ms": 0.0,
}


def record_prediction(prediction: str, inference_time_ms: float):
    with _lock:
        _metrics["total_predictions"] += 1

        if prediction == "Cat":
            _metrics["cat_predictions"] += 1
        elif prediction == "Dog":
            _metrics["dog_predictions"] += 1

        _metrics["total_inference_time_ms"] += inference_time_ms


def get_metrics():
    with _lock:
        total = _metrics["total_predictions"]

        average_latency = (
            _metrics["total_inference_time_ms"] / total
            if total > 0 else 0.0
        )

        return {
            "total_predictions": total,
            "cat_predictions": _metrics["cat_predictions"],
            "dog_predictions": _metrics["dog_predictions"],
            "average_inference_time_ms": round(average_latency, 2),
        }

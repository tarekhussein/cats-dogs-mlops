
from pathlib import Path
import sys

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "deploy/cloudrun.yaml"


def main():
    print("Validating Cloud Run configuration...")

    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["apiVersion"] == "serving.knative.dev/v1"
    assert config["kind"] == "Service"
    assert config["metadata"]["name"] == "cats-dogs-api"
    print("PASS: Service configuration")

    assert (
        config["metadata"]["annotations"]["run.googleapis.com/ingress"]
        == "internal"
    )
    print("PASS: Internal-only ingress")

    template = config["spec"]["template"]
    annotations = template["metadata"]["annotations"]

    assert annotations["autoscaling.knative.dev/minScale"] == "0"
    assert annotations["autoscaling.knative.dev/maxScale"] == "1"
    print("PASS: Autoscaling configuration")

    spec = template["spec"]
    container = spec["containers"][0]

    assert spec["containerConcurrency"] == 1
    assert spec["timeoutSeconds"] == 60
    assert container["ports"][0]["containerPort"] == 8080
    assert container["resources"]["limits"]["cpu"] == "1"
    assert container["resources"]["limits"]["memory"] == "1Gi"
    assert container["image"] == "IMAGE_URI_TO_BE_SET_AT_DEPLOYMENT"
    print("PASS: Container resources and configuration")

    print("Cloud Run offline validation successful!")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

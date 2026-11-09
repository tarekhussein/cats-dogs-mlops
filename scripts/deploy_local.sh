
#!/usr/bin/env bash
set -euo pipefail

IMAGE="cats-dogs-cloud:1.0"
CONTAINER="cats-dogs-production"
PORT=8080

echo "Starting local production deployment..."

# Stop an existing deployment if present
docker rm -f "$CONTAINER" 2>/dev/null || true

# Start production container
docker run -d \
  --platform linux/amd64 \
  --name "$CONTAINER" \
  -p "${PORT}:8080" \
  "$IMAGE"

echo "Waiting for API startup..."

for i in {1..30}; do
  if curl --fail --silent \
    "http://127.0.0.1:${PORT}/health" > /dev/null; then
    echo "Deployment successful!"
    curl --silent "http://127.0.0.1:${PORT}/health"
    echo
    exit 0
  fi

  sleep 2
done

echo "Deployment health check failed."
docker logs "$CONTAINER"
exit 1

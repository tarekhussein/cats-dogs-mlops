
#!/usr/bin/env bash
set -euo pipefail

echo "================================"
echo " Cats vs Dogs - Deployment Test"
echo "================================"

# Run from the project root
cd "$(dirname "$0")/.."

echo "Checking production deployment..."

python scripts/verify_deployment.py

echo "================================"
echo " All deployment checks passed!"
echo "================================"

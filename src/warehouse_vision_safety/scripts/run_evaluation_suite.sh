#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${WAREHOUSE_VISION_PYTHON:-python3}" "$SCRIPT_DIR/run_evaluation_suite.py" "$@"

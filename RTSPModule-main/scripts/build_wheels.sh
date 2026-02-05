#!/bin/bash
# Build wheels for multiple Python versions locally
# Prerequisites: Python 3.9-3.12 installed, GStreamer dev packages, CUDA Toolkit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Clean previous builds
rm -rf wheels/*.whl build/

# Python versions to build for
PYTHON_VERSIONS=("python3.9" "python3.10" "python3.11" "python3.12")

echo "=== RTSPModule Wheel Builder ==="
echo "Project: $PROJECT_DIR"
echo ""

# Ensure build tools are installed
pip install --quiet build scikit-build-core pybind11

for PY in "${PYTHON_VERSIONS[@]}"; do
    if command -v "$PY" &> /dev/null; then
        echo ">>> Building for $PY..."
        
        # Get Python version info
        VERSION=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo "    Python version: $VERSION"
        
        # Build wheel using specific Python interpreter
        "$PY" -m pip install --quiet build scikit-build-core pybind11 2>/dev/null || true
        "$PY" -m build --wheel
        
        echo "    ✓ Built successfully"
    else
        echo ">>> Skipping $PY (not installed)"
    fi
done

echo ""
echo "=== Build Complete ==="
echo "Wheels available in: $PROJECT_DIR/wheels/"
ls -la wheels/*.whl 2>/dev/null || echo "No wheels found"

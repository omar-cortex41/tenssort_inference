#!/bin/bash
# Build wheels for multiple Python versions using conda
# Usage: ./scripts/build_wheels_conda.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/wheels"

# Python versions to build for
PYTHON_VERSIONS=("3.9" "3.10" "3.11" "3.12")

echo "=== RTSPModule Multi-Version Wheel Builder ==="
echo "Project: $PROJECT_DIR"
echo "Output:  $OUTPUT_DIR"
echo ""

# Clean and create output directory
rm -rf "$OUTPUT_DIR"/*.whl 2>/dev/null || true
mkdir -p "$OUTPUT_DIR"

# Auto-detect and source conda
if command -v conda &> /dev/null; then
    # Detect from binary path (e.g., .../bin/conda -> ...)
    CONDA_BASE=$(dirname $(dirname "$(command -v conda)"))
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Warning: conda executable found but profile script not at $CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

# Fallback to standard locations if not yet sourced or detected
if [ -z "$CONDA_PREFIX" ] && [ -z "$CONDA_EXE" ]; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo "Error: Conda not found. Please ensure conda is in your PATH or installed in standard locations."
        exit 1
    fi
fi

for PY_VER in "${PYTHON_VERSIONS[@]}"; do
    ENV_NAME="wheel_py${PY_VER//./}"
    
    echo ">>> Building for Python $PY_VER (env: $ENV_NAME)..."
    
    # Create conda environment if it doesn't exist
    if ! conda info --envs | grep -q "^$ENV_NAME "; then
        echo "    Creating environment..."
        conda create -n "$ENV_NAME" python="$PY_VER" -y -q
    fi
    
    # Activate and install build deps
    conda activate "$ENV_NAME"
    
    echo "    Installing build dependencies..."
    pip install --quiet scikit-build-core pybind11 numpy 2>/dev/null
    
    # Build wheel (--no-deps excludes numpy and other dependencies)
    echo "    Building wheel..."
    cd "$PROJECT_DIR"
    pip wheel --no-build-isolation --no-deps -w "$OUTPUT_DIR" . 2>&1 | grep -E "(Created wheel|filename=|error)" || true
    
    conda deactivate
    
    echo "    ✓ Done"
    echo ""
done

echo "=== Build Complete ==="
echo ""
echo "Wheels created:"
ls -la "$OUTPUT_DIR"/*.whl 2>/dev/null | grep -v numpy || echo "No RTSPModule wheels found"

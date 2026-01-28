#!/bin/bash
set -e

# Build script for ByteTrack C++

cd "$(dirname "$0")"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_SHARED_LIB=ON

# Build
cmake --build . -j$(nproc)

echo ""
echo "Build complete!"
echo "Python module copied to: $(dirname $(pwd))/../bytetrack_cpp*.so"
echo ""
echo "Test with:"
echo "  python -c \"import bytetrack_cpp; print(bytetrack_cpp.__doc__)\""


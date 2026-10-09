#!/usr/bin/env bash
set -Eeuo pipefail

# Build the project and OpenCV 4.14.0 on Ubuntu/Debian.
# Can be run from any directory.

readonly OPENCV_VERSION="4.14.0"
readonly OPENCV_REPOSITORY="https://github.com/opencv/opencv.git"

# Locate the repository root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

if [[ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]]; then
    echo "Error: Could not find CMakeLists.txt in $PROJECT_ROOT" >&2
    exit 1
fi

# Allow paths and build parallelism to be overridden with environment variables.
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
OPENCV_INSTALL_PREFIX="${OPENCV_INSTALL_PREFIX:-$PROJECT_ROOT/opencv-install}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache}"
OPENCV_SOURCE_DIR="${OPENCV_SOURCE_DIR:-$CACHE_DIR/opencv-$OPENCV_VERSION-src}"
OPENCV_BUILD_DIR="${OPENCV_BUILD_DIR:-$CACHE_DIR/opencv-$OPENCV_VERSION-build}"
JOBS="${JOBS:-$(nproc)}"

# This script requires Ubuntu/Debian or another apt-get-based distribution.
if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: This script requires apt-get." >&2
    exit 1
fi

# Support execution as either root or a regular user with sudo.
if (( EUID == 0 )); then
    APT_PREFIX=()
else
    if ! command -v sudo >/dev/null 2>&1; then
        echo "Error: sudo is required to install dependencies." >&2
        exit 1
    fi
    APT_PREFIX=(sudo)
fi

# 1. Install system dependencies.
echo "==> Installing system dependencies"

"${APT_PREFIX[@]}" apt-get update

"${APT_PREFIX[@]}" apt-get install -y \
    git \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libeigen3-dev \
    libglew-dev \
    libglfw3-dev

# 2. Initialize Git submodules recursively.
if [[ -f "$PROJECT_ROOT/.gitmodules" ]]; then
    echo "==> Initializing Git submodules"
    git -C "$PROJECT_ROOT" submodule update --init --recursive
fi

# 3. Build OpenCV only if it is not already installed in our local prefix.
OPENCV_CONFIG="$OPENCV_INSTALL_PREFIX/lib/cmake/opencv4/OpenCVConfig.cmake"

if [[ -f "$OPENCV_CONFIG" ]]; then
    echo "==> OpenCV $OPENCV_VERSION is already installed; skipping build"
else
    mkdir -p "$CACHE_DIR"

    echo "==> Preparing OpenCV $OPENCV_VERSION source"

    if [[ ! -f "$OPENCV_SOURCE_DIR/CMakeLists.txt" ]]; then
        rm -rf -- "$OPENCV_SOURCE_DIR"

        git clone \
            --branch "$OPENCV_VERSION" \
            --depth 1 \
            "$OPENCV_REPOSITORY" \
            "$OPENCV_SOURCE_DIR"
    fi

    # Configure OpenCV.
    echo "==> Configuring OpenCV"

    cmake -S "$OPENCV_SOURCE_DIR" -B "$OPENCV_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$OPENCV_INSTALL_PREFIX" \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_DOCS=OFF

    # Compile OpenCV.
    echo "==> Building OpenCV using $JOBS parallel jobs"

    cmake --build "$OPENCV_BUILD_DIR" \
        --config Release \
        --parallel "$JOBS"

    # Install OpenCV locally; no system-wide installation is needed.
    echo "==> Installing OpenCV"

    cmake --install "$OPENCV_BUILD_DIR"
fi

# Ensure the expected OpenCV CMake configuration exists.
if [[ ! -f "$OPENCV_CONFIG" ]]; then
    echo "Error: OpenCVConfig.cmake was not generated at:" >&2
    echo "  $OPENCV_CONFIG" >&2
    exit 1
fi

# 4. Configure the project, explicitly selecting our OpenCV installation.
echo "==> Configuring project"

cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR="$(dirname -- "$OPENCV_CONFIG")"

# 5. Build the project.
echo "==> Building project using $JOBS parallel jobs"

cmake --build "$BUILD_DIR" \
    --config Release \
    --parallel "$JOBS"

echo "==> Build completed successfully"

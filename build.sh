#!/usr/bin/env bash
set -Eeuo pipefail

# Build this project and its pinned OpenCV dependency on Ubuntu/Debian.
# Run from any directory: ./build.sh

readonly OPENCV_VERSION="4.14.0"
readonly OPENCV_REPOSITORY="https://github.com/opencv/opencv.git"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  PROJECT_ROOT="$SCRIPT_DIR"
fi

if [[ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]]; then
  echo "Error: Could not find CMakeLists.txt in project root: $PROJECT_ROOT" >&2
  echo "Place this script in the repository (or one of its subdirectories) and try again." >&2
  exit 1
fi

BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
OPENCV_INSTALL_PREFIX="${OPENCV_INSTALL_PREFIX:-$PROJECT_ROOT/opencv-install}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache}"
OPENCV_SOURCE_DIR="${OPENCV_SOURCE_DIR:-$CACHE_DIR/opencv-$OPENCV_VERSION-src}"
OPENCV_BUILD_DIR="${OPENCV_BUILD_DIR:-$CACHE_DIR/opencv-$OPENCV_VERSION-build}"
JOBS="${JOBS:-$(nproc)}"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "Error: This script currently supports Ubuntu/Debian systems with apt-get." >&2
  exit 1
fi

if (( EUID == 0 )); then
  APT_PREFIX=()
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Error: sudo is required to install system dependencies. Run as root or install sudo." >&2
    exit 1
  fi
  APT_PREFIX=(sudo)
fi

echo "==> Installing system dependencies"
"${APT_PREFIX[@]}" apt-get update
"${APT_PREFIX[@]}" apt-get install -y \
  git \
  build-essential \
  cmake \
  pkg-config \
  libgl1-mesa-dev \
  libegl1-mesa-dev \
  libepoxy-dev \
  libglu1-mesa-dev \
  libeigen3-dev \
  libglew-dev \
  libglfw3-dev

# Match the checkout behavior from actions/checkout with submodules: recursive.
if [[ -f "$PROJECT_ROOT/.gitmodules" ]]; then
  echo "==> Initializing Git submodules recursively"
  git -C "$PROJECT_ROOT" submodule update --init --recursive
fi

OPENCV_CONFIG="$OPENCV_INSTALL_PREFIX/lib/cmake/opencv4/OpenCVConfig.cmake"
if [[ -f "$OPENCV_CONFIG" ]]; then
  echo "==> OpenCV $OPENCV_VERSION already installed at $OPENCV_INSTALL_PREFIX; skipping build"
else
  echo "==> Preparing OpenCV $OPENCV_VERSION source"
  mkdir -p "$CACHE_DIR" "$(dirname -- "$OPENCV_INSTALL_PREFIX")"
  if [[ ! -f "$OPENCV_SOURCE_DIR/CMakeLists.txt" ]]; then
    rm -rf -- "$OPENCV_SOURCE_DIR"
    git clone --branch "$OPENCV_VERSION" --depth 1 "$OPENCV_REPOSITORY" "$OPENCV_SOURCE_DIR"
  fi

  echo "==> Configuring OpenCV $OPENCV_VERSION"
  cmake -S "$OPENCV_SOURCE_DIR" -B "$OPENCV_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$OPENCV_INSTALL_PREFIX" \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_DOCS=OFF

  echo "==> Building OpenCV using $JOBS parallel jobs"
  cmake --build "$OPENCV_BUILD_DIR" --config Release --parallel "$JOBS"

  echo "==> Installing OpenCV to $OPENCV_INSTALL_PREFIX"
  cmake --install "$OPENCV_BUILD_DIR"
fi

if [[ ! -f "$OPENCV_CONFIG" ]]; then
  echo "Error: OpenCV configuration file not found after installation: $OPENCV_CONFIG" >&2
  exit 1
fi

echo "==> Configuring project"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$(dirname -- "$OPENCV_CONFIG")"

echo "==> Building project using $JOBS parallel jobs"
cmake --build "$BUILD_DIR" --config Release --parallel "$JOBS"

echo "==> Build completed successfully"

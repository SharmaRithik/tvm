#!/usr/bin/env bash
# setup.sh — Build TVM with WebGPU shader generation support.
#
# Detects available GPU backends (Vulkan, OpenCL) and installs missing
# system deps, initializes submodules, builds TVM natively, and sets up
# a Python venv ready to run generate_shaders.py and test.sh.
#
# Usage:
#   bash webgpu/setup.sh [--no-opencl] [--no-vulkan] [--jobs N]
#
# After running, activate the environment with:
#   source webgpu/env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TVM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$TVM_ROOT/build"
VENV_DIR="$TVM_ROOT/.venv"

# ── Defaults ────────────────────────────────────────────────────────────────
USE_OPENCL=ON
USE_VULKAN=ON
JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# ── Argument parsing ─────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --no-opencl) USE_OPENCL=OFF ;;
    --no-vulkan)  USE_VULKAN=OFF  ;;
    --jobs)       shift; JOBS=$1  ;;
    --jobs=*)     JOBS="${arg#*=}" ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
info()    { echo "[setup] $*"; }
warn()    { echo "[setup] WARNING: $*" >&2; }
die()     { echo "[setup] ERROR: $*" >&2; exit 1; }
has_cmd() { command -v "$1" &>/dev/null; }

# Detect package manager
if has_cmd apt-get;    then PKG_MGR=apt
elif has_cmd dnf;      then PKG_MGR=dnf
elif has_cmd pacman;   then PKG_MGR=pacman
elif has_cmd brew;     then PKG_MGR=brew
else                        PKG_MGR=unknown
fi

install_pkgs() {
  # Usage: install_pkgs pkg1 pkg2 ...  (distro-agnostic package names resolved inside)
  local pkgs=("$@")
  case $PKG_MGR in
    apt)
      sudo apt-get install -y "${pkgs[@]}"
      ;;
    dnf)
      sudo dnf install -y "${pkgs[@]}"
      ;;
    pacman)
      sudo pacman -S --noconfirm "${pkgs[@]}"
      ;;
    brew)
      brew install "${pkgs[@]}" || true   # brew doesn't fail on already-installed
      ;;
    *)
      warn "Unknown package manager. Please install manually: ${pkgs[*]}"
      ;;
  esac
}

# Map logical dep names to per-distro package names
pkg_name() {
  local dep=$1
  case "$PKG_MGR:$dep" in
    apt:spirv-headers)           echo "spirv-headers" ;;
    apt:spirv-cross)             echo "libspirv-cross-c-shared-dev" ;;
    apt:opencl-headers)          echo "opencl-headers" ;;
    apt:opencl-icd)              echo "ocl-icd-opencl-dev" ;;
    apt:vulkan-dev)              echo "libvulkan-dev" ;;
    apt:cmake)                   echo "cmake" ;;
    apt:ninja)                   echo "ninja-build" ;;
    apt:python3-venv)            echo "python3-venv" ;;
    dnf:spirv-headers)           echo "spirv-headers-devel" ;;
    dnf:spirv-cross)             echo "spirv-cross-devel" ;;
    dnf:opencl-headers)          echo "opencl-headers" ;;
    dnf:opencl-icd)              echo "ocl-icd-devel" ;;
    dnf:vulkan-dev)              echo "vulkan-headers vulkan-loader-devel" ;;
    dnf:cmake)                   echo "cmake" ;;
    dnf:ninja)                   echo "ninja-build" ;;
    dnf:python3-venv)            echo "" ;;   # included in python3 on Fedora
    pacman:spirv-headers)        echo "spirv-headers" ;;
    pacman:spirv-cross)          echo "spirv-cross" ;;
    pacman:opencl-headers)       echo "opencl-headers" ;;
    pacman:opencl-icd)           echo "ocl-icd" ;;
    pacman:vulkan-dev)           echo "vulkan-headers" ;;
    pacman:cmake)                echo "cmake" ;;
    pacman:ninja)                echo "ninja" ;;
    pacman:python3-venv)         echo "" ;;
    brew:spirv-headers)          echo "spirv-headers" ;;
    brew:spirv-cross)            echo "spirv-cross" ;;
    brew:opencl-headers)         echo "" ;;   # bundled with Xcode on macOS
    brew:opencl-icd)             echo "" ;;   # native on macOS
    brew:vulkan-dev)             echo "molten-vk" ;;
    brew:cmake)                  echo "cmake" ;;
    brew:ninja)                  echo "ninja" ;;
    brew:python3-venv)           echo "" ;;
    *)                           echo "$dep" ;;   # pass through as-is
  esac
}

try_install() {
  local dep=$1
  local pkg
  pkg=$(pkg_name "$dep")
  [[ -z "$pkg" ]] && return 0   # no package needed on this platform
  # shellcheck disable=SC2086
  install_pkgs $pkg
}

# ── Step 1: System dependencies ───────────────────────────────────────────────
info "=== Checking system dependencies ==="

MISSING_PKGS=()

# Build tools
has_cmd cmake  || MISSING_PKGS+=(cmake)
has_cmd ninja  || MISSING_PKGS+=(ninja)

# Vulkan
if [[ "$USE_VULKAN" == "ON" ]]; then
  if ! find /usr/include /usr/local/include 2>/dev/null | grep -q "vulkan/vulkan.h"; then
    MISSING_PKGS+=(vulkan-dev)
  fi
  if ! find /usr/include /usr/local/include 2>/dev/null | grep -q "spirv.hpp"; then
    MISSING_PKGS+=(spirv-headers spirv-cross)
  fi
fi

# OpenCL
if [[ "$USE_OPENCL" == "ON" ]]; then
  if ! find /usr/include /usr/local/include 2>/dev/null | grep -q "CL/cl.h"; then
    MISSING_PKGS+=(opencl-headers opencl-icd)
  fi
fi

# Python venv support
if ! python3 -m venv --help &>/dev/null; then
  MISSING_PKGS+=(python3-venv)
fi

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  info "Installing missing packages: ${MISSING_PKGS[*]}"
  for dep in "${MISSING_PKGS[@]}"; do
    try_install "$dep"
  done
else
  info "All system dependencies present."
fi

# ── Step 2: Submodules ────────────────────────────────────────────────────────
info "=== Initializing submodules ==="
cd "$TVM_ROOT"

# Always need tvm-ffi; only init others if already partially cloned
if [[ ! -f "3rdparty/tvm-ffi/CMakeLists.txt" ]]; then
  git submodule update --init --recursive 3rdparty/tvm-ffi
fi

# ── Step 3: config.cmake ──────────────────────────────────────────────────────
info "=== Configuring build ==="
mkdir -p "$BUILD_DIR"

# Only write config if it doesn't exist yet, so user edits are preserved
if [[ ! -f "$BUILD_DIR/config.cmake" ]]; then
  cp "$TVM_ROOT/cmake/config.cmake" "$BUILD_DIR/config.cmake"

  # Enable/disable backends based on what's available
  if [[ "$USE_VULKAN" == "ON" ]] && find /usr/include /usr/local/include 2>/dev/null | grep -q "vulkan/vulkan.h"; then
    sed -i 's/^set(USE_VULKAN OFF)/set(USE_VULKAN ON)  # auto-enabled by setup.sh/' "$BUILD_DIR/config.cmake"
    info "Vulkan: ON"
  else
    warn "Vulkan headers not found — disabling USE_VULKAN"
    USE_VULKAN=OFF
  fi

  if [[ "$USE_OPENCL" == "ON" ]] && find /usr/include /usr/local/include 2>/dev/null | grep -q "CL/cl.h"; then
    sed -i 's/^set(USE_OPENCL OFF)/set(USE_OPENCL ON)  # auto-enabled by setup.sh/' "$BUILD_DIR/config.cmake"
    info "OpenCL: ON"
  else
    warn "OpenCL headers not found — disabling USE_OPENCL"
    USE_OPENCL=OFF
  fi

  # Enable LLVM if llvm-config is available
  LLVM_CONFIG=$(command -v llvm-config || ls /usr/lib/llvm-*/bin/llvm-config 2>/dev/null | sort -V | tail -1 || true)
  if [[ -n "$LLVM_CONFIG" ]]; then
    sed -i "s|^set(USE_LLVM OFF)|set(USE_LLVM $LLVM_CONFIG)  # auto-detected by setup.sh|" "$BUILD_DIR/config.cmake"
    info "LLVM: ON ($LLVM_CONFIG)"
  else
    warn "llvm-config not found — CPU codegen disabled. Install llvm-dev to enable."
  fi

  sed -i 's/^set(SUMMARIZE OFF)/set(SUMMARIZE ON)/' "$BUILD_DIR/config.cmake"
else
  info "config.cmake already exists — skipping (delete $BUILD_DIR/config.cmake to reset)"
fi

# ── Step 4: CMake + build ─────────────────────────────────────────────────────
info "=== Running CMake ==="
cd "$BUILD_DIR"
cmake .. -G Ninja 2>&1 | grep -v "^--" || true
cmake .. -G Ninja   # run twice: first pass may have warnings, second is fast

info "=== Building TVM (jobs=$JOBS) ==="
ninja -j"$JOBS"

# ── Step 5: Python venv ───────────────────────────────────────────────────────
info "=== Setting up Python venv ==="
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"

# Install tvm-ffi (needs scikit-build-core + cython to compile extensions)
if ! "$VENV_DIR/bin/python" -c "import tvm_ffi" &>/dev/null 2>&1; then
  info "Installing tvm-ffi build deps..."
  "$PIP" install --quiet scikit-build-core cython setuptools-scm
  info "Installing tvm-ffi..."
  "$PIP" install --quiet -e "$TVM_ROOT/3rdparty/tvm-ffi"
fi

# Install TVM Python runtime deps
info "Installing TVM Python dependencies..."
"$PIP" install --quiet numpy psutil attrs cloudpickle decorator scipy synr

info "=== Verifying TVM import ==="
PYTHONPATH="$TVM_ROOT/python" \
LD_LIBRARY_PATH="$BUILD_DIR:$BUILD_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
DYLD_LIBRARY_PATH="$BUILD_DIR:$BUILD_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
"$VENV_DIR/bin/python" -c "import tvm; print('  TVM version:', tvm.__version__)" 2>/dev/null \
  || die "TVM import failed. Check build output above."

# ── Step 6: Write env.sh ──────────────────────────────────────────────────────
cat > "$SCRIPT_DIR/env.sh" <<EOF
# Source this file to activate the TVM WebGPU environment:
#   source webgpu/env.sh

export TVM_ROOT="$TVM_ROOT"
export PYTHONPATH="\$TVM_ROOT/python\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="\$TVM_ROOT/build:\$TVM_ROOT/build/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="\$TVM_ROOT/build:\$TVM_ROOT/build/lib\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"
export PATH="\$TVM_ROOT/.venv/bin:\$PATH"

echo "[env] TVM WebGPU environment active ($(python3 --version 2>&1))"
EOF

info ""
info "=== Setup complete ==="
info ""
info "Activate the environment with:"
info "  source webgpu/env.sh"
info ""
info "Then run tests:"
info "  bash webgpu/test.sh"
info ""
info "Generate shaders:"
info "  python webgpu/generate_shaders.py --size 1024 --outdir webgpu/generated-shaders"

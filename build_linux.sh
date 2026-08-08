#!/bin/bash
# ============================================================
# LightLLM — Linux Build Script
# Usage: bash build_linux.sh [gpu_arch]
#   gpu_arch: 75 (RTX 2060, default), 89 (RTX 4090), 80 (A100), 86 (RTX 3090)
# ============================================================
# NOTE: no `set -e` — individual checks report and continue
# so the user sees ALL problems at once instead of silent exit.

# ---- GPU arch (default 75 for RTX 2060) ----
ARCH="${1:-75}"
echo "GPU architecture: sm_${ARCH}"

# ---- Helper: check a command exists ----
have() { command -v "$1" >/dev/null 2>&1; }

# ---- Check CUDA ----
if have nvcc; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | grep -oE '[0-9.]+' | head -1)
    echo "nvcc: ${CUDA_VER:-unknown}"
    if [ -n "$CUDA_VER" ] && [[ "${CUDA_VER%%.*}" -lt 12 ]]; then
        echo "NOTE: CUDA ${CUDA_VER} — CMake will use C++17 for CUDA (11.x has partial C++20)."
        echo "      Consider CUDA 12.x for full C++20 + WMMA Tensor Core."
    fi
else
    echo "ERROR: nvcc not found. Install CUDA Toolkit first:"
    echo "  Ubuntu: sudo apt install nvidia-cuda-toolkit"
    echo "  云服务器 (conda): conda install -n lightllm cuda-toolkit=12.4 -c nvidia"
    echo "  Or: https://developer.nvidia.com/cuda-downloads"
fi

# ---- Check GCC ----
if have gcc; then
    GCC_VER=$(gcc -dumpversion 2>/dev/null | grep -oE '^[0-9]+')
    echo "gcc: ${GCC_VER:-unknown}"
    if [ -n "$CUDA_VER" ] && [ -n "$GCC_VER" ] \
       && [[ "${CUDA_VER%%.*}" -lt 12 ]] && [ "$GCC_VER" -gt 11 ]; then
        echo "WARNING: CUDA 11.x officially supports GCC <= 11, but GCC ${GCC_VER} found."
        echo "         This may cause nvcc errors. Install gcc-11:"
        echo "         sudo apt install gcc-11 g++-11"
        echo "         sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100"
        echo "         sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100"
    fi
else
    echo "ERROR: gcc not found. Install: sudo apt install gcc g++"
fi

# ---- Check deps ----
for cmd in cmake ninja; do
    if have $cmd; then
        echo "$cmd: OK ($($cmd --version 2>/dev/null | head -1))"
    else
        echo "ERROR: $cmd not found. Install: sudo apt install $cmd"
    fi
done

# ---- Check model ----
if [ -f "models/qwen2.5-0.5b/model.safetensors" ]; then
    echo "Model: found (models/qwen2.5-0.5b/)"
else
    echo "WARNING: Model not found at models/qwen2.5-0.5b/"
    echo "  Download Qwen2.5-0.5B-Instruct from HuggingFace:"
    echo "  export HF_ENDPOINT=https://hf-mirror.com"
    echo "  hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b"
fi

# ---- Build (proceed if nvcc + cmake exist; report errors otherwise) ----
if have nvcc && have cmake && have ninja; then
    mkdir -p build
    cd build
    echo ""
    echo "=== Configuring ==="
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
    if [ $? -ne 0 ]; then echo "ERROR: cmake configure failed."; exit 1; fi
    echo ""
    echo "=== Building ==="
    ninja -j$(nproc)
    if [ $? -ne 0 ]; then echo "ERROR: ninja build failed."; exit 1; fi
    echo ""
    echo "=== Build complete ==="
    echo "Run from project root:"
    echo "  ./build/test_generate"
    echo "  ./build/test_concurrent"
    echo "  ./build/bench_throughput 60 3.0"
else
    echo ""
    echo "=== Skipping build: missing required tools above ==="
    exit 1
fi

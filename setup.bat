@echo off
REM ============================================================
REM LightLLM — Windows Environment Setup
REM Run this once before building.
REM ============================================================

echo === LightLLM Environment Setup ===
echo.

REM ---- Step 1: Check CUDA ----
echo [1/4] Checking CUDA Toolkit...
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" (
    echo   CUDA 13.2 found.
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" (
    echo   CUDA 12.6 found.
) else (
    echo   WARNING: CUDA not found in default location.
    echo   Download from: https://developer.nvidia.com/cuda-downloads
    echo   Minimum version: 12.0
)
echo.

REM ---- Step 2: Check MSVC ----
echo [2/4] Checking Visual Studio...
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo   Visual Studio 2026 (v18) found.
    set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo   Visual Studio 2022 found.
    set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
) else (
    echo   WARNING: Visual Studio not found.
    echo   Download from: https://visualstudio.microsoft.com/downloads/
    echo   Required: Desktop development with C++ workload
)
echo.

REM ---- Step 3: Check CMake + Ninja ----
echo [3/4] Checking CMake and Ninja...
cmake --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   CMake found.
) else (
    echo   WARNING: CMake not found. Install from https://cmake.org/download/
)
ninja --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   Ninja found.
) else (
    echo   WARNING: Ninja not found. Install via: choco install ninja
    echo   Or download from: https://github.com/ninja-build/ninja/releases
)
echo.

REM ---- Step 4: Download model ----
echo [4/4] Model setup...
if exist "models\qwen2.5-0.5b\model.safetensors" (
    echo   Model found: models/qwen2.5-0.5b/
) else (
    echo   Model NOT found. Download Qwen2.5-0.5B-Instruct:
    echo   1. Visit https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
    echo   2. Download: config.json, model.safetensors, tokenizer.json
    echo   3. Place them in: models\qwen2.5-0.5b\
)
echo.

echo === Setup complete ===
echo Next: run build.bat to compile, or build_debug.bat for debug mode.
pause

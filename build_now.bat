@echo off
setlocal enabledelayedexpansion
echo === LightLLM Build Started ===
echo.

REM Locate VS
set "VCVARS="
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
)
if "%VCVARS%"=="" (
    echo ERROR: VS not found
    exit /b 1
)

echo [1/4] Setting up VS environment...
call "%VCVARS%" x64 >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: vcvars failed
    exit /b 1
)
echo       OK

echo [2/4] Configuring CMake...
cd /d "D:\ai infra project\lightllm"
if not exist build mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: cmake failed — retrying with output:
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
    exit /b 1
)
echo       OK

echo [3/4] Building with ninja...
ninja -j4
if %errorlevel% neq 0 (
    echo.
    echo === BUILD FAILED ===
    exit /b 1
)

echo.
echo [4/4] === BUILD SUCCESS ===
exit /b 0

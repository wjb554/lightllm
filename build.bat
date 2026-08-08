@echo off
REM ============================================================
REM LightLLM — Release Build
REM ============================================================
setlocal

REM ---- Locate vcvarsall ----
set VCVARS=
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
) else (
    echo ERROR: Visual Studio not found.
    exit /b 1
)

call %VCVARS% x64 >nul 2>&1

cd /d "%~dp0"
if not exist build mkdir build
cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 exit /b %errorlevel%

ninja -j4
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo === Build complete ===
echo Tests are in: build\
echo Run from project root: build\test_generate.exe

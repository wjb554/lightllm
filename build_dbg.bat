@echo off
setlocal

REM Locate vcvarsall
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

cd /d "D:\ai infra project\lightllm"
cmake -S . -B build_dbg -G Ninja -DCMAKE_BUILD_TYPE=Debug
if %errorlevel% neq 0 exit /b %errorlevel%

cd build_dbg
ninja test_attention_divergence.exe
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Debug build complete ===

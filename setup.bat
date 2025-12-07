@echo off
echo ================================================
echo    F5-TTS API Server - Setup Script
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing PyTorch with CUDA support...
echo Please select your CUDA version:
echo 1. CUDA 11.8
echo 2. CUDA 12.1
echo 3. CPU only (no GPU)
echo.

set /p cuda_choice="Enter choice (1/2/3): "

if "%cuda_choice%"=="1" (
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
) else if "%cuda_choice%"=="2" (
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Installing F5-TTS and dependencies...
pip install -r requirements.txt

echo.
echo ================================================
echo    Setup Complete!
echo ================================================
echo.
echo To start the server, run: start_server.bat
echo.
pause

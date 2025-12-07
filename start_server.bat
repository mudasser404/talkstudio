@echo off
echo ================================================
echo    F5-TTS Voice Cloning API Server
echo ================================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo Warning: No virtual environment found.
    echo Run setup.bat first to create one.
)

echo.
echo Starting F5-TTS API Server on port 8001...
echo API Docs: http://localhost:8001/docs
echo.

python main.py --host 0.0.0.0 --port 8001

pause

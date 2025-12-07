@echo off
echo ================================================
echo    F5-TTS Model Downloader
echo ================================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo This will download F5-TTS models (~1.5GB)
echo Make sure you have enough disk space.
echo.

python models/download_models.py

echo.
pause

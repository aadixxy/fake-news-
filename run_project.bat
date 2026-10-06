@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo Fake News Detection - Task 2
echo ==============================================
echo.

python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Install Python 3.10 or newer, then run this file again.
    pause
    exit /b 1
)

echo Installing required Python packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed.
    pause
    exit /b 1
)

echo.
echo Starting project...
python fake_news_detection.py

echo.
echo Project finished.
pause

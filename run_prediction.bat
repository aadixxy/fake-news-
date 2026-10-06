@echo off
setlocal
cd /d "%~dp0"

python -c "import joblib; joblib.load('models/fake_news_tfidf_linearsvc.joblib')" 2>nul
if errorlevel 1 (
    echo Model not found.
    echo Run run_project.bat first to download data and train the model.
    pause
    exit /b 1
)

python predict.py
pause

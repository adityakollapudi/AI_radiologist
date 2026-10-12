@echo off
cd /d "%~dp0"
echo ============================================================
echo Running Brain Tumor Detection Prediction...
echo ============================================================
.\.venv\Scripts\python.exe predict.py %*
echo.
pause

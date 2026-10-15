@echo off
cd /d "%~dp0"
echo ============================================================
echo Running Grad-CAM Brain Tumor Localization...
echo ============================================================
.\.venv\Scripts\python.exe gradcam.py %*
echo.
pause

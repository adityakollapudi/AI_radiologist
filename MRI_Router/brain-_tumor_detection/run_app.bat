@echo off
cd /d "%~dp0"
echo ============================================================
echo Starting Brain Tumor Detection & Grad-CAM Web Dashboard...
echo ============================================================
.\.venv\Scripts\streamlit.exe run app.py
pause

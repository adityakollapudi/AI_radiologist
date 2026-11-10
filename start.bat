@echo off
title MEDVISION AI Launcher
echo ========================================================
echo    MEDVISION AI — Unified Multimodal Radiology Platform
echo ========================================================
echo.

echo [1/3] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "MEDVISION AI Backend" cmd /k "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/3] Starting Vite Frontend on http://localhost:5173 ...
start "MEDVISION AI Frontend" cmd /k "npm --prefix frontend run dev"

echo [3/3] Waiting for servers to start up...
timeout /t 4 /nobreak >nul

echo.
echo Opening MEDVISION AI in default browser: http://localhost:5173
start http://localhost:5173

echo.
echo ========================================================
echo MEDVISION AI is running!
echo   - Web UI:       http://localhost:5173
echo   - Backend Docs: http://127.0.0.1:8000/docs
echo   - Health Check: http://127.0.0.1:8000/health
echo ========================================================
echo To stop servers later, run stop.bat or close their terminal windows.
echo.
pause

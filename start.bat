@echo off
echo ========================================
echo   BloodstockAI v3.0
echo   DeepLabCut SuperAnimal-Quadruped
echo ========================================
echo.

echo [1/2] Installing dependencies...
cd /d "%~dp0backend"
pip install -r requirements.txt --quiet

echo [2/2] Starting server...
echo.
echo   Frontend:  http://localhost:8000
echo   API:       http://localhost:8000/api
echo   Database:  %~dp0backend\bloodstockai.db
echo.
echo   Ctrl+C to stop
echo.

set BAI_FRONTEND_DIR=%~dp0frontend
python -m uvicorn app:app --host 0.0.0.0 --port 8000

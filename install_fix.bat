@echo off
echo ============================================================
echo    FIXING MISSING LIBRARIES (FLASK, SKLEARN, ETC.)
echo ============================================================
echo.
echo This will install the necessary tools to run your Plant Doctor app.
echo.

cd /d "%~dp0"

echo [1/2] Updating pip...
python -m pip install --upgrade pip

echo [2/2] Installing required libraries...
python -m pip install flask scikit-learn numpy pillow

echo.
echo ============================================================
echo    DONE! TRY RUNNING 'run_app.bat' NOW.
echo ============================================================
pause

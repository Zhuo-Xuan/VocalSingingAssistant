@echo off
echo ========================================
echo Melody Transformer Setup and Run
echo ========================================
echo.

echo Step 1: Installing dependencies...
echo This may take a few minutes...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Installation failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo Step 2: Starting the server...
echo.
echo The server will start on http://localhost:8000
echo Keep this window open while using the app.
echo Press Ctrl+C to stop the server.
echo.
echo ========================================
echo.

python start_server.py

pause

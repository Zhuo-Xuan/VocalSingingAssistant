@echo off
echo Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo.
    echo Installation completed successfully!
    echo.
    echo Next steps:
    echo 1. Copy .env.example to .env and configure it
    echo 2. Run: python start_server.py
) else (
    echo.
    echo Installation failed. Please check the error messages above.
    pause
)

@echo off
echo Installing minimal dependencies to get started...
echo.

python -m pip install fastapi uvicorn[standard] python-multipart

echo.
echo Minimal dependencies installed!
echo Now you can run: python start_server.py
pause

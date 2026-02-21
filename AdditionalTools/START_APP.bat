@echo off
echo ========================================
echo Starting Melody Transformer Application
echo ========================================
echo.
echo This will start BOTH the backend and frontend servers.
echo.
echo Backend will run on: http://localhost:8000
echo Frontend will run on: http://localhost:8080
echo.
echo Keep this window open while using the app.
echo Press Ctrl+C to stop both servers.
echo.
echo ========================================
echo.

REM Start backend server in background
start "Backend Server" cmd /k "cd /d %~dp0 && python start_server.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server
echo Starting frontend server...
echo Open your browser and go to: http://localhost:8080
echo.
cd frontend
python -m http.server 8080

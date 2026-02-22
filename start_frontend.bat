@echo off
echo Starting frontend server...
echo.
echo Open your browser and go to: http://localhost:8080
echo.
echo Press Ctrl+C to stop the server
echo.
cd frontend
python -m http.server 8080

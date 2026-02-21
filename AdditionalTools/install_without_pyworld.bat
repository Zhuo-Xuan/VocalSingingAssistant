@echo off
echo Installing dependencies (without pyworld)...
echo.
echo Note: pyworld requires Visual C++ compiler.
echo The app will work without it, using librosa for F0 replacement.
echo.

python -m pip install fastapi uvicorn[standard] python-multipart torch librosa soundfile numpy plotly openai

echo.
echo Installation completed!
echo You can now run: python start_server.py
pause

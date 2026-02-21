# How to Open the Frontend

## Quick Start

### Step 1: Start the Backend Server

Open PowerShell/Command Prompt and run:
```powershell
cd d:\musicAI
python start_server.py
```

Wait until you see: `Uvicorn running on http://0.0.0.0:8000`

### Step 2: Open the Frontend

**Option A: Double-click the HTML file**
- Navigate to `d:\musicAI\frontend\`
- Double-click `index.html`
- It will open in your default browser

**Option B: Use HTTP Server (Recommended)**
- Open a NEW PowerShell/Command Prompt window
- Run:
  ```powershell
  cd d:\musicAI\frontend
  python -m http.server 8080
  ```
- Open browser and go to: `http://localhost:8080`

**Option C: Use the batch file**
- Double-click `start_frontend.bat`
- Open browser and go to: `http://localhost:8080`

## Important Notes

⚠️ **The backend server MUST be running first!**
- The frontend needs to connect to `http://localhost:8000` (the backend API)
- If backend is not running, you'll see connection errors

## Troubleshooting

### "Connection refused" error
- Make sure backend server is running on port 8000
- Check that `start_server.py` is running without errors

### Frontend doesn't load
- Try using Option B (HTTP server) instead of double-clicking
- Some browsers block local file access for security

### Port 8080 already in use
- Change the port in `start_frontend.bat`:
  ```powershell
  python -m http.server 8081
  ```
- Then open `http://localhost:8081`

## What You Should See

Once everything is running:
1. A purple gradient page with "Melody Transformer" title
2. Upload Audio and Record Audio buttons
3. After uploading/recording, you'll see:
   - Melody comparison graph
   - AI feedback
   - Download button for processed audio

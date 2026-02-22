# Installation Guide

## Where to Install Dependencies

Install dependencies in your project directory: **`d:\musicAI`**

## Step-by-Step Installation

### Option 1: Using Virtual Environment (Recommended)

1. **Open PowerShell or Command Prompt** in the project directory:
   ```powershell
   cd d:\musicAI
   ```

2. **Create a virtual environment** (optional but recommended):
   ```powershell
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   ```powershell
   # PowerShell
   .\venv\Scripts\Activate.ps1
   
   # Or Command Prompt
   venv\Scripts\activate.bat
   ```

4. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

### Option 2: Install Directly (Without Virtual Environment)

1. **Open PowerShell or Command Prompt** in the project directory:
   ```powershell
   cd d:\musicAI
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Verify Installation

After installation, verify by running:
```powershell
python -c "import fastapi, torch, librosa; print('All dependencies installed!')"
```

## Troubleshooting

### If pip is not found:
- Make sure Python is installed and added to PATH
- Try `python -m pip install -r requirements.txt` instead

### If you get permission errors:
- Run PowerShell/Command Prompt as Administrator
- Or use `pip install --user -r requirements.txt`

### If torch installation fails:
- Visit https://pytorch.org/ to get the correct installation command for your system
- You may need to install CUDA version separately if using GPU

## Next Steps

After installing dependencies:
1. Set up your `.env` file (copy from `.env.example`)
2. Start the server: `python start_server.py`
3. Open `frontend/index.html` in your browser

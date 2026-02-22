#!/usr/bin/env python
"""
Startup script for Melody Transformer backend server
"""
import os
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Set default checkpoint path if not set
if "CHECKPOINT_PATH" not in os.environ:
    checkpoint_path = Path(__file__).parent / "epoch_50.pt"
    if checkpoint_path.exists():
        os.environ["CHECKPOINT_PATH"] = str(checkpoint_path)
    else:
        print("Warning: epoch_50.pt not found. Please set CHECKPOINT_PATH environment variable.")
        print(f"Looking for checkpoint at: {checkpoint_path}")

if __name__ == "__main__":
    import uvicorn
    
    # Use import string for reload to work properly
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False if you don't need auto-reload
        log_level="info"
    )

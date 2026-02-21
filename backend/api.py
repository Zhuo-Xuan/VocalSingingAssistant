# FastAPI Backend for Melody Transformer
import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import json
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")
    print("Install it with: pip install python-dotenv")

# Import model and helper functions from test script
import sys
scripts_path = str(Path(__file__).parent.parent / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from test_model import (
    MelodyTransformer, extract_features, to_3d, 
    hz_to_note_safe, note_to_midi_safe, normalize, align_length
)

app = FastAPI()

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 1024
OVERLAP = 256
MEL_DIM = 80
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "epoch_50.pt")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Debug: Print API key status (don't print the actual key)
if OPENAI_API_KEY:
    print(f"OpenAI API key loaded: {'*' * (len(OPENAI_API_KEY) - 8)}{OPENAI_API_KEY[-8:]}")
else:
    print("Warning: OPENAI_API_KEY not found in environment variables")
    print("Set it in .env file or as environment variable")

# --- Load model ---
print(f"Loading model from {CHECKPOINT_PATH}...")
model = MelodyTransformer().to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully")
else:
    print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")

# --- Helper Functions ---
def predict_melody(audio_path: str):
    """Extract features and predict melody"""
    mel, f0_norm, energy, f0_hz, f0_mean, f0_std, audio_len, sr = extract_features(audio_path)
    
    # Predict in chunks
    pred_list = []
    start = 0
    while start < mel.shape[0]:
        end = min(start + MAX_FRAMES, mel.shape[0])
        
        mel_chunk = mel[start:end]
        f0_chunk = f0_norm[start:end]
        energy_chunk = energy[start:end]
        
        min_frames = min(mel_chunk.shape[0], f0_chunk.shape[0], energy_chunk.shape[0])
        mel_chunk = mel_chunk[:min_frames]
        f0_chunk = f0_chunk[:min_frames]
        energy_chunk = energy_chunk[:min_frames]
        
        mel_chunk = to_3d(mel_chunk).to(DEVICE)
        f0_chunk = to_3d(f0_chunk).to(DEVICE)
        energy_chunk = to_3d(energy_chunk).to(DEVICE)
        
        with torch.no_grad():
            pred_chunk = model(mel_chunk, f0_chunk, energy_chunk)
        
        pred_list.append(pred_chunk.cpu())
        start += MAX_FRAMES - OVERLAP
    
    # Concatenate predictions
    pred_norm = torch.cat(pred_list, dim=1).squeeze(0).numpy()
    pred_hz = pred_norm * float(f0_std) + float(f0_mean)
    pred_hz_flat = pred_hz.flatten()
    
    # Align to original length
    pred_hz_aligned = np.interp(
        np.linspace(0, len(pred_hz_flat)-1, len(f0_hz)),
        np.arange(len(pred_hz_flat)),
        pred_hz_flat
    )
    
    return pred_hz_aligned, f0_hz, audio_len, sr

def replace_f0_in_audio(audio_path: str, new_f0: np.ndarray, output_path: str):
    """Replace F0 in audio using WORLD vocoder or PSOLA"""
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Get original F0 for comparison
    _, _, _, original_f0, _, _, _, _ = extract_features(audio_path)
    
    # Get frame times
    hop_length = 512
    frame_times = librosa.frames_to_time(np.arange(len(new_f0)), sr=sr, hop_length=hop_length)
    
    # Use pyworld for high-quality F0 replacement
    try:
        import pyworld as pw
        
        # Extract features using WORLD
        f0_contour = new_f0.copy()
        f0_contour[f0_contour <= 0] = 0  # Remove unvoiced
        
        # Get frame indices
        frame_indices = librosa.time_to_frames(frame_times, sr=sr, hop_length=hop_length)
        frame_indices = np.clip(frame_indices, 0, len(f0_contour) - 1)
        
        # Interpolate F0 to sample rate
        time_samples = np.arange(len(y)) / sr
        f0_interp = np.interp(time_samples, frame_times, f0_contour[frame_indices])
        
        # Extract WORLD features
        f0_world = f0_interp.astype(np.float64)
        sp = pw.cheaptrick(y.astype(np.float64), f0_world, sr)
        ap = pw.d4c(y.astype(np.float64), f0_world, sr)
        
        # Synthesize with new F0
        y_synth = pw.synthesize(f0_world, sp, ap, sr)
        
        # Save
        sf.write(output_path, y_synth, sr)
        return True
    except ImportError:
        # Fallback: use librosa's pitch shifting approach
        print("pyworld not available, using pitch shift method")
        # Simple approach: use librosa's pitch shifting
        # This is a simplified version - for production, use WORLD vocoder
        y_shifted = y.copy()
        hop_length_samples = hop_length
        
        # Apply pitch correction frame by frame
        for i in range(len(new_f0) - 1):
            if new_f0[i] > 0 and original_f0[i] > 0:
                ratio = new_f0[i] / original_f0[i] if original_f0[i] > 0 else 1.0
                start_sample = int(i * hop_length_samples)
                end_sample = int((i + 1) * hop_length_samples)
                
                if end_sample < len(y_shifted) and ratio != 1.0:
                    chunk = y_shifted[start_sample:end_sample]
                    if len(chunk) > hop_length:  # Need enough samples for pitch shift
                        try:
                            n_steps = 12 * np.log2(ratio)
                            shifted_chunk = librosa.effects.pitch_shift(
                                chunk, sr=sr, n_steps=n_steps
                            )
                            y_shifted[start_sample:end_sample] = shifted_chunk[:len(chunk)]
                        except:
                            pass  # Skip if pitch shift fails
        
        sf.write(output_path, y_shifted, sr)
        return True

def generate_graph_data(pred_hz: np.ndarray, gt_hz: np.ndarray, audio_len: float, sr: int):
    """Generate graph data for frontend"""
    frame_times = np.linspace(0, audio_len/sr, len(gt_hz))
    
    # Convert to notes
    gt_notes_all = [hz_to_note_safe(f) for f in gt_hz]
    pred_notes_all = [hz_to_note_safe(f) for f in pred_hz]

    # Filter out "Rest" frames so they don't appear on the graph
    keep_indices = [
        i for i, (g, p) in enumerate(zip(gt_notes_all, pred_notes_all))
        if g != "Rest" and p != "Rest"
    ]

    if keep_indices:
        frame_times = frame_times[keep_indices]
        gt_notes = [gt_notes_all[i] for i in keep_indices]
        pred_notes = [pred_notes_all[i] for i in keep_indices]
    else:
        # Fallback: no voiced frames, keep originals (graph will just be empty/flat)
        gt_notes = []
        pred_notes = []
        frame_times = np.array([])

    # Create note mapping without "Rest"
    all_notes = sorted(
        {n for n in gt_notes + pred_notes if n != "Rest"},
        key=note_to_midi_safe
    )
    note_to_int = {n: i for i, n in enumerate(all_notes)}

    gt_int = [note_to_int[n] for n in gt_notes] if note_to_int else []
    pred_int = [note_to_int[n] for n in pred_notes] if note_to_int else []

    return {
        "frame_times": frame_times.tolist(),
        "gt_notes": gt_notes,
        "pred_notes": pred_notes,
        "gt_int": gt_int,
        "pred_int": pred_int,
        "note_labels": all_notes,
        "note_to_int": note_to_int
    }

async def get_ai_feedback(graph_data: dict, accuracy: float) -> str:
    """Generate feedback using OpenAI GPT model."""
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        return "OpenAI API key not set or OpenAI package not installed."

    # Prepare a prompt based on graph data
    prompt = f"""
You are a singing coach. Analyze the following singing performance data and give feedback.
Accuracy: {accuracy:.2f}%
Melody data: {json.dumps(graph_data)}
Give concise, practical advice for improving pitch, timing, and stability. 
"""
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful singing coach."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        feedback = response.choices[0].message.content.strip()
        return feedback
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Failed to generate feedback from OpenAI API."


# --- API Endpoints ---
@app.post("/api/process")
async def process_audio(file: UploadFile = File(...)):
    """Process uploaded audio file"""
    tmp_path = None
    try:
        print(f"Received file: {file.filename}, size: {file.size}")
        
        # Save uploaded file temporarily
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        content = await file.read()
        tmp_file.write(content)
        tmp_file.close()
        tmp_path = tmp_file.name
        
        print(f"Saved temporary file: {tmp_path}")
        
        # Predict melody
        print("Starting melody prediction...")
        pred_hz, gt_hz, audio_len, sr = predict_melody(tmp_path)
        print(f"Prediction complete. Audio length: {audio_len/sr:.2f}s")
        
        # Generate graph data
        print("Generating graph data...")
        graph_data = generate_graph_data(pred_hz, gt_hz, audio_len, sr)
        
        # Calculate accuracy
        accuracy = sum(1 for p, g in zip(graph_data["pred_notes"], graph_data["gt_notes"]) 
                      if p == g) / len(graph_data["gt_notes"]) * 100
        
        # Get ChatGPT feedback
        print("Getting OpenAI feedback...")
        feedback = await get_ai_feedback(graph_data, accuracy)
        
        # Clean up
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        print("Processing complete, returning response")
        return JSONResponse({
            "success": True,
            "graph_data": graph_data,
            "accuracy": accuracy,
            "feedback": feedback
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing audio: {str(e)}")
        print(f"Traceback: {error_trace}")
        
        # Clean up on error
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return JSONResponse({
            "success": False, 
            "error": str(e),
            "traceback": error_trace
        }, status_code=500)

@app.post("/api/replace_f0")
async def replace_f0(file: UploadFile = File(...)):
    """Replace F0 in audio with predicted melody"""
    tmp_input = None
    output_path = None
    try:
        # Save uploaded file
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        content = await file.read()
        tmp_input.write(content)
        tmp_input.close()
        input_path = tmp_input.name
        
        # Predict melody
        pred_hz, gt_hz, audio_len, sr = predict_melody(input_path)
        
        # Replace F0
        output_path = tempfile.mktemp(suffix=".wav")
        replace_f0_in_audio(input_path, pred_hz, output_path)
        
        # Clean up input
        if os.path.exists(input_path):
            os.unlink(input_path)
        
        # Return output file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="processed_melody.wav"
        )
    except Exception as e:
        # Cleanup on error
        if tmp_input and os.path.exists(tmp_input.name):
            os.unlink(tmp_input.name)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/health")
async def health():
    """Health check"""
    return {"status": "ok", "device": str(DEVICE)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

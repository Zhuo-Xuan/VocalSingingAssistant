# FastAPI Backend for Melody Transformer
import os
import base64
import uuid
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

# --- VIDEO SUPPORT IMPORT ---
try:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

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

AI2_OUTPUT_DIR = Path(__file__).parent / "ai2_outputs"
AI2_OUTPUT_DIR.mkdir(exist_ok=True)


class AI2ProcessRequest(BaseModel):
    filename: str
    mime_type: str
    audio_base64: str
    generation_index: int = 1

# --- New Helper Function for Video Extraction ---
def extract_audio_if_video(file_path: str, extension: str) -> (str, bool):
    """
    Checks if the file is a video. If so, extracts audio to a temporary wav.
    Returns (path_to_work_with, was_video_flag)
    """
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.3gp', '.flv']
    
    if extension.lower() not in video_extensions:
        return file_path, False

    if not MOVIEPY_AVAILABLE:
        print("Warning: Video uploaded but MoviePy is not installed.")
        return file_path, False

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav_path = temp_wav.name
    temp_wav.close()

    video = None
    try:
        video = VideoFileClip(file_path)
        if video.audio is None:
            raise ValueError("The uploaded video has no audio track.")
        # Export as wav at 22050Hz to match your model's expected sample rate
        video.audio.write_audiofile(temp_wav_path, fps=22050, verbose=False, logger=None)
        return temp_wav_path, True
    except Exception as e:
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        raise e
    finally:
        if video:
            video.close()

# --- Original Helper Functions ---
def predict_melody(audio_path: str):
    """Extract features and predict melody"""
    mel, f0_norm, energy, f0_hz, f0_mean, f0_std, audio_len, sr = extract_features(audio_path)
    pred_list = []
    start = 0
    while start < mel.shape[0]:
        end = min(start + MAX_FRAMES, mel.shape[0])
        mel_chunk = mel[start:end]
        f0_chunk = f0_norm[start:end]
        energy_chunk = energy[start:end]
        min_frames = min(mel_chunk.shape[0], f0_chunk.shape[0], energy_chunk.shape[0])
        mel_chunk = to_3d(mel_chunk[:min_frames]).to(DEVICE)
        f0_chunk = to_3d(f0_chunk[:min_frames]).to(DEVICE)
        energy_chunk = to_3d(energy_chunk[:min_frames]).to(DEVICE)
        with torch.no_grad():
            pred_chunk = model(mel_chunk, f0_chunk, energy_chunk)
        pred_list.append(pred_chunk.cpu())
        start += MAX_FRAMES - OVERLAP
    pred_norm = torch.cat(pred_list, dim=1).squeeze(0).numpy()
    pred_hz = pred_norm * float(f0_std) + float(f0_mean)
    pred_hz_flat = pred_hz.flatten()
    pred_hz_aligned = np.interp(
        np.linspace(0, len(pred_hz_flat)-1, len(f0_hz)),
        np.arange(len(pred_hz_flat)),
        pred_hz_flat
    )
    return pred_hz_aligned, f0_hz, audio_len, sr

def replace_f0_in_audio(audio_path: str, new_f0: np.ndarray, output_path: str):
    """Replace F0 in audio using WORLD vocoder or PSOLA"""
    y, sr = librosa.load(audio_path, sr=22050)
    _, _, _, original_f0, _, _, _, _ = extract_features(audio_path)
    hop_length = 512
    frame_times = librosa.frames_to_time(np.arange(len(new_f0)), sr=sr, hop_length=hop_length)
    try:
        import pyworld as pw
        f0_contour = new_f0.copy()
        f0_contour[f0_contour <= 0] = 0
        frame_indices = np.clip(librosa.time_to_frames(frame_times, sr=sr, hop_length=hop_length), 0, len(f0_contour) - 1)
        time_samples = np.arange(len(y)) / sr
        f0_interp = np.interp(time_samples, frame_times, f0_contour[frame_indices])
        f0_world = f0_interp.astype(np.float64)
        sp = pw.cheaptrick(y.astype(np.float64), f0_world, sr)
        ap = pw.d4c(y.astype(np.float64), f0_world, sr)
        y_synth = pw.synthesize(f0_world, sp, ap, sr)
        sf.write(output_path, y_synth, sr)
        return True
    except ImportError:
        y_shifted = y.copy()
        for i in range(len(new_f0) - 1):
            if new_f0[i] > 0 and original_f0[i] > 0:
                ratio = new_f0[i] / original_f0[i]
                start_sample, end_sample = int(i * hop_length), int((i + 1) * hop_length)
                if end_sample < len(y_shifted) and ratio != 1.0:
                    try:
                        n_steps = 12 * np.log2(ratio)
                        y_shifted[start_sample:end_sample] = librosa.effects.pitch_shift(y_shifted[start_sample:end_sample], sr=sr, n_steps=n_steps)[:hop_length]
                    except: pass
        sf.write(output_path, y_shifted, sr)
        return True

def generate_graph_data(pred_hz: np.ndarray, gt_hz: np.ndarray, audio_len: float, sr: int):
    frame_times = np.linspace(0, audio_len/sr, len(gt_hz))
    gt_notes_all = [hz_to_note_safe(f) for f in gt_hz]
    pred_notes_all = [hz_to_note_safe(f) for f in pred_hz]
    keep = [i for i, (g, p) in enumerate(zip(gt_notes_all, pred_notes_all)) if g != "Rest" and p != "Rest"]
    if keep:
        frame_times, gt_notes, pred_notes = frame_times[keep], [gt_notes_all[i] for i in keep], [pred_notes_all[i] for i in keep]
    else:
        gt_notes, pred_notes, frame_times = [], [], np.array([])
    all_notes = sorted({n for n in gt_notes + pred_notes if n != "Rest"}, key=note_to_midi_safe)
    note_to_int = {n: i for i, n in enumerate(all_notes)}
    return {
        "frame_times": frame_times.tolist(), "gt_notes": gt_notes, "pred_notes": pred_notes,
        "gt_int": [note_to_int[n] for n in gt_notes], "pred_int": [note_to_int[n] for n in pred_notes],
        "note_labels": all_notes, "note_to_int": note_to_int
    }

async def get_ai_feedback(graph_data: dict, accuracy: float) -> str:
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        return "OpenAI API key not set or OpenAI package not installed."
    prompt = f"Analyze singing performance. Accuracy: {accuracy:.2f}%. Data: {json.dumps(graph_data)}. Give concise advice."
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], max_tokens=300)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {e}"

# --- API Endpoints ---
@app.post("/api/process")
async def process_audio(file: UploadFile = File(...)):
    tmp_path = None
    extracted_wav = None
    try:
        ext = Path(file.filename).suffix.lower()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_file.write(await file.read())
        tmp_file.close()
        tmp_path = tmp_file.name

        # --- VIDEO CHECK ---
        work_path, was_video = extract_audio_if_video(tmp_path, ext)
        if was_video: extracted_wav = work_path

        pred_hz, gt_hz, audio_len, sr = predict_melody(work_path)
        graph_data = generate_graph_data(pred_hz, gt_hz, audio_len, sr)
        accuracy = sum(1 for p, g in zip(graph_data["pred_notes"], graph_data["gt_notes"]) if p == g) / (len(graph_data["gt_notes"]) or 1) * 100
        feedback = await get_ai_feedback(graph_data, accuracy)
        
        return JSONResponse({"success": True, "graph_data": graph_data, "accuracy": accuracy, "feedback": feedback})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        for p in [tmp_path, extracted_wav]:
            if p and os.path.exists(p): os.unlink(p)

@app.post("/api/replace_f0")
async def replace_f0(file: UploadFile = File(...)):
    tmp_input = None
    extracted_wav = None
    output_path = None
    try:
        ext = Path(file.filename).suffix.lower()
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_input.write(await file.read())
        tmp_input.close()

        # --- VIDEO CHECK ---
        work_path, was_video = extract_audio_if_video(tmp_input.name, ext)
        if was_video: extracted_wav = work_path

        pred_hz, _, _, _ = predict_melody(work_path)
        output_path = tempfile.mktemp(suffix=".wav")
        replace_f0_in_audio(work_path, pred_hz, output_path)
        
        return FileResponse(output_path, media_type="audio/wav", filename="processed_melody.wav")
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        for p in [tmp_input.name if tmp_input else None, extracted_wav]:
            if p and os.path.exists(p): os.unlink(p)

@app.get("/api/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.get("/api/ai2/audio/{file_name}")
async def get_ai2_audio(file_name: str):
    safe_name = os.path.basename(file_name)
    file_path = AI2_OUTPUT_DIR / safe_name
    if not file_path.exists(): raise HTTPException(status_code=404)
    return FileResponse(str(file_path), media_type="audio/wav", filename=safe_name)

@app.post("/api/ai2/process_base64")
async def ai2_process_base64(payload: AI2ProcessRequest, request: Request):
    tmp_input = None
    extracted_wav = None
    try:
        audio_b64 = payload.audio_base64.split(",")[-1]
        audio_bytes = base64.b64decode(audio_b64)
        ext = Path(payload.filename).suffix.lower() or ".wav"
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_input.write(audio_bytes)
        tmp_input.close()

        # --- VIDEO CHECK ---
        work_path, was_video = extract_audio_if_video(tmp_input.name, ext)
        if was_video: extracted_wav = work_path

        pred_hz, gt_hz, audio_len, sr = predict_melody(work_path)
        graph_data = generate_graph_data(pred_hz, gt_hz, audio_len, sr)
        total = len(graph_data["gt_notes"]) if graph_data["gt_notes"] else 1
        accuracy = sum(1 for p, g in zip(graph_data["pred_notes"], graph_data["gt_notes"]) if p == g) / total * 100
        feedback = await get_ai_feedback(graph_data, accuracy)

        out_name = f"predicted_{uuid.uuid4().hex}.wav"
        out_path = AI2_OUTPUT_DIR / out_name
        replace_f0_in_audio(work_path, pred_hz, str(out_path))

        base_url = str(request.base_url).rstrip("/")
        return JSONResponse({
            "success": True, "accuracy": accuracy, "feedback": feedback,
            "generation": {"current": max(payload.generation_index, 1), "max": 5, "remaining": max(5 - payload.generation_index, 0)},
            "graph": {"type": "graph_data", "data": graph_data},
            "audio": {"type": "url", "predicted_audio_url": f"{base_url}/api/ai2/audio/{out_name}"}
        })
    except Exception as e:
        return JSONResponse({"success": False, "error_message": str(e)}, status_code=500)
    finally:
        for p in [tmp_input.name if tmp_input else None, extracted_wav]:
            if p and os.path.exists(p): os.unlink(p)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

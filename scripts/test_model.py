# ===============================================
# Melody Transformer Testing Pipeline - Local Version
# With Wrong Prediction Highlighting
# ===============================================

import torch
import torch.nn as nn
import numpy as np
import librosa
import plotly.graph_objects as go
import argparse
import os
from pathlib import Path

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
MAX_FRAMES = 1024
OVERLAP = 256
MEL_DIM = 80  # must match training

# --- Model definition ---
class MelodyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(MEL_DIM + 2, 256)  # mel + f0 + energy
        layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.out_proj = nn.Linear(256, 1)

    def forward(self, mel, f0, energy):
        x = torch.cat([mel, f0, energy], dim=-1)
        x = self.in_proj(x)
        x = self.encoder(x)
        return self.out_proj(x)

# --- Helper functions ---
def normalize(x):
    mean = x.mean()
    std = x.std() + 1e-6
    return (x - mean) / std, mean, std

def align_length(ref, target):
    ref_len = len(ref)
    target_len = len(target)
    if target_len == ref_len:
        return target
    x_old = np.arange(target_len)
    x_new = np.linspace(0, target_len-1, ref_len)
    return np.interp(x_new, x_old, target)

def extract_features(audio_path, sr=22050, n_mels=MEL_DIM):
    y, _ = librosa.load(audio_path, sr=sr)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel = librosa.power_to_db(mel).T  # (frames, n_mels)

    # F0 using pyin
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    f0 = np.nan_to_num(f0)

    # Energy (RMS)
    energy = librosa.feature.rms(y=y).T.squeeze()
    f0 = align_length(mel[:,0], f0)
    energy = align_length(mel[:,0], energy)

    # Torch tensors
    mel = torch.tensor(mel, dtype=torch.float32)
    f0 = torch.tensor(f0, dtype=torch.float32).unsqueeze(-1)
    energy = torch.tensor(energy, dtype=torch.float32).unsqueeze(-1)

    # Normalize
    mel_norm, mel_mean, mel_std = normalize(mel)
    f0_norm, f0_mean, f0_std = normalize(f0)
    energy_norm, energy_mean, energy_std = normalize(energy)

    return mel_norm, f0_norm, energy_norm, f0.squeeze(-1).numpy(), f0_mean, f0_std, y.shape[0], sr

def to_3d(x):
    if x.ndim == 2:
        return x.unsqueeze(0)
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(f"Unexpected tensor shape {x.shape}")

def hz_to_note_safe(f):
    f_scalar = float(np.squeeze(f))
    if np.isnan(f_scalar) or np.isinf(f_scalar) or f_scalar <= 0:
        return 'Rest'
    return librosa.hz_to_note(f_scalar, octave=True)

def note_to_midi_safe(note):
    if note == 'Rest':
        return -1
    return librosa.note_to_midi(note)

def main():
    parser = argparse.ArgumentParser(description='Test Melody Transformer model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., epoch_50.pt)')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: Save plot as HTML file')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    
    # --- Load model ---
    print(f"Loading model from {args.checkpoint}...")
    model = MelodyTransformer().to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint.get('epoch', 'unknown')
        else:
            # Assume it's just the state dict
            model.load_state_dict(checkpoint)
            epoch = 'unknown'
    else:
        model.load_state_dict(checkpoint)
        epoch = 'unknown'
    
    model.eval()
    print(f"Loaded model, epoch: {epoch}")
    
    # --- Extract features ---
    print(f"Extracting features from {args.audio}...")
    mel, f0_norm, energy, f0_hz, f0_mean, f0_std, audio_len, sr = extract_features(args.audio)
    
    # --- Predict in chunks ---
    print("Running inference...")
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

    # --- Concatenate predicted chunks ---
    pred_norm = torch.cat(pred_list, dim=1).squeeze(0).numpy()
    pred_hz = pred_norm * float(f0_std) + float(f0_mean)
    pred_hz_flat = pred_hz.flatten()

    # --- Align predicted to ground truth length ---
    pred_hz_aligned = np.interp(
        np.linspace(0, len(pred_hz_flat)-1, len(f0_hz)),
        np.arange(len(pred_hz_flat)),
        pred_hz_flat
    )

    # --- Frame times ---
    frame_times = np.linspace(0, audio_len/sr, len(f0_hz))

    # --- Convert Hz to note names ---
    print("Converting to notes...")
    gt_notes = [hz_to_note_safe(f) for f in f0_hz]
    pred_notes = [hz_to_note_safe(f) for f in pred_hz_aligned]

    # --- Y-axis ordering by MIDI (octave) ---
    all_notes_in_clip = sorted(set(gt_notes + pred_notes), key=note_to_midi_safe)
    note_to_int_clip = {n:i for i,n in enumerate(all_notes_in_clip)}

    gt_int_clip = [note_to_int_clip[n] for n in gt_notes]
    pred_int_clip = [note_to_int_clip[n] for n in pred_notes]

    # --- Determine which predictions are correct ---
    pred_colors = ['orange' if p == g else 'red' for p, g in zip(pred_int_clip, gt_int_clip)]
    
    # Calculate accuracy
    accuracy = sum(1 for p, g in zip(pred_int_clip, gt_int_clip) if p == g) / len(gt_int_clip) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # --- Plotly interactive dot plot ---
    print("Generating visualization...")
    fig = go.Figure()

    # Ground truth (blue)
    fig.add_trace(go.Scatter(
        x=frame_times,
        y=gt_int_clip,
        mode='markers',
        name='Ground Truth',
        marker=dict(color='blue', size=6),
        text=gt_notes,
        hovertemplate='Time: %{x:.2f}s<br>Note: %{text}'
    ))

    # Predicted (orange=correct, red=wrong)
    fig.add_trace(go.Scatter(
        x=frame_times,
        y=pred_int_clip,
        mode='markers',
        name='Predicted',
        marker=dict(color=pred_colors, size=6),
        text=pred_notes,
        hovertemplate='Time: %{x:.2f}s<br>Note: %{text}'
    ))

    fig.update_layout(
        title=f'Melody Prediction vs Ground Truth (Accuracy: {accuracy:.2f}%)',
        xaxis_title='Time (s)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(note_to_int_clip.values()),
            ticktext=list(note_to_int_clip.keys())
        ),
        legend=dict(x=0.01, y=0.99),
        height=600
    )

    # Save or show
    if args.output:
        fig.write_html(args.output)
        print(f"Plot saved to {args.output}")
    else:
        fig.show()

if __name__ == "__main__":
    main()

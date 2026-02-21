import os
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ======================
# CONFIG
# ======================
RAW_DIR = r"D:\musicAI\raw"
OUT_DIR = r"D:\musicAI\features"
SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_MELS = 80

ENERGY_TH = 1e-4
MIN_VOICED_FRAMES = 10

MAX_WORKERS = max(os.cpu_count() - 1, 1)

GOOD_DIR = os.path.join(OUT_DIR, "good")
os.makedirs(GOOD_DIR, exist_ok=True)

# ======================
# UTILS
# ======================
def find_wavs(root):
    wavs = []
    for path, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(path, f))
    return sorted(wavs)

def clean_name(wav_path):
    rel_path = os.path.relpath(wav_path, RAW_DIR)
    name = rel_path.replace(os.sep, "_")
    name = os.path.splitext(name)[0] + ".npz"
    return name

def extract_features(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        if len(y) < 100:
            return None

        energy = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        if np.max(energy) < ENERGY_TH:
            return None

        # f0 using pyin
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=100, fmax=2000, sr=sr,
            frame_length=HOP_LENGTH*2, hop_length=HOP_LENGTH
        )
        f0 = np.nan_to_num(f0)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel = librosa.power_to_db(mel).T

        min_len = min(len(f0), len(energy), mel.shape[0])
        f0 = f0[:min_len]
        energy = energy[:min_len]
        mel = mel[:min_len]

        mask = energy > ENERGY_TH
        if mask.sum() < MIN_VOICED_FRAMES:
            return None

        return {
            "mel": mel[mask],
            "energy": energy[mask],
            "f0": f0[mask]
        }

    except Exception as e:
        print(f"Skipped {wav_path}: {e}")
        return None

def process_one(wav_path):
    feats = extract_features(wav_path)
    if feats is None:
        return None

    out_path = os.path.join(GOOD_DIR, clean_name(wav_path))
    np.savez_compressed(out_path, **feats)
    return True

# ======================
# RESUME LOGIC
# ======================
def find_resume_wav():
    # last file you want to resume from
    last_file = r"D:\musicAI\raw\GTSinger\German\German\DE-Soprano-1\Pharyngeal\Ich Will Immer Wieder Dieses Fieber spu╠êrΓÇÿn\Paired_Speech_Group\0001.wav"
    return last_file

# ======================
# MAIN
# ======================
def main():
    wav_files = find_wavs(RAW_DIR)

    # resume from last file
    resume_file = find_resume_wav()
    if resume_file in wav_files:
        idx = wav_files.index(resume_file)
        wav_files = wav_files[idx:]  # continue from here
    else:
        print("Resume file not found, starting from beginning.")

    print(f"Total WAVs to process: {len(wav_files)}")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one, w) for w in wav_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing WAVs"):
            results.append(f.result())

    print(f"Done. {len([r for r in results if r])} files processed.")

if __name__ == "__main__":
    main()

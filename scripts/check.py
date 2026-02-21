import numpy as np
import os
from tqdm import tqdm

# ===== CONFIG =====
FEATURES_DIR = r"features"  # root folder containing 'good' and 'bad' subfolders

def check_npz_files(root_dir):
    missing_f0 = []

    # First, collect all npz files
    npz_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(subdir, file))

    print(f"Total npz files found: {len(npz_files)}")

    # Iterate with progress bar
    for path in tqdm(npz_files, desc="Checking npz files"):
        try:
            data = np.load(path)
            if "f0" not in data:
                missing_f0.append(path)
        except Exception as e:
            print(f"\nFailed to load {path}: {e}")
            missing_f0.append(path)

    if missing_f0:
        for f in missing_f0:
            print(f)
        print("\nNPZ files missing 'f0':")
        print(len(missing_f0))
    else:
        print("All npz files contain 'f0'.")

if __name__ == "__main__":
    check_npz_files(FEATURES_DIR)


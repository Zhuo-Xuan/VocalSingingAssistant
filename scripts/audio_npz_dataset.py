import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioNPZDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".npz"):
                    self.files.append(os.path.join(root, f))

        print(f"[Dataset] Found {len(self.files)} npz files in {root_dir}")
        if len(self.files) == 0:
            raise RuntimeError("No .npz files found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        mel = torch.tensor(data["mel"], dtype=torch.float32)
        f0 = torch.tensor(data["f0"], dtype=torch.float32)
        return mel, f0

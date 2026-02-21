import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SingingDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = sorted([
            os.path.join(feature_dir, f)
            for f in os.listdir(feature_dir)
            if f.endswith(".npz")
        ])
        if len(self.files) == 0:
            raise RuntimeError("No .npz files found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        mel = torch.from_numpy(data["mel"]).float()   # (T, 80)
        f0  = torch.from_numpy(data["f0"]).float()    # (T,)

        # log-scale pitch (stable regression)
        f0 = torch.log1p(f0)

        return mel, f0


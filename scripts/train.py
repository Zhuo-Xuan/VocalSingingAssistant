# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SingingDataset
from model import PitchTransformer

# ======================
# CONFIG
# ======================
DATA_DIR = r"features/good"
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
SAVE_EVERY = 5
CKPT_PATH = "checkpoint.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# COLLATE FUNCTION
# ======================
def collate_fn(batch):
    mels, f0s = zip(*batch)

    lengths = [m.shape[0] for m in mels]
    max_len = max(lengths)

    B = len(batch)
    mel_dim = mels[0].shape[1]

    mel_pad = torch.zeros(B, max_len, mel_dim)
    f0_pad = torch.zeros(B, max_len)
    mask = torch.ones(B, max_len, dtype=torch.bool)  # True = padding

    for i, (mel, f0) in enumerate(zip(mels, f0s)):
        T = mel.shape[0]
        mel_pad[i, :T] = mel
        f0_pad[i, :T] = f0
        mask[i, :T] = False

    return mel_pad, f0_pad, mask

# ======================
# MAIN
# ======================
def main():
    dataset = SingingDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = PitchTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    start_epoch = 1

    # Resume
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for mel, f0, mask in pbar:
            mel = mel.to(DEVICE)
            f0 = f0.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(mel, mask)

            loss = criterion(pred[~mask], f0[~mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f}")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict()
            }, CKPT_PATH)
            print(f"Checkpoint saved at epoch {epoch}")

if __name__ == "__main__":
    main()
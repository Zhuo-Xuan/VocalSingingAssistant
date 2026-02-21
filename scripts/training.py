import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

# -------------------------
# Configuration
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
MAX_FRAMES = 1024
OVERLAP = 256
EPOCHS = 50
LR = 1e-4

# Set your local paths here
SAVE_DIR = "./melody_training_logs"
DATA_DIR = "./features_good"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure dataset folder exists

# -------------------------
# Dataset
# -------------------------
class MelF0EnergyDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.files = []
        for r, _, f in os.walk(root):
            for x in f:
                if x.endswith(".npz"):
                    self.files.append(os.path.join(r, x))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        mel = torch.tensor(data["mel"], dtype=torch.float32)
        f0 = torch.tensor(data["f0"], dtype=torch.float32).unsqueeze(-1)
        energy = torch.tensor(data["energy"], dtype=torch.float32).unsqueeze(-1)
        return mel, f0, energy


# -------------------------
# Model
# -------------------------
class MelodyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(82, 256)
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.out_proj = nn.Linear(256, 1)

    def forward(self, mel, f0, energy):
        x = torch.cat([mel, f0, energy], dim=-1)
        x = self.in_proj(x)
        x = self.encoder(x)
        return self.out_proj(x)


# -------------------------
# Utility functions
# -------------------------
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


# -------------------------
# Training
# -------------------------
def train():
    dataset = MelF0EnergyDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MelodyTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    loss_history = []
    abnormal_log = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        valid_chunks = 0

        for mel, f0, energy in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="track"):
            mel = normalize(mel[0]).to(DEVICE)
            f0 = normalize(f0[0]).to(DEVICE)
            energy = normalize(energy[0]).to(DEVICE)

            start = 0
            optimizer.zero_grad()

            while start < mel.shape[0]:
                end = min(start + MAX_FRAMES, mel.shape[0])

                mel_c = mel[start:end].unsqueeze(0)
                f0_c = f0[start:end].unsqueeze(0)
                energy_c = energy[start:end].unsqueeze(0)

                pred = model(mel_c, f0_c, energy_c)
                loss = criterion(pred, f0_c)

                if torch.isnan(loss) or loss.item() > 1e6:
                    abnormal_log.append({
                        "epoch": epoch + 1,
                        "loss": float(loss.item())
                    })
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                epoch_loss += loss.item()
                valid_chunks += 1

                start += MAX_FRAMES - OVERLAP

            optimizer.step()

        avg_loss = epoch_loss / max(valid_chunks, 1)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Save logs
        np.save(os.path.join(SAVE_DIR, "train_loss.npy"), np.array(loss_history))
        np.save(os.path.join(SAVE_DIR, "abnormal_events.npy"), np.array(abnormal_log, dtype=object))

        # Save visualization
        model.eval()
        with torch.no_grad():
            mel_v, f0_v, energy_v = dataset[0]
            mel_v = normalize(mel_v).unsqueeze(0).to(DEVICE)
            f0_v = normalize(f0_v).unsqueeze(0).to(DEVICE)
            energy_v = normalize(energy_v).unsqueeze(0).to(DEVICE)

            pred_v = model(mel_v, f0_v, energy_v)

        np.savez(
            os.path.join(SAVE_DIR, f"viz_epoch_{epoch+1}.npz"),
            mel=mel_v.cpu().numpy(),
            gt_f0=f0_v.cpu().numpy(),
            pred_f0=pred_v.cpu().numpy(),
            energy=energy_v.cpu().numpy()
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss
            },
            os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt")
        )

    print("Training complete.")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    train()
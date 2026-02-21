# model.py
import torch
import torch.nn as nn

class PitchTransformer(nn.Module):
    def __init__(
        self,
        mel_dim=80,
        d_model=256,
        nhead=4,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(mel_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, mel, pad_mask):
        """
        mel: (B, T, 80)
        pad_mask: (B, T)  True = padding
        """
        x = self.input_proj(mel)          # (B, T, D)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        out = self.output_proj(x).squeeze(-1)  # (B, T)
        return out

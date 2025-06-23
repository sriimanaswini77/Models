import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class TransformerModel3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, max_seq_len=30, dropout=0.1):
        super().__init__()  # ✅ Corrected

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True  # ✅ Avoids permute
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        padding_mask = (x.abs().sum(dim=-1) == 0)  # [batch_size, seq_len]

        x = self.input_proj(x)          # [B, S, D]
        x = self.pos_encoder(x)         # [B, S, D]
        x = self.transformer_encoder(x) # [B, S, D] — no padding mask used

        # Masked mean pooling to exclude padded parts
        mask = (~padding_mask).unsqueeze(-1).type_as(x)  # [B, S, 1]
        x = x * mask
        summed = x.sum(dim=1)              # [B, D]
        lengths = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = summed / lengths          # [B, D]

        out = self.decoder(pooled).squeeze(-1)  # [B]
        return out

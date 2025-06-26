import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(d_model, max_len)  # [d_model, max_len]
        position = torch.arange(0, max_len).unsqueeze(0).float()  # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[0::2, :] = torch.sin(position * div_term.unsqueeze(1))
        pe[1::2, :] = torch.cos(position * div_term.unsqueeze(1))
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, d_model, seq_len]
        seq_len = x.size(2)
        return x + self.pe[:, :seq_len].unsqueeze(0)  # broadcast to [1, d_model, seq_len]


class SpacetimeformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(SpacetimeformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Spacetimeformer(nn.Module):
    def __init__(self, d_x, d_y, context_len, d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1):
        super(Spacetimeformer, self).__init__()
        self.input_proj = nn.Conv1d(d_x, d_model, kernel_size=1)  # [B, d_x, T] -> [B, d_model, T]
        self.pos_encoder = PositionalEncoding(d_model, max_len=context_len)

        self.encoder_layers = nn.ModuleList([
            SpacetimeformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
        ])

        self.output_proj = nn.Linear(d_model, d_y)

    def forward(self, x):
        # x: [batch, d_x, seq_len]
        x = self.input_proj(x)            # [batch, d_model, seq_len]
        x = self.pos_encoder(x)           # [batch, d_model, seq_len]
        x = x.transpose(1, 2)             # [batch, seq_len, d_model] for transformer layers

        for layer in self.encoder_layers:
            x = layer(x)

        x_last = x[:, -1, :]              # Take the last time step
        output = self.output_proj(x_last)  # [batch, d_y]
        return output

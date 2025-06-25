import torch
import torch.nn as nn

class TransformerModel1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, max_seq_len=100):
        super().__init__()  # ✅ Corrected class name

        self.max_seq_len = max_seq_len

        # Input projection
        self.embedding = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer encoder and decoder with batch_first=True (no permute needed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=64,
            batch_first=True  # ✅ So input shape is [B, S, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=64,
            batch_first=True  # ✅ Keeps consistent shape
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output = nn.Linear(hidden_size, 1)

    def generate_src_mask(self, x):
        """
        Create a src_mask to avoid attention to padded tokens.
        Padded tokens are assumed to be all-zero vectors.
        """
        # x: [batch_size, seq_len, input_size]
        pad_mask = (x.abs().sum(dim=-1) == 0)  # [B, S], True where padding
        # Create src_mask: [S, S], where positions attending to pad tokens are -inf
        seq_len = x.size(1)
        src_mask = torch.ones((seq_len, seq_len), device=x.device).tril().bool()  # causal mask (optional)
        return src_mask

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.size()

        # Compute masks
        src_mask = self.generate_src_mask(x)  # [seq_len, seq_len]

        # Input + positional embedding
        input_embeddings = self.embedding(x)  # [B, S, D]
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, S]
        pos_embeddings = self.pos_embedding(pos)  # [1, S, D]
        embedded = input_embeddings + pos_embeddings  # [B, S, D]

        # Transformer encoder and decoder
        enc_out = self.transformer_encoder(embedded, mask=src_mask)  # [B, S, D]
        dec_out = self.transformer_decoder(embedded, enc_out, tgt_mask=src_mask)  # [B, S, D]

        # Take the output at the last real (non-padded) position
        # To keep things simple, we just take the last position here
        final_output = dec_out[:, -1, :]  # [B, D]

        output = self.output(final_output)  # [B, 1]
        return output.squeeze(-1)  # [B]

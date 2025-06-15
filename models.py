import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, max_seq_len=100):
        super(TransformerModel, self).__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding layer
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=64,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        # Transformer decoder
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=64,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,
            num_layers=num_layers
        )

        # Final regression output (single scalar)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        seq_len = x.size(1)

        # Compute input embeddings
        input_embeddings = self.embedding(x)  # (batch_size, seq_len, hidden_size)

        # Generate positional encodings
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_embeddings = self.pos_embedding(pos)  # (1, seq_len, hidden_size)

        # Combine input embeddings and positional encodings
        embedded = input_embeddings + pos_embeddings  # (batch_size, seq_len, hidden_size)

        # Reshape for transformer: (seq_len, batch_size, hidden_size)
        embedded = embedded.permute(1, 0, 2)

        # Encoder
        enc_out = self.transformer_encoder(embedded)  # (seq_len, batch_size, hidden_size)

        # Decoder
        dec_out = self.transformer_decoder(embedded, enc_out)  # (seq_len, batch_size, hidden_size)

        # Use the last time step's output
        final_output = dec_out[-1]  # (batch_size, hidden_size)

        # Final regression layer
        output = self.output(final_output)  # (batch_size, 1)

        return output.squeeze(-1)  # (batch_size,)

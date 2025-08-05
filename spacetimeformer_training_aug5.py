import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= Model =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class Spacetimeformer(nn.Module):
    def __init__(self, input_dim, time_len, var_len, emb_dim=128, num_heads=4, num_layers=3, dropout=0.1, out_dim=12):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.var_len = var_len
        self.time_len = time_len
        self.emb_dim = emb_dim

        self.time_pos_emb = PositionalEncoding(emb_dim, max_len=time_len)
        self.var_pos_emb = nn.Parameter(torch.randn(1, var_len, emb_dim))

        encoder_layer_time = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer_time, num_layers=num_layers)

        encoder_layer_var = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer_var, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, x):
        B, T, V = x.shape
        x_time = self.input_proj(x)
        x_time = self.time_pos_emb(x_time)
        x_time = self.temporal_transformer(x_time)

        x_var = x.permute(0, 2, 1)
        x_var = self.input_proj(x_var) + self.var_pos_emb[:, :V, :]
        x_var = self.spatial_transformer(x_var)

        x_fused = (x_time[:, -1, :] + x_var.mean(dim=1)) / 2
        return self.decoder(x_fused)

# ============== Loss Functions ==============
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + 1e-6)

class RMSPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean(((y_true - y_pred) / (y_true + 1e-4)) ** 2)) * 100

# ============== Data Processing ==============
def sliding_windows(data, labels, max_seq_length=30, stride=1):
    X, Y = [], []
    for start in range(0, len(data) - max_seq_length - 12 + 1, stride):
        X.append(data[start:start + max_seq_length])
        Y.append(labels[start + max_seq_length:start + max_seq_length + 12])
    return np.array(X), np.array(Y)

def load_data(data_folder, input_size, max_seq_length=30, stride=1):
    file_paths = glob.glob(f"{data_folder}/*.csv")
    X_data, Y_data = [], []

    for path in file_paths:
        df = pd.read_csv(path, dtype=np.float32)
        features = df.iloc[:, [0, 2, 3, 4, 5, 6, 7]].values
        target = df.iloc[:, 20].values
        x, y = sliding_windows(features, target, max_seq_length, stride)
        X_data.extend(x)
        Y_data.extend(y)

    X = np.array(X_data)
    Y = np.array(Y_data)
    x_min = X.min(axis=(0, 1))
    x_max = X.max(axis=(0, 1))
    x_scaled = (X - x_min) / (x_max - x_min + 1e-6)
    y_mean = Y.mean()
    y_std = Y.std()
    y_scaled = (Y - y_mean) / y_std

    return x_scaled, y_scaled, x_min, x_max, y_mean, y_std

# ============== Training =================
def train():
    input_size = 7
    max_seq_length = 30
    model = Spacetimeformer(input_dim=7, time_len=30, var_len=7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Loading data...")
    X, Y, x_min, x_max, y_mean, y_std = load_data("data", input_size, max_seq_length)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).float())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    criterion = RMSELoss()
    loss2 = RMSPELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("run_logs", exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(100):
        model.train()
        train_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/100"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb) + 0.3 * loss2(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb) + 0.3 * loss2(pred, yb)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'x_min': x_min,
                'x_max': x_max,
                'y_mean': y_mean,
                'y_std': y_std
            }, "run_logs/best_model.pt")
            print(f"✅ Saved best model at epoch {epoch+1}")

if __name__ == "__main__":
    train()

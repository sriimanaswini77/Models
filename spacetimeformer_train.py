from models import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import glob
from sklearn.model_selection import train_test_split
import time
from spacetimeformer.spacetimeformer_model.nn import Spacetimeformer

# Create output directory
directory = "exp_new"
os.makedirs(directory, exist_ok=True)
file_index = 1
while os.path.isdir(f"{directory}/run{file_index}"):
    file_index += 1
os.mkdir(f"{directory}/run{file_index}")

# Hyperparameters
seq_length = 20
stride = 1
input_size = 7
hidden_size = 32
num_layers = 1
num_classes = 1  # one regression target
delta = 1e-6
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Load and prepare data
file_paths = glob.glob("data_30/*")
X_data, Y_data = [], []

def sliding_windows(data, seq_length, stride):
    x = []
    for i in range(0, len(data) - seq_length + 1, stride):
        _x = data[i: i + seq_length]
        x.append(_x)
    return np.array(x)

for path in file_paths:
    df = pd.read_csv(path, dtype=np.float32)
    features = df.iloc[:, [0, 2, 3, 4, 5, 6, 7]]
    y = df.iloc[seq_length - 1::9, 21].values
    x = sliding_windows(features.values, seq_length, stride)
    X_data.append(x)
    Y_data.append(y)

X_data = np.concatenate(X_data, axis=0)
Y_data = np.concatenate(Y_data, axis=0)

x_mean, x_std = np.mean(X_data.reshape(-1, input_size), axis=0), np.std(X_data.reshape(-1, input_size), axis=0)
x_min, x_max = np.min(X_data.reshape(-1, input_size), axis=0), np.max(X_data.reshape(-1, input_size), axis=0)
y_mean, y_std = np.mean(Y_data), np.std(Y_data)
Y_data = (Y_data - y_mean) / y_std

X_train, X_eval, y_train, y_eval = train_test_split(X_data, Y_data, test_size=0.2, random_state=42, shuffle=True)
X_train_scaled = (X_train - x_min) / (x_max - x_min + delta)
X_eval_scaled = (X_eval - x_min) / (x_max - x_min + delta)

X_train = X_train_scaled.reshape(X_train.shape[0], seq_length, input_size)
X_eval = X_eval_scaled.reshape(X_eval.shape[0], seq_length, input_size)

featuresTrain = torch.from_numpy(X_train).float()
featuresEval = torch.from_numpy(X_eval).float()
targetsTrain = torch.from_numpy(y_train).float()
targetsEval = torch.from_numpy(y_eval).float()

train_dataset = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
eval_dataset = torch.utils.data.TensorDataset(featuresEval, targetsEval)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Create Spacetimeformer model
def create_spacetimeformer(input_size, seq_length):
    return Spacetimeformer(
        d_x=input_size,
        d_y=1,
        context_len=seq_length,
        start_token_len=0,
        embed_method="spatio-temporal",
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=2,
        d_ff=128,
        dropout_emb=0.1,
        dropout_qkv=0.1,
        dropout_ff=0.1,
        global_self_attn="performer",
        local_self_attn="none",
        max_seq_len=seq_length,
    )

model = create_spacetimeformer(input_size, seq_length)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.HuberLoss()

best_val_loss = float('inf')
print(f"RESULTS WILL BE SAVED IN {directory}/run{file_index}")

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

    for i, (features, labels) in loop:
        x_context = features.permute(0, 2, 1)  # [B, N, T]

        outputs = model(x_context=x_context)  # [B, N, 1]
        y_pred = outputs[:, 0, 0]  # Select output of first variable

        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for features, labels in val_loader:
            x_context = features.permute(0, 2, 1)
            outputs = model(x_context=x_context)
            y_pred = outputs[:, 0, 0]
            val_loss = criterion(y_pred, labels)
            val_losses.append(val_loss.item())

    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch} Train Loss: {loss.item():.4f} Val Loss: {val_loss:.4f}")

    ckpt = {
        'state_dict': model.state_dict(),
        'model': model,
        'train_loss': loss.item(),
        'val_loss': val_loss,
        'epoch': epoch,
        'input_means': x_mean,
        'output_means': y_mean,
        'input_std': x_std,
        'output_std': y_std,
        'input_min': x_min,
        'input_max': x_max
    }

    torch.save(ckpt, f"{directory}/run{file_index}/ckpt_{epoch}.pt")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(ckpt, f"{directory}/run{file_index}/best_spacetimeformer.pt")
        print("Saved new best model.")

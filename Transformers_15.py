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

# ========== Setup ==========
file_index = 1
while os.path.isdir("exp_new/run%s" % file_index):
    file_index = file_index + 1
os.mkdir(f"exp_new/run{file_index}")

# ========== Parameters ==========
input_size = 7
hidden_size = 32
num_layers = 1
num_classes = 12
delta = 1e-6
max_seq_length = 30
stride = 1

# ========== Sliding Window Logic ==========
def sliding_windows(data, max_seq_length=30, stride=1):
    X = []
    c = 0

    # Phase 1: Growing window
    for seq_length in range(1, max_seq_length + 1):
        _X = data[c:c + seq_length]
        X.append(_X)

    # Phase 2: Fixed-size sliding window
    while c + max_seq_length < len(data):
        c += stride
        _X = data[c:c + max_seq_length]
        X.append(_X)

    return X

def generate_y_labels(labels, max_seq_length=30, stride=1):
    y = []
    c = 0

    # Phase 1: Growing window
    for seq_length in range(1, max_seq_length + 1):
        _Y = labels[seq_length - 1]
        y.append(_Y)

    # Phase 2: Fixed-size sliding window
    while c + max_seq_length < len(labels):
        c += stride
        _Y = labels[c + max_seq_length - 1]
        y.append(_Y)

    return np.array(y)

# ========== Data Loading ==========
file_paths = glob.glob("data_30/*")
print(file_paths)

X_data = []
Y_data = []

for i in range(len(file_paths)):
    training_set = pd.read_csv(file_paths[i], dtype=np.float32)
    features = training_set.iloc[:, [0, 2, 3, 4, 5, 6, 7]].values
    labels = training_set.iloc[:, 20].values

    x = sliding_windows(features, max_seq_length=max_seq_length, stride=stride)
    y = generate_y_labels(labels, max_seq_length=max_seq_length, stride=stride)

    # Pad variable length sequences
    x_padded = [np.pad(seq, ((0, max_seq_length - seq.shape[0]), (0, 0)), mode='constant') for seq in x]

    X_data.extend(x_padded)
    Y_data.extend(y)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

# ========== Normalization ==========
x_mean = np.mean(X_data.reshape(-1, input_size), axis=0)
x_std = np.std(X_data.reshape(-1, input_size), axis=0)
x_min = np.min(X_data.reshape(-1, input_size), axis=0)
x_max = np.max(X_data.reshape(-1, input_size), axis=0)
print(x_mean, x_std)

y_mean = np.mean(Y_data)
y_std = np.std(Y_data)

Y_data = (Y_data - y_mean) / y_std

# ========== Train/Test Split ==========
from sklearn.model_selection import train_test_split
X_train, X_eval, y_train, y_eval = train_test_split(X_data, Y_data, test_size=0.2, random_state=42, shuffle=True)
print(X_train.shape, X_eval.shape, y_train.shape)

# ========== Scaling Features ==========
X_train_scaled = (X_train - x_min) / (x_max - x_min + delta)
X_eval_scaled = (X_eval - x_min) / (x_max - x_min + delta)
print("min, max",x_max,x_min)
print("values",np.min(X_train_scaled,axis=0),np.max(X_train_scaled,axis=0))

X_train = X_train_scaled.reshape(X_train.shape[0], max_seq_length, input_size)
X_eval = X_eval_scaled.reshape(X_eval.shape[0], max_seq_length, input_size)

featuresTrain = torch.from_numpy(X_train)
featuresEval = torch.from_numpy(X_eval)
targetsTrain = torch.from_numpy(y_train)
targetsEval = torch.from_numpy(y_eval)

# ========== Model Setup ==========
learning_rate = 0.001
criterion = torch.nn.HuberLoss()

model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, max_seq_len=max_seq_length)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

batch_size = 32
n_iters = int(len(X_train) / batch_size)
num_epochs = 100

print('Epochs : {} Batchsize: {} Iterations :{}'.format(num_epochs, batch_size, n_iters))

train_dataset = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
eval_dataset = torch.utils.data.TensorDataset(featuresEval, targetsEval)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
torch.manual_seed(1)

def L2_penalty(model):
    return sum(p.pow(2).sum() for p in model.parameters())

# ========== Training Loop ==========
import time
count = 0
loss_list = []
train_losses_list = []
val_losses_list = []

best_val_loss = float('inf')
best_model_state_dict = None
best_epoch = 0

print(f"RESULTS WILL BE SAVED IN exp_new/run{file_index}")

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

    for i, (features, labels) in loop:
        train = features
        labels = labels.squeeze()

        optimizer.zero_grad()
        outputs = model(train)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

        count += 1
        if count % 1 == 0:
            correct = 0
            total = 0
            err = 0

        loss_list.append(loss.data)

    train_loss = sum(loss_list) / len(loss_list)
    train_losses_list.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_losses = []
        for i, (features, labels) in enumerate(val_loader):
            val_x = features
            y = labels.squeeze()

            y_pred = model(val_x)
            y_pred = y_pred.squeeze()
            val_loss = criterion(y_pred, y)
            val_losses.append(val_loss.item())

        val_loss = sum(val_losses) / len(val_losses)
        val_losses_list.append(val_loss)

        if epoch % 1 == 0:
            print('Epoch :{} Train_loss : {} Val_loss : {:.6f}'.format(epoch, train_loss, val_loss))
            torch.save({
                'state_dict': model.state_dict(),
                'model': model,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': f"{epoch+1}/{num_epochs}",
                'input_means': x_mean,
                'output_means': y_mean,
                'input_std': x_std,
                'output_std': y_std,
                'input_min': x_min,
                'input_max': x_max
            }, f"exp_new/run{file_index}/ckpt_{epoch}.pt")

        if val_loss < best_val_loss:
            print('Validation loss decreased from {:.6f} to {:.6f}. Saving model...'.format(best_val_loss, val_loss))
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            torch.save({
                'state_dict': model.state_dict(),
                'model': model,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': f"{epoch+1}/{num_epochs}",
                'input_means': x_mean,
                'output_means': y_mean,
                'input_std': x_std,
                'output_std': y_std,
                'input_min': x_min,
                'input_max': x_max
            }, f"exp_new/run{file_index}/best_model.pt")
            best_epoch = epoch

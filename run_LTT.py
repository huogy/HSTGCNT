import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from stgcn import *
from LTT import temporal_transformer
import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

matrix_path = "dataset/W_228.csv"
data_path = "dataset/V_228.csv"
save_path = "save/model_Tae.pt"
save_path_AE = "save/TrueAE_model_D7_60.pt"


day_slot = 288
n_train, n_val, n_test = 31, 4, 9

n_his = 12
n_pred = 12
n_route = 228
Ks, Kt = 3, 3
blocks = [[1, 32, 64], [64, 32, 128]]
drop_prob = 0

batch_size = 16
epochs = 200
lr = 1e-3

d_model = 64  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 8  # Number of heads
N = 1  # Number of encoder and decoder to stack
attention_size = None  # Attention window size
dropout = 0.2  # Dropout rate
pe = "regular" # Positional encoding
chunk_mode = None

W = load_matrix(matrix_path)
L = scaled_laplacian(W)
Lk = cheb_poly(L, Ks)
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)

train, val, test = load_data(data_path, n_train * day_slot, n_val * day_slot)
scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)

loss = nn.MSELoss()
model = temporal_transformer(1,d_model,1, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in tqdm.tqdm(train_iter):
        x=x.permute(0,3,2,1)
        y_pred,x1,x2,x_bar = model(x)
        y_pred = y_pred.view(len(x), -1)
        l1 = loss(x_bar, x)
        l2 = loss(y_pred, y)

        l = 0.01*l1 + l2
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_ae_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path_AE)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)

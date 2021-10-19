import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from HSTGCNT import *
import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

matrix_path = "dataset/W_228.csv"
data_path = "dataset/V_228.csv"
save_path = "save/modelHSTGCNT15.pt"
save_path_AE = "save/AE_model1.pt"

day_slot = 288
n_train, n_val, n_test = 31, 4, 9

n_his = 12
n_pred = 12
n_route = 228
Ks, Kt = 3, 3
blocks = [[1, 32, 64], [64, 32, 128]]
drop_prob = 0

batch_size = 20
epochs = 50
lr = 1e-3

d_input = 1
d_output =1

d_model = 64  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 8  # Number of heads
N = 1  # Number of encoder and decoder to stack
attention_size = None  # Attention window size
dropout = 0.3  # Dropout rate
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
loss2 = nn.L1Loss()
model = HSTGCNT(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob,d_input,d_model,d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe, AEpath=save_path_AE).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)



min_val_loss = np.inf
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in tqdm.tqdm(train_iter):

        y_pred,x_bar,L_pred = model(x)
        y_pred = y_pred.view(len(x), -1)
        x_bar = x_bar.permute(0, 3, 2, 1)
        L_pred =L_pred.permute(0, 3, 2, 1).view(len(x), -1)

        l1 = loss(y_pred, y)
        l2 = loss2(y_pred, y)
        l3 = loss(x_bar, x)
        l4 = loss(L_pred,y)

        l = 1*l1+0.1*l2+0.01*l3+0.001*l4
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)

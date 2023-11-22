import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ArchSimple import SimpleNEt  # Assuming this is a custom model you've defined

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
df_x = pd.read_csv('./datasets/X.csv')
df_y = pd.read_csv('./datasets/Y.csv')


SS = StandardScaler()
df_x = SS.fit_transform(df_x)
df_y = SS.fit_transform(df_y)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

# Convert numpy arrays to torch tensors and move to the selected device
x_train = torch.tensor(x_train).float().to(device)
x_test = torch.tensor(x_test).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)

# Dataset and DataLoader
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

train_batch = DataLoader(train_data, batch_size=512, shuffle=True)
test_batch = DataLoader(test_data, batch_size=128, shuffle=True)

# Model, weights initialization, optimizer, and scheduler
model = SimpleNEt().to(device)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

# Training loop
loss_train = []
loss_eval = []

for i in range(20000):
    model.train()
    for x, y in train_batch:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(y)
        loss = F.mse_loss(output, x)
        loss.backward()
        optimizer.step()
    print('epoch: ', i, 'loss: ', loss.item())
    
    if i % 5 == 0:
        model.eval()
        loss_train.append(loss.item())
        with torch.no_grad():
            arr = []
            for x, y in test_batch:
                x, y = x.to(device), y.to(device)
                output = model(y)
                loss = F.mse_loss(output, x)  
                arr.append(loss.item())  
            loss_eval.append(np.mean(arr))
        print('test_loss ', loss.item(), 'eval_loss ', np.mean(arr))
        torch.save(model.state_dict(), './models/ResnetLastG.pth')
    if i % 2000 == 0:
        lr_scheduler.step()

# Plotting and evaluation
plt.plot(loss_train, label='train')
plt.plot(loss_eval, label='eval')
plt.legend()
plt.show()
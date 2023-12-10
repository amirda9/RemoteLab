import torch 
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
from torch.utils.data import DataLoader,Dataset
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

class DynamicNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_func):
        super(DynamicNet, self).__init__()
        layers = []

        # Add first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation_func())

        # Add subsequent hidden layers (if any)
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(activation_func())

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
StateModel = DynamicNet(344, [2048,1024], 236, nn.ReLU).to(device)
StateModel.load_state_dict(torch.load('./models/model_reverse1.pth', map_location=device))
print(device)
# Data preprocessing
df_x = pd.read_csv('./datasets/X.csv')
df_y = pd.read_csv('./datasets/Y.csv')


# Data preprocessing (assuming you have already loaded your data into df_x and df_y)
SS = StandardScaler()
x_processed = SS.fit_transform(df_x)
y_processed = SS.fit_transform(df_y)
x_train, x_test, y_train, y_test = train_test_split(x_processed, y_processed, test_size=0.2, random_state=42)

# Convert to torch tensors
x_train = torch.tensor(x_train).float().to(device)
x_test = torch.tensor(x_test).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)


rand = np.random.randint(0, 14000)
st = x_test[rand, :]
meas = y_test[rand, :]

meas_attacked = meas.clone()  # Clone to avoid modifying the original tensor

bus_idx = np.random.randint(0, 344, 20)

loss_criterion = nn.MSELoss()
for bus in bus_idx:
    meas_attacked[bus] = 0
    
    
loss_arr = []





# get from args
idx = sys.argv[1]
            

with open('./Missing/meas_'+str(idx)+'.pkl', 'wb') as f:
    pickle.dump(meas, f)

with open('./Missing/meas_attacked_'+str(idx)+'.pkl', 'wb') as f:
    pickle.dump(meas_attacked, f)
    
# save the bus_idx
with(open('./Missing/bus_idx_'+str(idx)+'.pkl', 'wb')) as f:
    pickle.dump(bus_idx, f)

print(loss_criterion(meas_attacked, meas))
print(loss_criterion(StateModel(meas_attacked.detach()), StateModel(meas.detach())))
# print(loss)
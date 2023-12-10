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
from scipy.io import loadmat
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error



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

MeasModel = DynamicNet(236, [1024], 344, nn.ReLU).to(device)
MeasModel.load_state_dict(torch.load('./models/model_1.pth', map_location=device))




idx = sys.argv[1]


with open('./Missing/meas_'+str(idx)+'.pkl', 'rb') as f:
    meas_gt_tensor = pickle.load(f)
with open('./Missing/meas_attacked_'+str(idx)+'.pkl', 'rb') as f:
    meas_attacked = pickle.load(f)
with open('./Missing/bus_idx_{}.pkl'.format(idx), 'rb') as f:
    bus_idx = pickle.load(f)

loss_criterion = nn.MSELoss()

loss_main = []
loss_meas = []
loss_state = []

alpha = 0.001

meas_temp = meas_attacked.clone().detach()



_ = 0

while True:
    _ = _ + 1
    # alpha*=0.99999
    meas_temp.requires_grad = True
    # st_temp.requires_grad = True
    st_temp = StateModel(meas_temp.reshape(1,-1))
    corr_meas = MeasModel(st_temp)
    cycle_state = StateModel(corr_meas)
    
    loss = loss_criterion(st_temp, cycle_state) + loss_criterion(corr_meas.reshape(-1,344), meas_temp.reshape(-1,344))
    loss.backward()
    
    with torch.no_grad():
        grad_arr = meas_temp.grad.numpy()
        meas_temp_arr = np.array(meas_temp)
        
        if np.linalg.norm(grad_arr) > 1:
            grad_arr /= np.linalg.norm(grad_arr)
        
        
        for bus in bus_idx:
            meas_temp_arr[bus] -= 2*alpha*grad_arr[bus]
        # meas_temp_arr -= 2*alpha*grad_arr
        
        
        meas_temp = torch.tensor(meas_temp_arr, dtype=torch.float32)
    
        loss_main.append(loss.item())
        loss_meas.append(mean_absolute_error(meas_temp.reshape(-1,344), meas_gt_tensor.reshape(-1,344)))
        loss_state.append(mean_absolute_error(StateModel(meas_temp.reshape(1,-1)).reshape(-1,236), StateModel(meas_gt_tensor.reshape(1,-1)).reshape(-1,236)))

        
    if _>10000:
        break
    # h = 0.001
    



plt.figure()
plt.plot(loss_main)
plt.title('loss_main')
plt.figure()
plt.plot(loss_state)
plt.title('loss_state')
plt.figure()
plt.plot(loss_meas)
plt.title('loss_meas')

plt.show()
print(idx)
# save the meas temp to the pickle file
with open('./Missing/meas_temp_'+str(idx)+'.pkl', 'wb') as f:
    pickle.dump(meas_temp, f)



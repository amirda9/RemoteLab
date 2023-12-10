import torch 
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import preprocessing
from ArchSimple import Model, SimpleNEt, SimpleNEt2
import datetime
from torch.utils.data import DataLoader,Dataset
import pickle
import sys
from scipy.io import loadmat
import scipy.io
import datetime


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


# MeasModel = Model(236, 512, 344)
# MeasModel.load_state_dict(torch.load('./SimpleF.pth', map_location=torch.device('cpu')))
# StateModel = Model(344, 512, 236)
# StateModel.load_state_dict(torch.load('./SimpleG.pth', map_location=torch.device('cpu')))

# MeasModel = SimpleNEt2()
# MeasModel.load_state_dict(torch.load('./ResnetLastF.pth', map_location=torch.device('cpu')))
# StateModel = SimpleNEt()
# StateModel.load_state_dict(torch.load('./ResnetLastG.pth', map_location=torch.device('cpu')))




 
StateModel.eval()
MeasModel.eval()

idx = sys.argv[1]


with open('./FDISample/meas_'+str(idx)+'.pkl', 'rb') as f:
    meas_gt_tensor = pickle.load(f)
with open('./FDISample/meas_attacked_'+str(idx)+'.pkl', 'rb') as f:
    meas_attacked = pickle.load(f)


meas_gt_tensor = meas_gt_tensor.to(device)
meas_attacked = meas_attacked.to(device)


loss_main = []
loss_state = []

alpha = 0.001


meas_temp = meas_attacked.clone().detach()


_ = 0


loss_criterion = nn.MSELoss()

while True:
    _ += 1
    meas_temp.requires_grad = True
    corr_state = StateModel(meas_temp)
    corr_meas = MeasModel(corr_state)
    cycle_state = StateModel(corr_meas)

    loss = loss_criterion(corr_state.reshape(-1,236), cycle_state.reshape(-1,236)) + loss_criterion(meas_temp.reshape(-1,344), corr_meas.reshape(-1,344))  
    loss.backward()
    with torch.no_grad():
        # alpha *= 0.9999
        # grad clipping
        start_time = datetime.datetime.now() 
        grad_arr = np.array(meas_temp.grad.detach().cpu())
        end_time = datetime.datetime.now()
        gradient_time = end_time-start_time
        print(f"Gradient calculation time: {gradient_time} seconds")
        # if np.linalg.norm(grad_arr) > 1:
        #     grad_arr = grad_arr / np.linalg.norm(grad_arr)     
        meas_temp_arr = np.array(meas_temp.detach().cpu())
        meas_temp_arr -= 2*alpha * grad_arr
        meas_temp = torch.tensor(meas_temp_arr, dtype=torch.float32)
        loss_main.append(loss.item())
        
        
        # early stopping with no change in loss
        if np.linalg.norm(grad_arr) < 1e-2:
            break
        # print(np.linalg.norm(loss))
        if _>5000:
            break


plt.figure()
plt.plot(loss_main)
plt.title('loss_main')
plt.figure()
plt.plot(loss_state)
plt.title('loss_state')

# plt.show()
print(idx)
# save the meas temp to the pickle file
with open('./FDI/meas_temp_'+str(idx)+'.pkl', 'wb') as f:
    pickle.dump(meas_temp, f)



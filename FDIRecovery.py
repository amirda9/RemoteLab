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


MeasModel = Model(236, 512, 344)
MeasModel.load_state_dict(torch.load('./SimpleF.pth', map_location=torch.device('cpu')))
StateModel = Model(344, 512, 236)
StateModel.load_state_dict(torch.load('./SimpleG.pth', map_location=torch.device('cpu')))

# MeasModel = SimpleNEt2()
# MeasModel.load_state_dict(torch.load('./ResnetLastF.pth', map_location=torch.device('cpu')))
# StateModel = SimpleNEt()
# StateModel.load_state_dict(torch.load('./ResnetLastG.pth', map_location=torch.device('cpu')))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
StateModel.to(device)
MeasModel.to(device)  


 
StateModel.eval()

idx = sys.argv[1]


with open('./FDISample/meas_'+str(idx)+'.pkl', 'rb') as f:
    meas_gt_tensor = pickle.load(f)
with open('./FDISample/meas_attacked_'+str(idx)+'.pkl', 'rb') as f:
    meas_attacked = pickle.load(f)


meas_gt_tensor = meas_gt_tensor.to(device)
meas_attacked = meas_attacked.to(device)


loss_main = []
loss_st = []
loss_init = []
grad_mag = []

alpha = 0.001


meas_temp = meas_attacked.detach().cpu()
_ = 0

meas_temp_arr = meas_temp.detach().cpu().numpy()


meas_temp = torch.tensor(meas_temp_arr, dtype=torch.float32)

loss_criterion = nn.MSELoss()

while True:
    _ += 1
    meas_temp.requires_grad = True
    corr_state = StateModel(meas_temp)
    corr_meas = MeasModel(corr_state)
    cycle_state = StateModel(corr_meas)

    loss = 3*loss_criterion(corr_state.reshape(-1,236), cycle_state.reshape(-1,236)) + loss_criterion(meas_temp.reshape(-1,344), corr_meas.reshape(-1,344))  
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
        loss_init.append(loss_criterion(StateModel(meas_temp.float()).reshape(-1,236), StateModel(meas_gt_tensor.float()).reshape(-1,236)).item())
        grad_mag.append(np.linalg.norm(grad_arr))
        # early stopping with no change in loss
        if np.linalg.norm(grad_arr) < 1e-2:
            break
        # print(np.linalg.norm(loss))
        if _>1000:
            break


plt.figure()
plt.plot(loss_main)
plt.title('loss_main')
plt.figure()
plt.plot(loss_init)
plt.title('State loss from initial state')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(['recovery algorithm', 'attacked error', 'healthy error'])
plt.figure()
plt.plot(grad_mag)
plt.title('Gradient magnitude')
plt.xlabel('iteration')
plt.ylabel('')


# plt.show()
print(idx)
# save the meas temp to the pickle file
with open('./FDISample/meas_temp_'+str(idx)+'.pkl', 'wb') as f:
    pickle.dump(meas_temp, f)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
import numpy as np
import os
from datetime import datetime
import itertools
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import scipy.io
from torch.utils.data import TensorDataset, DataLoader
from ArchSimple import SimpleNEt, SimpleNEt2,Model,Model2
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train test split 
states_pd = pd.read_csv('./datasets/X.csv')
meas_pd = pd.read_csv('./datasets/Y.csv')
states_train, states_test, meas_train, meas_test = train_test_split(states_pd, meas_pd, test_size=0.2, random_state=42)
states_train = torch.tensor(states_train.values).float().to(device)
states_test = torch.tensor(states_test.values).float().to(device)
meas_train = torch.tensor(meas_train.values).float().to(device)
meas_test = torch.tensor(meas_test.values).float().to(device)

# Dataset and DataLoader
train_data = TensorDataset(states_train, meas_train)
test_data = TensorDataset(states_test, meas_test)

# Create train dataloader
train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True)


#get the y matrix
Y_bus = loadmat('Y_bus118.mat')['Y_bus118']
Y_bus = Y_bus.astype('complex64')
# sparse matrix to dense matrix
Y_bus = Y_bus.todense()
Y_bus = np.array(Y_bus)
# Y_bus = torch.tensor(Y_bus, dtype=torch.complex64)


gen_bus = scipy.io.loadmat('IEEE118_gen.mat')['gen118'][:,0]
gen_bus = gen_bus.astype(int)


# Define the loss functions binary cross entropy and MSE
loss_criteria = nn.MSELoss()
model = Model(236,512,344).to(device)
model2 = Model2(236,512,344).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

optimizer2 = optim.Adam(model2.parameters(), lr=0.01)
lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1000, gamma=0.9)

for epoch in range(15000):
    model.train()
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        
        
        optimizer2.zero_grad()
        output = model2(x)
        
        vm = x[:,0:118] 
        va = x[:,118:236]
        v_r = torch.mul(vm,torch.cos(va*torch.tensor(np.pi/180,dtype=torch.float32))).to(device)
        v_i = torch.mul(vm,torch.sin(va*torch.tensor(np.pi/180,dtype=torch.float32))).to(device)
        V = torch.complex(v_r,v_i).to(device)
    
        S = torch.zeros((x.shape[0], 118), dtype=torch.complex64).to(device)
        I = torch.zeros((x.shape[0], 118), dtype=torch.complex64).to(device)
        for sample in range(x.shape[0]):
            I[sample] = torch.matmul(torch.tensor(Y_bus, dtype=torch.complex64).to(device), V[sample])
            S[sample] = torch.mul(V[sample], torch.conj(I[sample]))
        S_real_r = torch.real(S)*100
        S_real_i = torch.imag(S)*100
        
        
        S_r = -output[:,0:118]
        S_i = -output[:,118:236]
        for (idx,gen) in enumerate(gen_bus):
            # generation - demand
            # element wise addition
            S_r[:,gen-1] = S_r[:,gen-1] + output[:,236+idx]
            S_i[:,gen-1] = S_i[:,gen-1] + output[:,290+idx]
        
        loss = F.mse_loss(output, y) + F.mse_loss(torch.tensor(S_real_r, dtype=torch.float32), S_r) + F.mse_loss(torch.tensor(S_real_i, dtype=torch.float32), S_i)
        loss.backward()
        optimizer2.step()
    lr_scheduler.step()
    lr_scheduler2.step()
    
    loss_test = []
    loss_test2 = []
    if epoch % 5 == 0:
        model.eval()
        model2.eval()
        with torch.no_grad():
            arr = []
            arr2 = []
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = F.mse_loss(output, y)
                arr.append(loss.item())  
                
                output = model2(x)
                loss2 = F.mse_loss(output, y)
                arr2.append(loss2.item())
                
            loss_test.append(np.mean(arr))
            loss_test2.append(np.mean(arr2))
        print('epoch: ', epoch, 'loss: ', np.mean(arr), 'loss2: ', np.mean(arr2))
    
    
plt.figure()
plt.plot(loss_test)
plt.plot(loss_test2)
plt.legend(['model1','model2'])
plt.xlabel('epoch')
plt.ylabel('mse loss')
plt.show()

        

        
        


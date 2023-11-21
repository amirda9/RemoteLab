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


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class Model2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.dropout1 = nn.Dropout(0.2)         # Dropout for regularization
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        return out
    

class SimpleNEt(nn.Module):
    def __init__(self):
        super(SimpleNEt, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(344, 344),
            nn.ReLU(),
            nn.Linear(344, 300),
            nn.ReLU(),
            nn.Linear(300, 236)
        ) 
        
        self.net2 = nn.Sequential(
            nn.Linear(236+344, 512),
            nn.ReLU(),
            nn.Linear(512, 236)
        )
        
        
        
    def forward(self, x):
        res = self.net(x)
        try:
            concat = torch.cat((res, x), dim=1)
        except:
            concat = torch.cat((res, x), dim=0)
        res2 = self.net2(concat)
        return res2
    
    
class SimpleNEt2(nn.Module):
    def __init__(self):
        super(SimpleNEt2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(236, 256),
            nn.ReLU(),
            nn.Linear(256, 344),
            nn.ReLU(),
            nn.Linear(344, 344)
        ) 
        
        self.net2 = nn.Sequential(
            nn.Linear(236+344, 512),
            nn.ReLU(),
            nn.Linear(512, 344)
        )
        
        
        
    def forward(self, x):
        res = self.net(x)
        try:
            concat = torch.cat((res, x), dim=1)
        except:
            concat = torch.cat((res, x), dim=0)
        res2 = self.net2(concat)
        return res2
    
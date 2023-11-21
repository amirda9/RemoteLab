import torch 
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import preprocessing
from ArchSimple import Model,Model2,SimpleNEt, SimpleNEt2
import datetime
from torch.utils.data import DataLoader,Dataset
import pickle
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MeasModel = Model(236, 512, 344)
# MeasModel.load_state_dict(torch.load('./SimpleF.pth', map_location=torch.device('cpu')))
# MeasModel = Model2(236, 1024, 344)
# MeasModel.load_state_dict(torch.load('./SimpleFPrime.pth', map_location=torch.device('cpu')))
# StateModel = Model(344, 512, 236)
# StateModel.load_state_dict(torch.load('./SimpleG.pth', map_location=torch.device('cpu')))
# MeasModel = SimpleNEt()
# MeasModel.load_state_dict(torch.load('./ResnetLastF.pth', map_location=torch.device('cpu')))
StateModel = SimpleNEt2()
StateModel.load_state_dict(torch.load('./ResnetLastG.pth', map_location=torch.device('cpu')))


df_x = pd.read_csv('./datasets/X.csv')
df_y = pd.read_csv('./datasets/Y.csv')



# import standard scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

SS = StandardScaler()
df_x = SS.fit_transform(df_x)
df_y = SS.fit_transform(df_y)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=True)


import time

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)
            y_true.append(y.cpu())
            y_pred.append(predictions.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    a = torch.from_numpy(x_test[0]).float().to(device)
    start_time = time.time()
    model(a.unsqueeze(0))
    elapsed_time = time.time() - start_time

    print(f"R2 Score: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Time taken: {elapsed_time:.6f} seconds")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluate_model(MeasModel, test_loader, device)


from sklearn.linear_model import LinearRegression

def train_and_evaluate_linear_regression(x_train, y_train, x_test, y_test):
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    y_pred = lr_model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    start_time = time.time()
    lr_model.predict(x_test[0].reshape(1, -1))
    elapsed_time = time.time() - start_time

    print(f"R2 Score: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Time taken: {elapsed_time:.6f} seconds")

train_and_evaluate_linear_regression(x_train, y_train, x_test, y_test)



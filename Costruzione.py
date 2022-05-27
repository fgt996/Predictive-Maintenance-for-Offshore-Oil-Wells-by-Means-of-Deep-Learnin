#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:41:33 2022

@author: fede
"""

#%% COSA PRIMA

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

PATH = '/home/fede/Projects/Federico/PdM/ANSIA/data'
WINDOWS_LENGTH = 301
STATSxFEAT = 63

def StatVariables(batch):
    res = list()
    for col in range(batch.shape[1]):
        temp = batch[:, col]
        res.append([np.mean(temp), np.std(temp), skew(temp),
                    kurtosis(temp), np.median(temp), np.quantile(temp, 0.25),
                    np.quantile(temp, 0.75), np.max(temp), np.min(temp)])
    del(temp)
    res = np.array(res).flatten()
    return res      

def File2Windows_Stat(file_path, len_win, target_folder):
    #Define initial variables
    xs, ys = list(), list()
    half_len = len_win // 2
    
    #Load file
    df = pd.read_csv(file_path, index_col=0).fillna(0)
    #Remove T-JUS-CKGL as full of NaN
    df.pop('T-JUS-CKGL')
    #Extract classes 
    classes = list(df.pop('class'))
    #For each windows
    for window in range(0, len(df)-len_win, len_win):
        #Add value to xs
        xs.append(StatVariables(df.iloc[window:window+len_win].values))
        #Add class to ys according to transient or faulty period
        if classes[window:window+len_win].count(0) <= half_len:
            ys.append(int(target_folder))
        else:
            ys.append(0)
    #Add value to xs
    xs.append(StatVariables(df.iloc[-len_win:].values))
    #Add class to ys according to transient or faulty period
    if classes[-len_win:].count(0) <= half_len:
        ys.append(int(target_folder))
    else:
        ys.append(0)
    return np.array(xs), np.array(ys)

def Folder2Windows_Stat(folder_path, len_win, target_folder, statsxfeat):
    #Define initial variables
    x_train, y_train = np.zeros((1, statsxfeat)), np.zeros(1)
    x_val, y_val = np.zeros((1, statsxfeat)), np.zeros(1)
    x_test, y_test = np.zeros((1, statsxfeat)), np.zeros(1)
    
    #Split time series into simulated, real and drawn
    real, simulated = list(), list()
    for file in os.listdir(folder_path):
        if file.split('_')[0] == 'SIMULATED':
            simulated.append(file)
        elif file.split('-')[0] == 'WELL':
            real.append(file)
        elif file.split('_')[0] == 'DRAWN':
            pass
        else:
            raise ValueError(
                'Error: Data type not real, simulated or drawn')
    
    #Estimate the number of time series to be included in the test set
    test = int((len(real) / 5) + 0.999999)
    real.sort()
    #Add simulated data into train set
    for file in simulated:
        x_temp, y_temp = File2Windows_Stat(folder_path+'/'+file,
                                      len_win, target_folder)
        x_train = np.concatenate([x_train, x_temp], axis=0)
        y_train = np.concatenate([y_train, y_temp])
    #Add real data into train set
    for file in real[:-2*test]:
        x_temp, y_temp = File2Windows_Stat(folder_path+'/'+file,
                                      len_win, target_folder)
        x_train = np.concatenate([x_train, x_temp], axis=0)
        y_train = np.concatenate([y_train, y_temp])
    #Add real data into validation set
    for file in real[-2*test:-test]:
        x_temp, y_temp = File2Windows_Stat(folder_path+'/'+file,
                                      len_win, target_folder)
        x_val = np.concatenate([x_val, x_temp], axis=0)
        y_val = np.concatenate([y_val, y_temp])
    #Add real data into test set
    for file in real[-test:]:
        x_temp, y_temp = File2Windows_Stat(folder_path+'/'+file,
                                      len_win, target_folder)
        x_test = np.concatenate([x_test, x_temp], axis=0)
        y_test = np.concatenate([y_test, y_temp])
    #Remove the first, fake rows
    x_train = x_train[1:]; y_train = y_train[1:]
    x_val = x_val[1:]; y_val = y_val[1:]
    x_test = x_test[1:]; y_test = y_test[1:]
    
    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, y_train = np.zeros((1, STATSxFEAT)), np.zeros(1)
x_val, y_val = np.zeros((1, STATSxFEAT)), np.zeros(1)
x_test, y_test = np.zeros((1, STATSxFEAT)), np.zeros(1)
for folder in list(range(7))+[8]:
    x_train_t, x_val_t, x_test_t, y_train_t, y_val_t, y_test_t =\
        Folder2Windows_Stat(f'{PATH}/{folder}',
                            WINDOWS_LENGTH, folder, STATSxFEAT)
    x_train = np.concatenate([x_train, x_train_t], axis=0)
    y_train = np.concatenate([y_train, y_train_t])
    x_val = np.concatenate([x_val, x_val_t], axis=0)
    y_val = np.concatenate([y_val, y_val_t])
    x_test = np.concatenate([x_test, x_test_t], axis=0)
    y_test = np.concatenate([y_test, y_test_t])
#Remove the first, fake rows
x_train = x_train[1:]; y_train = y_train[1:]
x_val = x_val[1:]; y_val = y_val[1:]
x_test = x_test[1:]; y_test = y_test[1:]
#Scale the data
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)




#COSA VOGLIO OTTENERE? STUDIO x_train10 SALVATO

with open(f'{PATH}/x_train.pickle', 'rb') as f:
    df_rif_train = pickle.load(f)
with open(f'{PATH}/x_val.pickle', 'rb') as f:
    df_rif_val = pickle.load(f)
with open(f'{PATH}/x_test.pickle', 'rb') as f:
    df_rif_test = pickle.load(f)
with open(f'{PATH}/y_train.pickle', 'rb') as f:
    y_rif = pickle.load(f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%% COSA SECONDA

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PATH = '/home/fede/Projects/Federico/PdM/ANSIA/data'
WINDOWS_LENGTH = 301
STATS = 9
FEAT = 7

def File2Windows(file_path, len_win, target_folder):
    #Define initial variables
    xs, ys = list(), list()
    half_len = len_win // 2
    
    #Load file
    df = pd.read_csv(file_path, index_col=0).fillna(0)
    #Remove T-JUS-CKGL as full of NaN
    df.pop('T-JUS-CKGL')
    #Extract classes 
    classes = list(df.pop('class'))
    #For each windows
    for window in range(0, len(df)-len_win, len_win):
        #Fit scaler
        scaler = MinMaxScaler().fit(df.iloc[:window+len_win].values)
        #Add value to xs
        xs.append(scaler.transform(df.iloc[window:window+len_win].values))
        #Add class to ys according to transient or faulty period
        if classes[window:window+len_win].count(0) <= half_len:
            ys.append(int(target_folder))
        else:
            ys.append(0)
    #Fit scaler
    scaler = MinMaxScaler().fit(df.values)
    #Add value to xs
    xs.append(scaler.transform(df.iloc[-len_win:].values))
    #Add class to ys according to transient or faulty period
    if classes[-len_win:].count(0) <= half_len:
        ys.append(int(target_folder))
    else:
        ys.append(0)
    return np.array(xs), np.array(ys)

folder = 1
folder_path = f'{PATH}/{folder}'
target_folder = 1
len_win = 301
n_feat = FEAT





x_train, y_train = np.zeros((1, len_win, n_feat)), np.zeros(1)
x_val, y_val = np.zeros((1, len_win, n_feat)), np.zeros(1)
x_test, y_test = np.zeros((1, len_win, n_feat)), np.zeros(1)

#Split time series into simulated, real and drawn
real, simulated = list(), list()
for file in os.listdir(folder_path):
    if file.split('_')[0] == 'SIMULATED':
        simulated.append(file)
    elif file.split('-')[0] == 'WELL':
        real.append(file)
    elif file.split('_')[0] == 'DRAWN':
        pass
    else:
        raise ValueError(
            'Error: Data type not real, simulated or drawn')

#Estimate the number of time series to be included in the test set
test = int((len(real) / 5) + 0.999999)
real.sort()
#Add simulated data into train set
for file in simulated:
    x_temp, y_temp = File2Windows(folder_path+'/'+file,
                                  len_win, target_folder)
    x_train = np.concatenate([x_train, x_temp], axis=0)
    y_train = np.concatenate([y_train, y_temp])
#Add real data into train set
for file in real[:-2*test]:
    x_temp, y_temp = File2Windows(folder_path+'/'+file,
                                  len_win, target_folder)
    x_train = np.concatenate([x_train, x_temp], axis=0)
    y_train = np.concatenate([y_train, y_temp])
#Add real data into validation set
for file in real[-2*test:-test]:
    x_temp, y_temp = File2Windows(folder_path+'/'+file,
                                  len_win, target_folder)
    x_val = np.concatenate([x_val, x_temp], axis=0)
    y_val = np.concatenate([y_val, y_temp])
#Add real data into test set
for file in real[-test:]:
    x_temp, y_temp = File2Windows(folder_path+'/'+file,
                                  len_win, target_folder)
    x_test = np.concatenate([x_test, x_temp], axis=0)
    y_test = np.concatenate([y_test, y_temp])
#Remove the first, fake rows
x_train = x_train[1:]; y_train = y_train[1:]
x_val = x_val[1:]; y_val = y_val[1:]
x_test = x_test[1:]; y_test = y_test[1:]

return x_train, x_val, x_test, y_train, y_val, y_test

#%% COSA TERZA

import os
import pickle
import numpy as np
import pandas as pd
import multiprocessing
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler

WINDOWS_LENGTH = 301
PATH = '/home/fede/Projects/Federico/PdM/ANSIA/data'
STATS = 9
FEAT = 7

with open(f'{PATH}/Savings/x_train_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_train = np.array(pickle.load(f))
with open(f'{PATH}/Savings/x_val_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_val = np.array(pickle.load(f))
with open(f'{PATH}/Savings/x_test_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_test = np.array(pickle.load(f))

#Define immutable params while optimizing
imm_params = {'input_dim':(x_train.shape[1], x_train.shape[2]),
              'filt_1': 5, 'kernel_1': 7, 'stride_1': 4,
              'adj_kernel_1':13, 'adj_stride_1':8,
              'filt_2': 2, 'kernel_2':7, 'stride_2': 3,
              'adj_kernel_2':4, 'adj_stride_2':2,
              'patience':15,
              'TEMP_PATH':f'{PATH}/Savings/',
              'WINDOWS_LENGTH': WINDOWS_LENGTH}

params = imm_params.copy()
# Mutable params
params['batch'] = int(scale_number(chromosome[0], 20, 36))
params['lr'] = scale_number(chromosome[1], 1e-7, 1e-2)
params['reg2'] = scale_number(chromosome[2], 1e-8, 1e-2)








def Evaluation(file_prefix, file_suffix):
    #Load target
    with open(f'{file_prefix}/y_test.pickle', 'rb') as f:
        y_test = np.array(pickle.load(f))

    #Load stat predictions
    with open(f'{file_prefix}/RFC_stat.pickle', 'rb') as f:
        _, _, _, _, RFC_Stat, _, _, _ = pickle.load(f)
    with open(f'{file_prefix}/KNN_stat.pickle', 'rb') as f:
        _, _, _, _, KNN_Stat, _, _, _ = pickle.load(f)
    with open(f'{file_prefix}/GNB_stat.pickle', 'rb') as f:
        _, _, _, _, GNB_Stat, _, _, _ = pickle.load(f)
    with open(f'{file_prefix}/QDA_stat.pickle', 'rb') as f:
        _, _, _, _, QDA_Stat, _, _, _ = pickle.load(f)

    #Load enc predictions
    with open(f'{file_prefix}/RFC_Enc.pickle', 'rb') as f:
        _, _, _, _, RFC_Enc, _, _, _ = pickle.load(f)
    with open(f'{file_prefix}/KNN_Enc.pickle', 'rb') as f:
        _, _, _, _, KNN_Enc, _, _, _ = pickle.load(f)
    with open(f'{file_prefix}/GNB_Enc.pickle', 'rb') as f:
        _, _, _, _, GNB_Enc, _, _, _ = pickle.load(f)
    with open(f'{file_prefix}/QDA_Enc.pickle', 'rb') as f:
        _, _, _, _, QDA_Enc, _, _, _ = pickle.load(f)
        
    base = "{:<10} {:<10} {:<10}"
    #Evaluate prediction
    for Pred_test1, Pred_test2, Pred_name in zip([RFC_Stat, KNN_Stat,
                                                  GNB_Stat, QDA_Stat],
                                                 [RFC_Enc, KNN_Enc,
                                                  GNB_Enc, QDA_Enc],
                                                 ['RFC', 'KNN', 'GNB', 'QDA']):
        to_print = f'{Pred_name}'
        print(to_print)
        
        print(base.format('Metric','Stat','Enc'))
        print(base.format('Accuracy',
                          f'{round(acc(y_test, Pred_test1), 3)}',
                          f'{round(acc(y_test, Pred_test2), 3)}'))
        print(base.format('Precision',
                          f'{round(prec(y_test,Pred_test1,average=None)[0],3)}',
                          f'{round(prec(y_test,Pred_test2,average=None)[0],3)}'))
        print(base.format('Recall',
                          f'{round(rec(y_test,Pred_test1,average=None)[0],3)}',
                          f'{round(rec(y_test,Pred_test2,average=None)[0],3)}'))
        print(base.format('F1_Score',
                          f'{round(f1(y_test,Pred_test1,average=None)[0],3)}',
                          f'{round(f1(y_test,Pred_test2,average=None)[0],3)}'))
        print('\n')

PATH = '/home/fede/Projects/Federico/PdM/ANSIA/data'
Evaluation(f'{PATH}', WINDOWS_LENGTH)








with open(f'{file_prefix}/RFC_Enc.pickle', 'rb') as f:
    prova = pickle.load(f)

























mdl.AutoEncoder.summary()











PATH = '/home/fede/Projects/Federico/PdM/ANSIA/data/Savings/'
WINDOWS_LENGTH = 301



with open(f'{PATH}/Savings/x_train_Enc_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_train_example = pickle.load(f)

with open(f'{PATH}/Savings/x_train_Enc_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_train_saved = pickle.load(f)
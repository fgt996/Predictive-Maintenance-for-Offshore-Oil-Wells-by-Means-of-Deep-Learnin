#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

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
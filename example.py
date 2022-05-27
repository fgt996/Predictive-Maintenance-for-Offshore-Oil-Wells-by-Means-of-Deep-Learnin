#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

        
#%% Features Extraction

#%%% Create time windows

from features_extraction.create_windows import Folder2Windows

x_train, y_train = np.zeros((1, WINDOWS_LENGTH, FEAT)), np.zeros(1)
x_val, y_val = np.zeros((1, WINDOWS_LENGTH, FEAT)), np.zeros(1)
x_test, y_test = np.zeros((1, WINDOWS_LENGTH, FEAT)), np.zeros(1)
for folder in list(range(7))+[8]:
    x_train_t, x_val_t, x_test_t, y_train_t, y_val_t, y_test_t =\
        Folder2Windows(f'{PATH}/{folder}', WINDOWS_LENGTH, folder, FEAT)
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

#Save full time windows
with open(f'{PATH}/Savings/x_train_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(x_train, f)
with open(f'{PATH}/Savings/y_train_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(np.array(y_train), f)
with open(f'{PATH}/Savings/x_val_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(x_val, f)
with open(f'{PATH}/Savings/y_val_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(np.array(y_val), f)
with open(f'{PATH}/Savings/x_test_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(x_test, f)
with open(f'{PATH}/Savings/y_test_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(np.array(y_test), f)
        
#%%% Apply Statistical transformation

from features_extraction.statistical_features import Folder2Windows_Stat

x_train, y_train = np.zeros((1, STATS*FEAT)), np.zeros(1)
x_val, y_val = np.zeros((1, STATS*FEAT)), np.zeros(1)
x_test, y_test = np.zeros((1, STATS*FEAT)), np.zeros(1)
for folder in list(range(7))+[8]:
    x_train_t, x_val_t, x_test_t, y_train_t, y_val_t, y_test_t =\
        Folder2Windows_Stat(f'{PATH}/{folder}',
                            WINDOWS_LENGTH, folder, STATS*FEAT)
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

#Save Statistical features
with open(f'{PATH}/Savings/x_train_Stat_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(np.array(x_train), f)
with open(f'{PATH}/Savings/x_val_Stat_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(np.array(x_val), f)
with open(f'{PATH}/Savings/x_test_Stat_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(np.array(x_test), f)

#%%% Apply AutoEncoder transformation

from features_extraction.encoder_features import (BrkgaAutoEncoder, Encode_TW,
                                                  get_immutable_params)

#Load raw time windows
with open(f'{PATH}/Savings/x_train_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_train = np.array(pickle.load(f))
with open(f'{PATH}/Savings/x_val_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_val = np.array(pickle.load(f))
with open(f'{PATH}/Savings/x_test_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_test = np.array(pickle.load(f))

#Define immutable params while optimizing
imm_params = get_immutable_params(WINDOWS_LENGTH)
imm_params['input_dim'] = (x_train.shape[1], x_train.shape[2])
imm_params['patience'] = 15
imm_params['TEMP_PATH'] = f'{PATH}/Savings/'
imm_params['WINDOWS_LENGTH'] = WINDOWS_LENGTH
#Compute hyperparameters and best score for the AutoEncoder
params, score = BrkgaAutoEncoder(x_train, x_val, imm_params)
#Save the result
with open(f'{PATH}/Savings/AE_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump(params, f)
#Load result and encode time windows
with open(f'{PATH}/Savings/AE_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    params = pickle.load(f)
p = multiprocessing.Process(target=Encode_TW, args=(x_train, x_val,
                                                               x_test, params))
p.start()
p.join()

#%% Classification

from classifiers import (RFC_Classification, KNN_Classification,
                        GNB_Classification, QDA_Classification)

#%%% Results for statistical-based classifiers

#Load xs
with open(f'{PATH}/Savings/x_train_Stat_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_train = pickle.load(f)
with open(f'{PATH}/Savings/x_val_Stat_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_val = pickle.load(f)
with open(f'{PATH}/Savings/x_test_Stat_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_test = pickle.load(f)

#Load ys
with open(f'{PATH}/Savings/y_train_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    y_train = np.array(pickle.load(f))
with open(f'{PATH}/Savings/y_val_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    y_val = np.array(pickle.load(f))

#Make and save predictions
params, score, opt_time, Pred_train, Pred_test =\
    RFC_Classification(x_train, y_train, x_val, y_val, x_test)
with open(f'{PATH}/Savings/RFC_Stat_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump([params, score, opt_time, Pred_train, Pred_test], f)
params, score, opt_time, Pred_train, Pred_test =\
    KNN_Classification(x_train, y_train, x_val, y_val, x_test)
with open(f'{PATH}/Savings/KNN_Stat_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump([params, score, opt_time, Pred_train, Pred_test], f)
params, score, opt_time, Pred_train, Pred_test =\
    GNB_Classification(x_train, y_train, x_val, y_val, x_test)
with open(f'{PATH}/Savings/GNB_Stat_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump([params, score, opt_time, Pred_train, Pred_test], f)
params, score, opt_time, Pred_train, Pred_test =\
    QDA_Classification(x_train, y_train, x_val, y_val, x_test)
with open(f'{PATH}/Savings/QDA_Stat_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump([params, score, opt_time, Pred_train, Pred_test], f)

#%%% Results for autoencoder-based classifiers

#Load xs
with open(f'{PATH}/Savings/x_train_Enc_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_train = pickle.load(f)
    x_train = x_train.reshape((x_train.shape[0],\
                               x_train.shape[1]*x_train.shape[2]))
with open(f'{PATH}/Savings/x_val_Enc_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_val = pickle.load(f)
    x_val = x_val.reshape((x_val.shape[0],\
                           x_val.shape[1]*x_val.shape[2]))
with open(f'{PATH}/Savings/x_test_Enc_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    x_test = pickle.load(f)
    x_test = x_test.reshape((x_test.shape[0],\
                             x_test.shape[1]*x_test.shape[2]))

#Load ys
with open(f'{PATH}/Savings/y_train_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    y_train = np.array(pickle.load(f))
with open(f'{PATH}/Savings/y_val_{WINDOWS_LENGTH}.pickle', 'rb') as f:
    y_val = np.array(pickle.load(f))

#Make and save predictions
params, score, opt_time, Pred_train, Pred_test =\
    RFC_Classification(x_train, y_train, x_val, y_val, x_test)
with open(f'{PATH}/Savings/RFC_Enc_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump([params, score, opt_time, Pred_train, Pred_test], f)
params, score, opt_time, Pred_train, Pred_test =\
    KNN_Classification(x_train, y_train, x_val, y_val, x_test)
with open(f'{PATH}/Savings/KNN_Enc_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump([params, score, opt_time, Pred_train, Pred_test], f)
params, score, opt_time, Pred_train, Pred_test =\
    GNB_Classification(x_train, y_train, x_val, y_val, x_test)
with open(f'{PATH}/Savings/GNB_Enc_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump([params, score, opt_time, Pred_train, Pred_test], f)
params, score, opt_time, Pred_train, Pred_test =\
    QDA_Classification(x_train, y_train, x_val, y_val, x_test)
with open(f'{PATH}/Savings/QDA_Enc_{WINDOWS_LENGTH}.pickle', 'wb') as f:
    pickle.dump([params, score, opt_time, Pred_train, Pred_test], f)

#%% Evaluate results

from evaluation import Evaluation

Evaluation(f'{PATH}/Savings', WINDOWS_LENGTH)
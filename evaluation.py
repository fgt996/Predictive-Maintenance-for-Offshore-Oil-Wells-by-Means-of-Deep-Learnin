#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as prec
from sklearn.metrics import recall_score as rec
from sklearn.metrics import f1_score as f1

def Evaluation(file_prefix, file_suffix):
    #Load target
    with open(f'{file_prefix}/y_test_{file_suffix}.pickle', 'rb') as f:
        y_test = np.array(pickle.load(f))

    #Load stat predictions
    with open(f'{file_prefix}/RFC_Stat_{file_suffix}.pickle', 'rb') as f:
        _, _, _, _, RFC_Stat = pickle.load(f)
    with open(f'{file_prefix}/KNN_Stat_{file_suffix}.pickle', 'rb') as f:
        _, _, _, _, KNN_Stat = pickle.load(f)
    with open(f'{file_prefix}/GNB_Stat_{file_suffix}.pickle', 'rb') as f:
        _, _, _, _, GNB_Stat = pickle.load(f)
    with open(f'{file_prefix}/QDA_Stat_{file_suffix}.pickle', 'rb') as f:
        _, _, _, _, QDA_Stat = pickle.load(f)

    #Load enc predictions
    with open(f'{file_prefix}/RFC_Enc_{file_suffix}.pickle', 'rb') as f:
        _, _, _, _, RFC_Enc = pickle.load(f)
    with open(f'{file_prefix}/KNN_Enc_{file_suffix}.pickle', 'rb') as f:
        _, _, _, _, KNN_Enc = pickle.load(f)
    with open(f'{file_prefix}/GNB_Enc_{file_suffix}.pickle', 'rb') as f:
        _, _, _, _, GNB_Enc = pickle.load(f)
    with open(f'{file_prefix}/QDA_Enc_{file_suffix}.pickle', 'rb') as f:
        _, _, _, _, QDA_Enc = pickle.load(f)
        
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

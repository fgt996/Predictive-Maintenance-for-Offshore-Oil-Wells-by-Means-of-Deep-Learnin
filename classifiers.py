#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 16:08:33 2022

@author: fede
"""

import pickle
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import warnings
warnings.filterwarnings('ignore')

def scale_number(unscaled, to_min, to_max, from_min=0, from_max=1):
    return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min

#%% Random Forest Classifier (RFC)

class RandomForestDecoder:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.X_train = x_train
        self.X_val = x_val
        self.Y_train = y_train
        self.Y_val = y_val

    def get_decoded(self, chromosome):
        bootstrap = (chromosome[0] <= 0.5)
        max_depth = int(scale_number(chromosome[1], 1, 25))
        max_feature = 'log2'
        if chromosome[2] <= 0.5:
            max_feature = 'sqrt'
        class_weight = None
        if chromosome[3] <= 0.33:
            class_weight = 'balanced'
        elif chromosome[3] >= 0.67:
            class_weight = 'balanced_subsample'

        n_estimators = int(scale_number(chromosome[3], 25, 1000))
        params = {
            # Parameters to optimize
            'bootstrap': bootstrap,
            'max_depth': max_depth,
            'max_features': max_feature,
            'n_estimators': n_estimators,
            'class_weight': class_weight,
            # Immutable Params
            'verbose': 0,
            'n_jobs': 8
        }

        return params

    def decode(self, chromosome):
        parameters = self.get_decoded(chromosome)
        mdl = RandomForestClassifier(
            **parameters).fit(self.X_train, self.Y_train)
        Y_pred = mdl.predict(self.X_val)
        return -accuracy_score(self.Y_val, Y_pred)


def RandomForest_BrkgaSearch(X_train, Y_train, X_val, Y_val, n_iter=100):
    from brkga import BRKGA
    decoder = RandomForestDecoder(X_train, Y_train, X_val, Y_val)
    chromosome_size = 4
    evolutions = 10
    elements = int(n_iter / evolutions)
    b = BRKGA.BRKGA(chromosome_size, elements, 0.25,
                    0.2, 0.6, 0, decoder, maximize=False)
    b.optimize(evolutions)
    return decoder.get_decoded(b.get_best_chromosome()), b.get_best_fitness()

def RFC_Classification(x_train, y_train, x_val, y_val, x_test):
    #Hyperparameters optimization
    START = time()
    params, score = RandomForest_BrkgaSearch(x_train, y_train, x_val, y_val)
    opt_time = time() - START
    #Fit and predict
    mdl = RandomForestClassifier(**params).fit(x_train, y_train)
    Pred_train = mdl.predict(np.concatenate([x_train, x_val]))
    mdl = RandomForestClassifier(**params).fit(np.concatenate([x_train, x_val]),
                                               np.concatenate([y_train, y_val]))
    Pred_test = mdl.predict(x_test)
    return params, score, opt_time, Pred_train, Pred_test
    
#%% K-Nearest Neighbour (KNN)

class KNNDecoder:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.X_train = x_train
        self.X_val = x_val
        self.Y_train = y_train
        self.Y_val = y_val

    def get_decoded(self, chromosome):
        n_neighbors = int(scale_number(chromosome[0], 3, 14))
        weights = 'uniform'
        if chromosome[1] <= 0.5:
            weights = 'distance'
        p = int(scale_number(chromosome[2], 1, 6))
        params = {
            # Parameters to optimize
            'n_neighbors': n_neighbors,
            'weights': weights,
            'p': p,
            # Immutable Params
            'n_jobs': 8
        }

        return params

    def decode(self, chromosome):        
        parameters = self.get_decoded(chromosome)
        mdl = KNeighborsClassifier(
            **parameters).fit(self.X_train, self.Y_train)
        Y_pred = mdl.predict(self.X_val)

        return -accuracy_score(self.Y_val, Y_pred)


def KNN_BrkgaSearch(X_train, Y_train, X_val, Y_val, n_iter=100):
    from brkga import BRKGA
    decoder = KNNDecoder(X_train, Y_train, X_val, Y_val)
    chromosome_size = 3
    evolutions = 10
    elements = int(n_iter / evolutions)
    b = BRKGA.BRKGA(chromosome_size, elements, 0.25,
                    0.2, 0.6, 0, decoder, maximize=False)
    b.optimize(evolutions)
    return decoder.get_decoded(b.get_best_chromosome()), b.get_best_fitness()

def KNN_Classification(x_train, y_train, x_val, y_val, x_test):
    #Hyperparameters optimization
    START = time()
    params, score = KNN_BrkgaSearch(x_train, y_train, x_val, y_val)
    opt_time = time() - START
    #Fit and predict
    mdl = KNeighborsClassifier(**params).fit(x_train, y_train)
    Pred_train = mdl.predict(np.concatenate([x_train, x_val]))
    mdl = KNeighborsClassifier(**params).fit(np.concatenate([x_train, x_val]),
                                             np.concatenate([y_train, y_val]))
    Pred_test = mdl.predict(x_test)
    return params, score, opt_time, Pred_train, Pred_test

#%% Gaussian Naive Bayes (GNB)

class GNBDecoder:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.X_train = x_train
        self.X_val = x_val
        self.Y_train = y_train
        self.Y_val = y_val

    def get_decoded(self, chromosome):
        var_smoothing = scale_number(chromosome[0], 1e-5, 1e-11)
        params = {
            # Parameters to optimize
            'var_smoothing': var_smoothing
        }

        return params

    def decode(self, chromosome):        
        parameters = self.get_decoded(chromosome)
        mdl = GaussianNB(**parameters).fit(self.X_train, self.Y_train)
        Y_pred = mdl.predict(self.X_val)

        return -accuracy_score(self.Y_val, Y_pred)


def GNB_BrkgaSearch(X_train, Y_train, X_val, Y_val, n_iter=100):
    from brkga import BRKGA
    decoder = GNBDecoder(X_train, Y_train, X_val, Y_val)
    chromosome_size = 1
    evolutions = 8
    elements = int(n_iter / evolutions)
    b = BRKGA.BRKGA(chromosome_size, elements, 0.25,
                    0.2, 0.6, 0, decoder, maximize=False)
    b.optimize(evolutions)
    return decoder.get_decoded(b.get_best_chromosome()), b.get_best_fitness()

def GNB_Classification(x_train, y_train, x_val, y_val, x_test):
    #Hyperparameters optimization
    START = time()
    params, score = GNB_BrkgaSearch(x_train, y_train, x_val, y_val)
    opt_time = time() - START
    #Fit and predict
    mdl = GaussianNB(**params).fit(x_train, y_train)
    Pred_train = mdl.predict(np.concatenate([x_train, x_val]))
    mdl = GaussianNB(**params).fit(np.concatenate([x_train, x_val]),
                                   np.concatenate([y_train, y_val]))
    Pred_test = mdl.predict(x_test)
    return params, score, opt_time, Pred_train, Pred_test

#%% Quadratic Discriminant Analysis (QDA)

class QDADecoder:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.X_train = x_train
        self.X_val = x_val
        self.Y_train = y_train
        self.Y_val = y_val

    def get_decoded(self, chromosome):
        reg_param = scale_number(chromosome[0], 0, 1)
        params = {
            # Parameters to optimize
            'reg_param': reg_param
        }

        return params

    def decode(self, chromosome):
        parameters = self.get_decoded(chromosome)
        mdl = QuadraticDiscriminantAnalysis(**parameters).fit(self.X_train,
                                                              self.Y_train)
        Y_pred = mdl.predict(self.X_val)

        return -accuracy_score(self.Y_val, Y_pred)


def QDA_BrkgaSearch(X_train, Y_train, X_val, Y_val, n_iter=25):
    from brkga import BRKGA
    decoder = QDADecoder(X_train, Y_train, X_val, Y_val)
    chromosome_size = 1
    evolutions = 5
    elements = int(n_iter / evolutions)
    b = BRKGA.BRKGA(chromosome_size, elements, 0.25,
                    0.2, 0.6, 0, decoder, maximize=False)
    b.optimize(evolutions)
    return decoder.get_decoded(b.get_best_chromosome()), b.get_best_fitness()

def QDA_Classification(x_train, y_train, x_val, y_val, x_test):
    #Hyperparameters optimization
    START = time()
    params, score = QDA_BrkgaSearch(x_train, y_train, x_val, y_val)
    opt_time = time() - START
    #Fit and predict
    mdl = QuadraticDiscriminantAnalysis(**params).fit(x_train, y_train)
    Pred_train = mdl.predict(np.concatenate([x_train, x_val]))
    mdl = QuadraticDiscriminantAnalysis(**params).fit(np.concatenate([x_train,
                                                                      x_val]),
                                                      np.concatenate([y_train,
                                                                      y_val]))
    Pred_test = mdl.predict(x_test)
    return params, score, opt_time, Pred_train, Pred_test
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import multiprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Input

def get_immutable_params(w_len):
    if w_len == 301:
        return {'filt_1':5, 'kernel_1':7, 'stride_1':4,
                'adj_kernel_1':13, 'adj_stride_1':8,
                'filt_2':2, 'kernel_2':7, 'stride_2':3,
                'adj_kernel_2':4, 'adj_stride_2':2}
    elif w_len == 451:
        return {'filt_1':5, 'kernel_1':7, 'stride_1':4,
                'adj_kernel_1':28, 'adj_stride_1':6,
                'filt_2':2, 'kernel_2':7, 'stride_2':3,
                'adj_kernel_2':9, 'adj_stride_2':2}
    elif w_len == 601:
        return {'filt_1':5, 'kernel_1':7, 'stride_1':4,
                'adj_kernel_1':29, 'adj_stride_1':9,
                'filt_2':2, 'kernel_2':7, 'stride_2':4,
                'adj_kernel_2':5, 'adj_stride_2':6}
    else:
        raise ValueError(f'Windows length {w_len} not recognized')
    
class AutoEncoder():
    def __init__(self, params):
        self.params = params
        self.regularizer = self.Define_Regularizer()
        self.optimizer = self.Define_Optimizer()
        self.AutoEncoder = self.Create_AutoEncoder()
    
    def Define_Regularizer(self):
        regularizer = l2(self.params['reg2'])
        return regularizer
    
    def Define_Optimizer(self):
        opt = Adam(lr=self.params['lr'])
        return opt
    
    def Create_AutoEncoder(self):
        #Input layer
        input_x = Input(shape=self.params['input_dim'])
        #Encoder
        conv_1 = Conv1D(filters=self.params['filt_1'],
                        kernel_size=self.params['kernel_1'],
                        strides=self.params['stride_1'],
                        activation='relu',
                        kernel_regularizer=self.regularizer)(input_x)
        conv_2 = Conv1D(filters=self.params['filt_2'],
                        kernel_size=self.params['kernel_2'],
                        strides=self.params['stride_2'],
                        activation='relu',
                        kernel_regularizer=self.regularizer,
                        name='EncFinal')(conv_1)
        #Decoder
        dec_2 = Conv1DTranspose(filters=self.params['filt_1'],
                                kernel_size=self.params['kernel_2']+\
                                    self.params['adj_kernel_2'],
                                dilation_rate=self.params['stride_2']+\
                                    self.params['adj_stride_2'],
                                activation='relu')(conv_2)
        dec_1 = Conv1DTranspose(filters=self.params['input_dim'][-1],
                                kernel_size=self.params['kernel_1']+\
                                    self.params['adj_kernel_1'],
                                dilation_rate=self.params['stride_1']+\
                                    self.params['adj_stride_1'],
                                activation='relu')(dec_2)
        #Define model
        mdl = Model(inputs=input_x, outputs=dec_1)
        return mdl

def run_func(x_train, x_val, params):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    #Define and fit AutoEncoder
    mdl = AutoEncoder(params)
    mdl.AutoEncoder.compile(loss='mse', optimizer=mdl.optimizer, metrics=[
                RootMeanSquaredError(name='rmse')])
    history = mdl.AutoEncoder.fit(x_train, x_train, batch_size=params['batch'],
                      epochs=2000, verbose=0,
                      callbacks=[EarlyStopping(monitor='val_rmse',
                                               patience=params['patience'],
                                               verbose=0,
                                               restore_best_weights=True)],
                      validation_data=(x_val, x_val))
    
    #Save the result
    with open(f'{params["TEMP_PATH"]}Temp.pickle', 'wb') as f:
        pickle.dump(history.history['val_rmse'][-params['patience']-1], f)
        

def AE_function(x_train, x_val, params):
    #Launch AutoEncoder
    p = multiprocessing.Process(target=run_func, args=(x_train, x_val, params))
    p.start()
    p.join()
    #Obtain results
    try:
        with open(f'{params["TEMP_PATH"]}Temp.pickle', 'rb') as f:
            ris = pickle.load(f)
        os.remove(f'{params["TEMP_PATH"]}Temp.pickle')
    except:
        ris = np.inf
    return ris


def scale_number(unscaled, to_min, to_max, from_min=0, from_max=1):
    return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min


class AE_Decoder:
    def __init__(self, x_train, x_val, immutable_params):
        self.x_train = x_train
        self.x_val = x_val
        self.immutable_params = immutable_params

    def get_decoded(self, chromosome):
        # Immutable params
        params = self.immutable_params.copy()
        # Mutable params
        params['batch'] = int(scale_number(chromosome[0], 20, 36))
        params['lr'] = scale_number(chromosome[1], 1e-7, 1e-2)
        params['reg2'] = scale_number(chromosome[2], 1e-8, 1e-2)
        return params

    def decode(self, chromosome):
        parameters = self.get_decoded(chromosome)
        ris = AE_function(self.x_train, self.x_val, parameters)
        return ris


def BrkgaAutoEncoder(x_train, x_val, immutable_params=dict(),  n_iter=49):
    from brkga import BRKGA
    decoder = AE_Decoder(x_train, x_val, immutable_params)
    chromosome_size = 3
    evolutions = 7
    elements = int(n_iter / evolutions)
    b = BRKGA.BRKGA(chromosome_size, elements, 0.25,
                    0.2, 0.6, 0, decoder, maximize=False)
    b.optimize(evolutions)
    return decoder.get_decoded(b.get_best_chromosome()), b.get_best_fitness()

def Encode_TW(x_train, x_val, x_test, params):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    #Define and fit AutoEncoder
    mdl = AutoEncoder(params)
    mdl.AutoEncoder.compile(loss='mse', optimizer=mdl.optimizer, metrics=[
                RootMeanSquaredError(name='rmse')])
    mdl.AutoEncoder.fit(x_train, x_train, batch_size=params['batch'],
                      epochs=2000, verbose=2,
                      callbacks=[EarlyStopping(monitor='val_rmse',
                                               patience=params['patience'],
                                               verbose=0,
                                               restore_best_weights=True)],
                      validation_data=(x_val, x_val))
    #Extract Encoder and encode time windows
    encoder = Model(inputs=mdl.AutoEncoder.inputs,
                    outputs=mdl.AutoEncoder.get_layer(name='EncFinal').output)
    Train = encoder.predict(x_train)
    Val = encoder.predict(x_val)
    Test = encoder.predict(x_test)
    #Save the result
    with open(f'{params["TEMP_PATH"]}x_train_Enc_{params["WINDOWS_LENGTH"]}'+\
              '.pickle', 'wb') as f:
        pickle.dump(Train, f)
    with open(f'{params["TEMP_PATH"]}x_val_Enc_{params["WINDOWS_LENGTH"]}'+\
              '.pickle', 'wb') as f:
        pickle.dump(Val, f)
    with open(f'{params["TEMP_PATH"]}x_test_Enc_{params["WINDOWS_LENGTH"]}'+\
              '.pickle', 'wb') as f:
        pickle.dump(Test, f)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:23:42 2020

@author: iagorosa
"""

#%%
import numpy as np
from PIL import Image as pil
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelBinarizer

import aux_functions as af

#%%

class BaseELM():
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh', random_state=None, bias = True):
        self.n_hidden = n_hidden
        self.func_hidden_layer = func_hidden_layer
        self.random_state = random_state
        self.activation_func = activation_func
        self.bias = bias
        
    def create_random_layer(self):
        return self.func_hidden_layer(self.input_lenght, self.n_hidden, self.random_state)
    
    def create_bias(self):
        return np.random.normal(size=[self.n_hidden])

    def input_to_hidden(self, X): # produto pesos de entrada com neuronios de pesos aleatorios + função de ativação
        if self.bias:
            B_mat = np.tile(self.B, (X.shape[0], 1)) # repete a linha varias vezes
            a = np.dot(X, self.W) + B_mat
        else:
            a = np.dot(X, self.W) 
            
        return af.f_activation(a, self.activation_func)
        
    def fit(self, X, y):
        
        self.input_lenght = X.shape[1]
        y_bin = self.binarizer.fit_transform(y)  
        
        self.W = self.create_random_layer()
        if self.bias:
            self.B = self.create_bias()
            
        self.H = self.input_to_hidden(X)
        Ht = self.H.T
        
        self.beta = np.dot(np.linalg.pinv(np.dot(Ht, self.H)), np.dot(Ht, y_bin))
        
        return self.beta
        
    
    def predict(self, y, y_pred):
#        H = self.input_to_hidden(X)
#        y_pred = np.dot(H, self.beta)
        
        correct = 0
        total = y.shape[0]
        r_test = []
        for i in range(total):
            predicted = np.argmax(y_pred[i])
        #    test = np.argmax(y_test[i])
            test = y[i]
            r_test.append(predicted)
            correct = correct + (1 if predicted == test else 0)
        return correct/total, r_test
    
    
    def _get_weights(self):
        return self.W  
    
    def _get_bias(self):
        return self.B
    
    def _get_beta(self):
        return self.beta
    
    def _get_H(self):
        return self.H
    

class ELMMLPClassifier(BaseELM):
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh',  binarizer=LabelBinarizer(0, 1), random_state=None, bias=True):
        
        super(ELMMLPClassifier, self).__init__(n_hidden=n_hidden, func_hidden_layer = func_hidden_layer, activation_func=activation_func, random_state=random_state, bias=True)
        
        self.binarizer = binarizer
        
    def fit(self, X, y):
        
#        self.beta = self._get_beta()
#        self.H = self._get_H()
        super(ELMMLPClassifier, self).fit(X, y)
        
        self.H1 = np.dot(y, np.linalg.pinv(self.beta)) 
        self.He = np.concatenate([self.H, np.ones(self.H.shape[0]).reshape(-1, 1)], axis=1)
        
        H1_inv = af.af_inverse(X, self.activation_func) 
        
        Whe = np.dot(np.linalg.pinv(self.He), H1_inv)
        
        self.Wh = Whe[:-1, :]
        self.B1  = Whe[-1, :]
        
        #TODO: melhorar essa parte. Colocar os calculos de H2 (input_to_hidden) em uma nova funcao?
        B1_mat = np.tile(self.B1, (self.H.shape[0], 1))
        self.H2 = af.f_activation(np.dot(self.H, self.Wh) + B1_mat, self.activation_func)
        print(self.H)
        self.beta2 = np.dot(np.linalg.pinv(self.H2), y)
        
        return self.beta2
          

    def predict(self, X, y):

        B1_mat = np.tile(self.B1, (X.shape[0], 1))
        B_mat = np.tile(self.B1, (X.shape[0], 1))

        p1 = af.f_activation(np.dot(X, self.W)  + B_mat, self.activation_func)
        p2 = af.f_activation(np.dot(p1, self.W) + B1_mat, self.activation_func)
        
        y_pred = np.dot(p2, self.beta2)
        
        return super(ELMMLPClassifier, self).predict(y, y_pred)
        
        
    def get_weights(self):
        return super(ELMClassifier, self)._get_weights()
    
    def get_bias(self):
        return super(ELMClassifier, self)._get_bias()
        



class ELMClassifier(BaseELM):
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh',  binarizer=LabelBinarizer(0, 1), random_state=None, bias = True):
        
        super(ELMClassifier, self).__init__(n_hidden=n_hidden, func_hidden_layer = func_hidden_layer, activation_func=activation_func, random_state=random_state, bias=bias)
        
        self.binarizer = binarizer
    
    def get_weights(self):
        return super(ELMClassifier, self)._get_weights()
    
    def get_bias(self):
        return super(ELMClassifier, self)._get_bias()
    
    def predict(self, X, y):
        H = self.input_to_hidden(X)
        y_pred = np.dot(H, self.beta)
        
        return super(ELMClassifier, self).predict(y, y_pred)
        
        


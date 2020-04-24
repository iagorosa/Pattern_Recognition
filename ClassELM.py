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

from sklearn.utils.extmath import safe_sparse_dot

import aux_functions as af

import scipy as sc

#%%

class BaseELM():
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh', random_state=None, bias = True, regressor = 'pinv', lbd = 0.01, degree = 1):
        self.n_hidden = n_hidden
        self.func_hidden_layer = func_hidden_layer
        self.random_state = random_state
        self.activation_func = activation_func
        self.bias = bias
        self.regressor = regressor
        self.lbd = lbd
        self.degree = degree
        
    def create_random_layer(self):
        return self.func_hidden_layer(self.input_lenght, self.n_hidden, self.random_state, self.n_classes)
    
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
        
        self.y_bin = self.binarizer.fit_transform(y)  
        
        self.input_lenght = X.shape[1]
        
        self.n_classes = self.y_bin.shape[1]
        
        self.W = self.create_random_layer()
        if self.bias:
            self.B = self.create_bias()
            
        self.H = self.input_to_hidden(X)
        Ht = self.H.T
        
#        self.beta = np.dot(np.linalg.pinv(np.dot(Ht, self.H)), np.dot(Ht, self.y_bin))
#        self.beta = np.dot(np.linalg.pinv(self.H), self.y_bin)
        
        self.beta = self._regressor(self.H, self.y_bin)
        
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
    
    def _regressor(self, A, d):
        
        if self.regressor ==  None:
            return np.linalg.solve(A, d )
        
        elif self.regressor == 'pinv':
#            return np.dot(np.linalg.pinv(A), d ) #TODO: verificar parametro rcond
            return np.dot(sc.linalg.pinv2(A), d )
        
        elif self.regressor == 'ls':
#            np.linalg.solve(np.dot(A.T, A), np.dot(A.T, d))  #Somar ldb diagonal principal
            return np.linalg.lstsq(A, d, rcond=None)[0]
        
        elif self.regressor == 'ls_dual':
            self.K = A @ A.T
            I = np.identity(len(self.K)) # identidade
            
            #TODO: preocupacao com bias?
            self.alpha = np.linalg.solve( self.K ** self.degree + self.lbd * I , d)  #TODO: usei solve comum
            
#            print('alpha:', self.alpha.shape, '\nA:', A.shape)
#            print((self.alpha.T @ A).shape)
            return (self.alpha.T @ A).T  # Transposto pq na classificacao da incompatibilidade de dimensao
            
     

class ELMMLPClassifier(BaseELM):
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh',  binarizer=LabelBinarizer(0, 1), random_state=None, regressor = 'pinv', lbd = 0.01):
        
        super(ELMMLPClassifier, self).__init__(n_hidden=n_hidden, func_hidden_layer = func_hidden_layer, activation_func=activation_func, random_state=random_state, bias=True)
        
        self.binarizer = binarizer
        
    def fit(self, X, y):
        
#        self.beta = self._get_beta()
#        self.H = self._get_H()
#        print("fit")
        super(ELMMLPClassifier, self).fit(X, y)
        
        self.H1 = np.dot(self.y_bin, np.linalg.pinv(self.beta)) 
        self.He = np.concatenate([self.H, np.ones(self.H.shape[0]).reshape(-1, 1)], axis=1)
        
#        print(self.H1)s
        H1_inv = af.af_inverse(self.H1, self.activation_func)  
        #TODO: Problemas com a aplicacao da funcao inversa
#        print("H1^-1")
#        print(H1_inv)
#        print()
        
        Whe = np.dot(np.linalg.pinv(self.He), H1_inv)
        
        self.W1 = Whe[:-1, :]
        self.B1  = Whe[-1, :]
        
        #TODO: melhorar essa parte. Colocar os calculos de H2 (input_to_hidden) em uma nova funcao?
        B1_mat = np.tile(self.B1, (self.H.shape[0], 1))
        self.H2 = af.f_activation(np.dot(self.H, self.W1) + B1_mat, self.activation_func)
        self.beta2 = np.dot(np.linalg.pinv(self.H2), self.y_bin)
        
        return self.beta2
          

    def predict(self, X, y):

        B1_mat = np.tile(self.B1, (X.shape[0], 1))
        B_mat  = np.tile(self.B, (X.shape[0], 1))

#        print(B_mat.shape, X.shape, self.W.shape)
        f1 = af.f_activation( (np.dot(X,  self.W)  + B_mat), self.activation_func)
        f2 = af.f_activation( (np.dot(f1, self.W1) + B1_mat), self.activation_func)
        
        y_pred = np.dot(f2, self.beta2)
        
        return super(ELMMLPClassifier, self).predict(y, y_pred)
        
        
    def get_weights(self):
        return super(ELMClassifier, self)._get_weights()
    
    def get_bias(self):
        return super(ELMClassifier, self)._get_bias()
        



class ELMClassifier(BaseELM):
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh',  binarizer=LabelBinarizer(0, 1), random_state=None, bias = True, regressor = 'pinv', lbd = 0.01, degree = 1):
        
        super(ELMClassifier, self).__init__(n_hidden=n_hidden, func_hidden_layer = func_hidden_layer, activation_func=activation_func, random_state=random_state, bias=bias, regressor = regressor, degree = degree)
                
        self.binarizer = binarizer
    
    def get_weights(self):
        return super(ELMClassifier, self)._get_weights()
    
    def get_bias(self):
        return super(ELMClassifier, self)._get_bias()
    
    def predict(self, X, y):
        self.H_test = self.input_to_hidden(X)
#        y_pred = np.dot(H, self.beta)
        
        '''
        if (self.regressor == 'ls_dual'):
            y_pred = X @ self.beta
            
        else:
#            y_pred = safe_sparse_dot(H, self.beta)
            y_pred = self.H_test @ self.beta
        '''
            
        y_pred = self.H_test @ self.beta
        
        if self.regressor == 'ls_dual':
            mat = self.H @ self.H_test.T
            y_pred = self.alpha.T @ (mat ** self.degree)
            y_pred = y_pred.T
        
        
        return super(ELMClassifier, self).predict(y, y_pred)
        
        


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:48:57 2020

@author: iagorosa
"""
   
import numpy as np
import scipy as sc

def normal_random_layer(input_leght, n_hidden, random_state=None, n_classes=None):
    
    np.random.seed(random_state)
    
    return np.random.normal(size=[input_leght, n_hidden])

def uniform_random_layer(input_leght, n_hidden, random_state=None, n_classes=None):
    
    np.random.seed(random_state)
    
    return np.random.uniform(-1, 1, size=[input_leght, n_hidden])

def SCAWI(input_leght, n_hidden, random_state=None, n_classes=None):

    np.random.seed(random_state)
    
    W = np.random.uniform(-1, 1, size=(input_leght, n_hidden))
    
    if random_state != None:
        np.random.seed(random_state+105)
        
    r = np.random.uniform(-1, 1, size=(input_leght, n_hidden))

    v = np.sum(W**2) / n_classes
    const = ( 1.3 / (np.sqrt(1 + input_leght*v**2)) ) 
    
    return const * r

def nguyanatal(input_leght, n_hidden, random_state=None, n_classes=None):
    
    np.random.seed(random_state)
    
    W = np.random.uniform(-1, 1, size=(input_leght, n_hidden))
    
    beta = 0.7**(1/input_leght)
    gamma = np.sum(W**2) ** 0.5
    
    return (beta/gamma) * W

def f_activation(x, activaction_func):
    if activaction_func == 'relu':
        return np.maximum(x, 0, x) 
    if activaction_func == 'tanh':
        print(x)
        print(x.max(), x.min())
        return np.tanh(x)
    if activaction_func == 'sigmoid':
        return sc.special.expit(x)
    
def inverse_ReLU(x):
    xx = np.where(x > 0, x, 0)
    pos = np.where(x <= 0, 1, 0)
    rand = np.random.uniform(-1, 0, size=x.shape)    
    aux = pos * rand
    
    return aux + xx

def af_inverse(x, activaction_func):
    if activaction_func == 'relu':
        return inverse_ReLU(x)
    if activaction_func == 'tanh':
        return np.arctanh(x)
    if activaction_func == 'sigmoid':
        return sc.special.logit(x)
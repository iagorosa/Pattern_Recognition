#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:42:53 2020

@author: iagorosa
"""

#%%
import numpy as np
import pandas as pd
from PIL import Image as pil
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor, ELMClassifier 

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

import ClassELM as elm
import aux_functions as af

from itertools import product, combinations

from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.datasets import mnist

import leitura_geral as lg
from sklearn.model_selection import train_test_split 

import leitura_geral as lg
        
#%%

#X = np.loadtxt("./dataset/smallNORB/smallNORB-32x32.csv", delimiter=',', skiprows=1)
# col 0: label
#X_train = X[:, 1:]
#y_train = X[:, 0]

#XX = np.loadtxt("./dataset/smallNORB/smallNORB-32x32(test).csv", delimiter=',', skiprows=1)
# col 0: label
#X_test = XX[:, 1:]
#y_test = XX[:, 0]

#%%

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]).astype('float')
y_train = y_train.astype('int')

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]).astype('float')
y_test = y_test.astype('int')
#%%

# Normalização

X_train = X_train / 255.0
X_test  = X_test / 255.0

#%%

#X = lg.leitura('umist')
X = lg.leitura('umist').astype('float')
X[:, 2:] = X[:, 2:] / 255.0

X_train,X_test,y_train,y_test=train_test_split(X[:, 2:],X[:,0],test_size=1.0/3,random_state=None)

pl.hist(y_train)
pl.hist(y_test)
#%%

db = lg.Databases('yale')
(X_train, y_train), (X_test, y_test) = db.load_train_test(pTrain=8, conf_op=10)
#%%
######### SETTING

func_hidden_layer = af.normal_random_layer
random_state = 10
activation_func='tanh'
n_hidden=1000

celm = elm.ELMClassifier(n_hidden=n_hidden, activation_func='sigmoid', func_hidden_layer = func_hidden_layer, bias=True, random_state= random_state, regressor = 'ls_dual', degree=3, lbd=2)
cmelm = elm.ELMMLPClassifier(n_hidden=n_hidden, activation_func='relu', func_hidden_layer = func_hidden_layer, random_state= random_state, regressor = 'pinv')

skt_elm = ELMClassifier(n_hidden=n_hidden, random_state=random_state, activation_func=activation_func, binarizer=LabelBinarizer(0, 1))
#r2 = 0
#%%
######### TRAIN

celm.fit(X_train, y_train)
cmelm.fit(X_train, y_train)
#skt_elm.fit(X_train, y_train)

#%%
######### TEST


r1 = celm.predict(X_test, y_test)[0]
r2 = cmelm.predict(X_test, y_test)[0]
#r3 = skt_elm.score(X_test, y_test)
r3=0
    
#%%

print('r1:', r1, '\nr2:', r2, '\nr3:', r3)

#%%

for img0 in X[X[:,1] == 20][:,2:]:
    print(img0.shape)
    m0 = img0.reshape(112,92).T
    img = pil.fromarray(m0, 'L')
    img.show()
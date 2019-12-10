#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:32:31 2019

@author: iagorosa
"""

#%%
import numpy as np
from PIL import Image as pil
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor, ELMClassifier 

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
        
#%%

X = np.loadtxt("./Yale_32x32.csv", delimiter=',', skiprows=1)
# col 0: pessoa
# col 1: label

#scaler = MinMaxScaler()
scaler = MaxAbsScaler()

X_ = scaler.fit_transform(X[:, 2:].T) 

X[:, 2:] = X_.T 

#%%
y = X[:, 1]


v = 1
clsf = ELMClassifier()
#clsf.fit(X[:v, 2:], y[:v])
clsf.fit(X[X[:, 0] == v][2:], X[X[:, 0] == v][:, 1])

print()

print(clsf.predict(X[v:, 2:]))

#print()

print(y[v:])

print('Score:', clsf.score(X[v:, 2:], y[v:]))

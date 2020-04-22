#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:27:16 2020

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
        
#%%

X = np.loadtxt("./Yale_32x32.csv", delimiter=',', skiprows=1)
# col 0: pessoa
# col 1: label

#%%

# Normalização

X[:, 2:] = X[:, 2:] / 255.0


#%%
random_state = None
n_person  = len(np.unique(X[:, 0])) 
n_classes = len(np.unique(X[:, 1])) 

np.random.seed(random_state)

v = np.arange(1, n_person+1)
CV_random_matrix = np.zeros((n_classes, n_person))

for i in range(n_classes):
    np.random.shuffle(v)
    CV_random_matrix[i] = v
    
CV_random_matrix = CV_random_matrix.astype(int)

#%%

# col 0: pessoa
# col 1: label
k_fold = 5

X_train = CV_random_matrix[:, :5]

#np.delete(CV_random_matrix, 1, 0)
X_folds = []

for i in range(int(n_person/k_fold)):
#    X_new.append([])
    Xi = []
    for j in range(n_classes):
        aux = X[X[:, 1] == j]
        for k in CV_random_matrix[j, i*k_fold : i*k_fold+k_fold]:
            Xi.append(aux[aux[:, 0] == k][0])
    
    X_folds.append(Xi)
    
X_folds = np.array(X_folds)

#%%

'''
valores = range(int(n_person/k_fold))            
comb =  list(combinations(valores, 3))
print(comb)
'''
func_hidden_layer = af.nguyanatal
random_state = 10
activation_func='tanh'
hidden_layer = 20


#clsf.fit(X[:v, 2:], y[:v])
#clsf.fit(X_train, y_train)


celm = elm.ELMClassifier(hidden_layer, activation_func='relu', func_hidden_layer = func_hidden_layer, bias=True, random_state= random_state)
cmelm = elm.ELMMLPClassifier(hidden_layer, activation_func='relu', func_hidden_layer = func_hidden_layer, random_state= random_state)

skt_elm = ELMClassifier(hidden_layer, random_state=random_state, activation_func=activation_func, binarizer=LabelBinarizer(0, 1))

results = [[[], [], []], [[], [], []]]

for i in range(X_folds.shape[0]):
    
    val = list(range(int(n_person/k_fold)))
    val.remove(i)
    
    X_train = np.concatenate([X_folds[val[0]], X_folds[val[1]]])
    X_test = X_folds[i]
    
    y_train = X_train[:, 1]
    y_test  = X_test[:, 1]
    
    X_train = X_train[:, 2:]
    X_test  = X_test[:, 2:]

    celm.fit(X_train, y_train)
    r1 = celm.predict(X_test, y_test)
    
    cmelm.fit(X_train, y_train)
    r2 = cmelm.predict(X_test, y_test)
    
    skt_elm.fit(X_train, y_train)
    r3 = skt_elm.score(X_test, y_test)
    
    results[0][0].append(r1[0])
    results[0][1].append(r2[0])
    results[0][2].append(r3)
    
    celm.fit(X_test, y_test)
    r1 = celm.predict(X_train, y_train)
    
    cmelm.fit(X_test, y_test)
    r2 = cmelm.predict(X_train, y_train)
    
    skt_elm.fit(X_test, y_test)
    r3 = skt_elm.score(X_train, y_train)
    
    results[1][0].append(r1[0])
    results[1][1].append(r2[0])
    results[1][2].append(r3)

print(results)
#%%

df = pd.DataFrame([])
n = int(n_person/k_fold)

df['treino'] = n*[0] + n*[1] # informa se o id_df (identificador do DataFrama X_fold) foi usado para treino ou para teste
df['id_df'] = list(range(n))*2 # identifacador do DataFrame utilizado de X_fold

df['ELM']  = 0 # coluna dos resultados do ELM comum
df['MELM'] = 0 # coluna dos resultados do ELM multicamadas
df['SELM'] = 0
    
for i in range(len(results)):
    aux = np.transpose(results[i])
    df.iloc[i*n:i*n+n, 2:5] = aux 
    
#%%


print(df)    

    #%%

cmelm = elm.ELMMLPClassifier(activation_func='tanh', func_hidden_layer = af.uniform_random_layer, random_state=10)


cmelm.fit(X_train, y_train)

cmelm.predict(X_test, y_test)

#%%

celm = elm.ELMClassifier(n_hidden=1000, activation_func='relu', func_hidden_layer = af.uniform_random_layer, random_state=10, regressor='ls_dual', lbd = 0.3, degree = 2)


celm.fit(X_train, y_train)

p = celm.predict(X_test, y_test)

print(p[0])


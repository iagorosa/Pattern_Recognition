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

#%%

# Para reconstruir a imagem, precisa passar X para uint8
'''
for i in range(11):
    m0 = X[10+i*11, 2:].reshape(32,32).T
#    m0 = m[i].reshape(32,32).T
    img = pil.fromarray(m0, 'L')
    img.show()
'''

#%%

#scaler = MinMaxScaler()
scaler = MaxAbsScaler()

X_ = scaler.fit_transform(X[:, 2:].T)


X[:, 2:] = X_.T

#%%

###### DEFINIÇÕES DE TREINO E TESTE ##############

v = 1

X_train = X[X[:, 0] != v][:, 2:]
y_train = X[X[:, 0] != v][:,  1]

X_test  = X[X[:, 0] == v][:, 2:]
y_test  = X[X[:, 0] == v][:,  1]

#%%
#### ELM DEFINIDO PELO SKLEARN  


clsf = ELMClassifier(random_state=5)
#clsf.fit(X[:v, 2:], y[:v])
clsf.fit(X_test, y_test)

print()

print(clsf.predict(X_test))

#print()

#print(y[v:])

print('Score:', clsf.score(X_test, y_test))

#%%

################# ELM IMPLENTÇÃO DO KAGGLE  ################
# https://www.kaggle.com/robertbm/extreme-learning-machine-example #

# funções
def input_to_hidden(x, Win): # produto pesos de entrada com neuronios de pesos aleatorios + função de ativação
    a = np.dot(x, Win)
    a = np.maximum(a, 0, a) # ReLU
    return a

def predict(x, Win, Wout):
    x = input_to_hidden(x, Win)
    y = np.dot(x, Wout)
    return y

#%%


try:
    y_train.shape[1]
except:
    labels = y_train.astype('int')
    CLASSES = len(labels)
    y_train = np.zeros([labels.shape[0], CLASSES])
    for i in range(labels.shape[0]):
            y_train[i][labels[i]] = 1
#y_train.view(type=np.matrix)


#%%

INPUT_LENGHT = X_train.shape[1] # 784 
HIDDEN_UNITS = 7

Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS])
print('Input Weight shape: {shape}'.format(shape=Win.shape))

#%%

### MODIFICACAO

def weight_func(X, Win, d):
    v = np.average(X)
    const = ( 1.3 / (np.sqrt(1 + d*v**2)) ) 
    return const * Win
    
#%%
    
### MODIFICACAO

W = np.random.uniform(-1, 1, size=(INPUT_LENGHT, HIDDEN_UNITS))

Win = weight_func(X_train, W, HIDDEN_UNITS)


#%%

XX = input_to_hidden(X_train, Win)
Xt = np.transpose(XX)
Wout = np.dot(np.linalg.inv(np.dot(Xt, XX)), np.dot(Xt, y_train))
print('Output weights shape: {shape}'.format(shape=Wout.shape))



#%%

y = predict(X_test, Win, Wout)
correct = 0
total = y.shape[0]
for i in range(total):
    predicted = np.argmax(y[i])
    test = np.argmax(y_test[i])
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: {:f}'.format(correct/total))


#%%
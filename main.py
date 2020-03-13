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

import ClassELM as elm
import aux_functions as af

        
#%%

X = np.loadtxt("./Yale_32x32.csv", delimiter=',', skiprows=1)
# col 0: pessoa
# col 1: label

#%%

# Para reconstruir a imagem, precisa passar X para uint8

#for i in range(11):
#    m0 = X[10+i*11, 2:].reshape(32,32).T
#    img = pil.fromarray(m0, 'L')
#    img.show()
    
for x in X[X[:, 1] == 8]:
#    m0 = X[10+i*11, 2:].reshape(32,32).T
    m0 = x[2:].reshape(32,32).T
    img = pil.fromarray(m0, 'L')
    img.show()
    
'''
labels = {
         'centerlight':  0,
         'glasses':      1,
         'happy':        2,
         'leftlight':    3,
         'noglasses':    4,
         'normal':       5,
         'rightlight':   6,
         'sad':          7,
         'sleepy':       8,
         'surprised':    9,
         'wink':        10
         }
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

y_train_real = y_train.copy()

#%%
#### ELM DEFINIDO PELO SKLEARN  


clsf = ELMClassifier(random_state=100)
#clsf.fit(X[:v, 2:], y[:v])
clsf.fit(X_train, y_train)

Htreino = clsf._get_weights()

print()

print(clsf.predict(X_test))

#print()

#print(y[v:])

print('Score:', clsf.score(X_test, y_test))

Hteste = clsf._get_weights()

#%%

################# ELM IMPLENTÇÃO DO KAGGLE  ################
# https://www.kaggle.com/robertbm/extreme-learning-machine-example #

# funções

def relu(a):
    return np.maximum(a, 0, a) 

def input_to_hidden(x, W, B, bias = True): # produto pesos de entrada com neuronios de pesos aleatorios + função de ativação
    if bias:
        B_mat = np.tile(B, (x.shape[0], 1)) # repete a linha varias vezes
        a = np.dot(x, W) + B_mat
    else:
        a = np.dot(x, W) 
    a = relu(a) # ReLU
    return a

def predict(x, W, beta, B, bias):
    x = input_to_hidden(x, W, B, bias)
    y = np.dot(x, beta)
    return y

#%%


try:
    y_train.shape[1]
except:
    labels = y_train.astype('int')
    CLASSES = len(np.unique(labels))
    y_train = np.zeros([labels.shape[0], CLASSES])
    for i in range(labels.shape[0]):
            y_train[i][labels[i]] = 1
#y_train.view(type=np.matrix)


#%%

INPUT_LENGHT = X_train.shape[1] # 784 
HIDDEN_UNITS = 20

np.random.seed(20)

W = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS],)
print('Input Weight shape: {shape}'.format(shape=W.shape))

B = np.random.normal(size=[HIDDEN_UNITS]) #Bias

#%%

### MODIFICACAO

def weight_func(X, W, d):
    v = np.average(X)
    const = ( 1.3 / (np.sqrt(1 + d*v**2)) ) 
    return const * W
    
#%%
    
### MODIFICACAO

Wuni = np.random.uniform(-1, 1, size=(INPUT_LENGHT, HIDDEN_UNITS))

W = weight_func(X_train, Wuni, HIDDEN_UNITS)


#%%

bias = False

H = input_to_hidden(X_train, W, B, bias)
Ht = np.transpose(H)
#beta = np.dot(np.linalg.inv(np.dot(Ht, H)), np.dot(Ht, y_train))
beta = np.dot(np.linalg.pinv(np.dot(Ht, H)), np.dot(Ht, y_train))

print('Output weights shape: {shape}'.format(shape=beta.shape))



#%%

y = predict(X_test, W, beta, B, bias)
correct = 0
total = y.shape[0]
r_test = []
for i in range(total):
    predicted = np.argmax(y[i])
#    test = np.argmax(y_test[i])
    test = y_test[i]
    r_test.append(predicted)
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: {:f}'.format(correct/total))
print(r_test)


#%%

def inverse_ReLU(x):
    xx = np.where(x > 0, x, 0)
    pos = np.where(x <= 0, 1, 0)
    rand = np.random.uniform(-1, 0, size=x.shape)    
    aux = pos * rand
    
    return pos, rand, aux + xx


##### MULTICAMADAS

# necessario calcular o beta na parte anterior

# H1 = T beta*
H1 = np.dot(y_train, np.linalg.pinv(beta)) 

# He = [H 1s], onde 1s eh vetor linha de 1 
He = np.concatenate([H, np.ones(H.shape[0]).reshape(-1, 1)], axis=1)

#Whe = np.dot(np.linalg.pinv(He), inverse_ReLU(H1))

pos, rand, H1i = inverse_ReLU(H1)
Whe = np.dot(np.linalg.pinv(He), H1i)


Wh = Whe[:-1, :]
B1  = Whe[-1, :]

H2 = input_to_hidden(H, Wh, B1, bias)
beta2 = np.dot(np.linalg.pinv(H2), y_train)

#%%

#saida

p = np.dot(input_to_hidden(X_test, W, B), Wh) + B1
f = np.dot(relu(p), beta2)


#%%


#y = predict(X_test, W, f, B)
y = f
correct = 0
total = y.shape[0]
r_test = []
for i in range(total):
    predicted = np.argmax(y[i])
#    test = np.argmax(y_test[i])
    test = y_test[i]
    r_test.append(predicted)
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: {:f}'.format(correct/total))
print(r_test)


#%%

celm = elm.ELMClassifier(activation_func='relu', func_hidden_layer = af.SCAWI, bias=True, random_state=20)


celm.fit(X_train, y_train)

celm.predict(X_test, y_test)


#%%

cmelm = elm.ELMMLPClassifier(activation_func='relu', func_hidden_layer = af.uniform_random_layer, random_state=20)


cmelm.fit(X_train, y_train)

cmelm.predict(X_test, y_test)

#%%

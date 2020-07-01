#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:25:47 2020

@author: iagorosa
"""

from leitura_geral import Databases
import ClassELM as elm
import aux_functions as af
import numpy as np


db = Databases('yale')

#%%

#(X_train, y_train), (X_test, y_test) = db.load_train_test(pTrain=6, conf_op=10)
#
#print(len(X_train))
#print(len(X_test))
#print(len(X_train) / (len(X_test) + len(X_train)))

#%%

hidden_layer = 1000
activation_func = 'relu'
func_hidden_layer = af.uniform_random_layer
regressor='pinv'
degree=3
lbd=0.5
random_state = 13

erros_elm  = []
erros_melm = []

for i in range(1, 51):
    (X_train, y_train), (X_test, y_test) = db.load_train_test(pTrain=7, conf_op=i)
    
    celm = elm.ELMClassifier(n_hidden=hidden_layer, activation_func=activation_func, func_hidden_layer = func_hidden_layer, bias=True, random_state= random_state, regressor=regressor, degree=degree, lbd=lbd)
    cmelm = elm.ELMMLPClassifier(n_hidden=hidden_layer, activation_func='relu', func_hidden_layer = func_hidden_layer, random_state= random_state, regressor=regressor)
    
    celm.fit(X_test, y_test)
    r1 = celm.predict(X_train, y_train)
    erros_elm.append(1- r1[0])
    
    cmelm.fit(X_test, y_test)
    r2 = cmelm.predict(X_train, y_train)
    erros_melm.append(1 - r2[0])
    

media_erro_elm  = np.mean(erros_elm)
devpad_elm = np.std(erros_elm)

media_erro_melm = np.mean(erros_melm)
devpad_melm = np.std(erros_melm)


print("Media erro ELM:", media_erro_elm)
print("Desvio padrao do erro ELM:", devpad_elm)

print("Media erro MELM:", media_erro_melm)
print("Desvio padrao do erro MELM:", devpad_melm)

    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 19:11:28 2020

@author: iagorosa
"""

from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

import numpy as np

import csv 

import sys
sys.path.insert(1, './../')

from ClassELM import * 

ELMClassifier

#%%

# LEITURA DE ARQUIVOS
with open('housing.csv', newline='\n') as csvfile:
    arq = csv.reader(csvfile, delimiter=',')
    dados = list(arq)

#%%

# Transofrmando em array
  
colunas = dados[0]
dados = np.array(dados)[1:].astype(float)
dados

#%%

# Seprando os conjuntos
X = dados[:, :3]
y = dados[:, -1]

#%%

# Separando em treino e teste

X_treino,X_teste,y_treino,y_teste=train_test_split(X,y,test_size=0.33,random_state=50)


#%%

#Construindo modelo
reg = LinearRegression()

# Treinando o modelo
reg.fit(X_treino, y_treino)

# Testando o modelo
reg.predict(X_teste)

# Verificando acerto
reg.score(X_teste, y_teste)
#%%

#elm = ELMRegressor(n_hidden = 100, activation_func='sigmoid')
elm = ELMRegressor()

elm.fit(X_treino, y_treino)

y_pred_elm = elm.predict(X_teste)

elm.score(X_teste, y_teste)

#res_elm = (y_teste - y_pred_elm)

# %%


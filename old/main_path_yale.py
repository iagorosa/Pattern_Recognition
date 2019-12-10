#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:10:33 2019

@author: iagorosa
"""
#%%
import numpy as np
from PIL import Image as pil
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor, ELMClassifier

#%% 

#LEITURA DE TODOS OS ARQUIVOS DA PASTA
def name_files(mypath):

    arqs = [f for f in listdir(mypath) if ( isfile(join(mypath, f)) and f not in arq_ignore ) ]

    return arqs

# FUNCAO DE PRODUTO DAS DIMENSOES DA IMAGEM
def prod(x):
    return x[0] * x[1]

#%%

path = 'yalefaces/'
arq_ignore = ['Readme.txt']

arqs = name_files(path)  # LEITURA DAS IMAGENS

arr = []  # INICIALIZACAO DO VETOR DAS IMAGENS
rot = []  # INFORMACAO DAS PESSOAS E LABEL

# CRIACAO DO VETOR COM TODAS AS IMAGENS
for i, arq in enumerate(arqs):

    img = pil.open('./'+path+arq)
    aux_arr = np.array(img, dtype='int')
    aux_arr = aux_arr.reshape((1, prod(aux_arr.shape)))[0]

    if i == 0:
        print(aux_arr.shape)

#    arr.append([aux_arr[:10].tolist(), *arq.split('.')])
    arr.append(aux_arr.tolist())
    rot.append([*arq.split('.')])
    
#%%

#arr = np.array(arr)
#class_dict = dict(zip(np.unique(arr[:,2]), range(len(np.unique(arr[:,2])))))
rot = np.array(rot)
class_dict = dict(zip(np.unique(rot[:,1]), range(len(np.unique(rot[:,1])))))

print('Quantidade de rotulos:', len(np.unique(rot[:,1])))
print('Quantidade de pessoas:', len(np.unique(rot[:,0])))

X = np.array(arr) # TRANSFORMANDO VETOR DAS IMAGENS EM ARRAY
y = rot[:, 1]     # PEGANDO APENAS OS LABELS 

y = np.vectorize(class_dict.get)(y)  # TRANSFORMANDO LABELS EM NUMEROS

#a = ELMRegressor()
#a.fit(X, y)

v = -11
clsf = ELMClassifier()
clsf.fit(X[:v], y[:v])

print()

print(clsf.predict(X[v:]))
print(y[v:])

print('Socore:', clsf.score(X[v:], y[v:]))

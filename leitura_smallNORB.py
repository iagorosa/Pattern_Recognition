#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 11:56:33 2019

@author: iagorosa
""" 

import scipy.io as scio
from PIL import Image as pil
import numpy as np
import pandas as pd

mat = scio.loadmat('./dataset/mnist/mnist.mat') #leitura de arquivo .mat no python.
mat_test = scio.loadmat('./dataset/mnist/mnist.t.mat') #leitura de arquivo .mat no python.
                                     #Retorna um dicionario com 5 chaves onde  
m = mat['Z'] # a chave 'Z' no dicionario mat contem uma imagem por linha em um vetor com 32x32=1024 posicoes por imagem m_test

y = mat['y']

m_test = mat_test['Z']
y_test = mat_test['y']

#%%

mm = np.concatenate([y, m], axis=1).astype('uint8')
mm_test = np.concatenate([y_test, m_test], axis=1).astype('uint8') 

#%%

#m2 = zip(m, list(range(1, 16))*11, list(labels.values())*15)
#m3 = list(m2)
#m4 = pd.DataFrame(m3)


np.savetxt("./dataset/smallNORB/smallNORB-32x32.csv", mm, header='label', comments = '', delimiter=',', fmt="%8u") #salva imagens em um csv
#, fmt="%d"

np.savetxt("./dataset/smallNORB/smallNORB-32x32(test).csv", mm_test, header='label', comments = '', delimiter=',', fmt="%8u") #salva imagens em um csv
#%%
# Caso queira visualizar qualquer imagem do vetor, descomente abaixo. Mude a posicao de m[] para uma imagem diferentes.
'''
for i in range(11):
    m0 = m[10+i*11].reshape(32,32).T
#    m0 = m[i].reshape(32,32).T
    img = pil.fromarray(m0, 'L')
    img.show()
'''    
#%%
''' 
for i in range(11):
    m0 = mm[10+i*11, 2:].reshape(32,32).T
#    m0 = m[i].reshape(32,32).T
    img = pil.fromarray(m0, 'L')
    img.show()
''' 
    
#%%
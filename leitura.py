#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 12:15:33 2019

@author: iagorosa
""" 

import scipy.io as scio
from PIL import Image as pil
import numpy as np
import pandas as pd

mat = scio.loadmat('Yale_32x32.mat') #leitura de arquivo .mat no python.
                                     #Retorna um dicionario com 5 chaves onde  
m = mat['fea'] # a chave 'fea' no dicionario mat contem uma imagem por linha em um vetor com 32x32=1024 posicoes por imagem 
p = mat['gnd']


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



#person = np.array(list(range(1, 16))*11).reshape(165,1)
person = np.array([int(i//len(labels)+1) for i in range(len(m))]).reshape(165, 1)
label = np.array(list(labels.values())*15).reshape(165,1)
mm = np.concatenate([person, label, m], axis=1).astype('uint8')


#m2 = zip(m, list(range(1, 16))*11, list(labels.values())*15)
#m3 = list(m2)
#m4 = pd.DataFrame(m3)


np.savetxt("Yale_32x32.csv", mm, header='person,label', comments = '', delimiter=',', fmt="%8u") #salva imagens em um csv
#, fmt="%d"
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
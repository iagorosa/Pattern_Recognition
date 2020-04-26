#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 00:58:17 2020

@author: iagorosa
"""

import scipy.io as scio
from PIL import Image as pil
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from itertools import chain


databases = ['Yale', 'ORL', 'UMist', 'GTface', 'PIE', 'Letter', 'Shuttle', 'USPS', 'MNIST']

def leitura(db):
    db = db.lower()
    if db in [x.lower() for x in ['Yale', 'ORL']]:
        mat = scio.loadmat('./dataset/'+db+'_32x32'+'.mat')

        m = mat['fea'] # a chave 'fea' no dicionario mat contem uma imagem por linha em um vetor com 32x32=1024 posicoes por imagem 
        p = mat['gnd']
        
        label = np.array(list(range(len(p[p==1])))*p[-1][0]).reshape(m.shape[0],1)
#        mm = np.concatenate([p, label, m], axis=1).astype('uint8')
#        np.savetxt("./dataset/"+db+".csv", mm, header='person,label', comments = '', delimiter=',', fmt="%8u") 
        
    elif db == 'umist':
        mat = scio.loadmat('./dataset/'+db+'_cropped.mat')
        
        m = mat['facedat'][0]
        p = mat['dirnames'][0]
        
        
        label =  [i for j in range(m.shape[0]) for i in range(m[j].shape[2])]
        label = np.array(label)
        label = label.reshape(-1, 1)
    
        p = [[i]*m[i].shape[2] for i in range(m.shape[0])]
        p = np.array(list(chain(*p)))
        p = p.reshape(-1, 1)
        
#        m = m.reshape(m.shape[0], m.shape[1]*m.shape[2]).astype('float')
#        m = np.concatenate([m[i].reshape(m[i].shape[2], m[i].shape[0]*m[i].shape[1]).astype('float') for i in range(m.shape[0])]) #TODO: Acho que ta errado
        m = [m[i][j].reshape(1,-1)[0].astype('float').tolist() for i in range(m.shape[0]) for j in range(m[i].shape[2])]
        
        
#        label = np.array(list(p[0])*int(len(m)/len(p)))
    
    mm = np.concatenate([p, label, m], axis=1).astype('uint8')
    np.savetxt("./dataset/"+db+".csv", mm, header='person,label', comments = '', delimiter=',', fmt="%8u") 
    
    return mm
        
        
#    elif db == 'MNIST':
#        (X_train, y_train), (X_test, y_test) = mnist.load_data()




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
from sklearn.model_selection import train_test_split 
import sys

'''
    Aqui existem várias implementações de leitura dos dados disponiveis. 
    Quando o arquivo está em formato do matlab, está com extensão .mat e usa-se a biblioteca do scipy para conseguir realizar a leitura.
    Algumas bases são apenas carregadas de bibliotecas especificas
    O método load_train_test faz o carregamento dos dados em treino e teste. A forma de fazê-lo depende das possibilidades da base de dados. 
        O parâmetro load_type controla essa forma de fazer a leitura. Para as bases 'Yale', 'ORL', 'YaleB', a load_type = 'standard' executa a leitura usando a pasta 'pTrain' que trás as combinações possíveis para p imagens de cada pessoa como treino. Quando load_type = 'fold', escolhe-se uma determinada quantidade de humor por pessoa. 
    Para utilizar, basta instanciar a classe usando o nome da base de dados e depois carregar a parte de treino e teste conforme queira.
    
'''

# Nome das bases de dados disponíveis 
databases = ['Yale', 'ORL', 'UMist', 'GTface', 'PIE', 'Letter', 'Shuttle', 'USPS', 'MNIST', "smallNORB", 'YaleB']
    
# Classe para trabalhar com leitura e carregamento das bases de dados
class Databases():
    def __init__(self, database_name, save_csv = False):
        self.database_name = database_name
        self.save_csv = save_csv 
        self.load()
        self.X_folds = []
        
        
    def load(self):
        
        db = self.database_name.lower()
        if  db == 'MNIST'.lower():
            (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

            self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1]*self.X_train.shape[2]).astype('float')
            self.y_train = self.y_train.astype('int')
            
            self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1]*self.X_test.shape[2]).astype('float')
            self.y_test = self.y_test.astype('int')
            
            if self.database_name:            
                self.X = np.concatenate([ np.concatenate([self.y_train.reshape(-1,1), self.X_train], axis=1) , np.concatenate([self.y_test.reshape(-1,1), self.X_test], axis=1) ], axis = 0)
                
            return
                        
            
        if db in [x.lower() for x in ['yale', 'orl', 'yaleb']]:
            mat = scio.loadmat('./dataset/'+db+'/'+db+'_32x32'+'.mat')
    
            m = mat['fea'] # a chave 'fea' no dicionario mat contem uma imagem por linha em um vetor com 32x32=1024 posicoes por imagem 
            p = mat['gnd']
            
            print(mat)
            
            try:
                label = np.array(list(range(len(p[p==1])))*p[-1][0]).reshape(m.shape[0],1)
            except:
                label = p
        
            p = p - 1
            
            self.X = np.concatenate([p, label, m], axis=1).astype('uint8')
            
        
        if db == "smallnorb":
#            X_test = np.loadtxt("./dataset/smallNORB/smallNORB-32x32.csv", skiprows=1, dtype=int, delimiter=',')
#            X_train = np.savetxt("./dataset/smallNORB/smallNORB-32x32(test).csv", mm_test, header='label', comments = '', delimiter=',', fmt="%8u")
            
            mat = scio.loadmat('./dataset/smallNORB/smallNORB-32x32.mat') #leitura de arquivo .mat no python.
            mat_test = scio.loadmat('./dataset/smallNORB/smallNORB-32x32.t.mat') #leitura de arquivo .mat no python.
                                                 #Retorna um dicionario com 5 chaves onde  
            m = mat['Z'] # a chave 'Z' no dicionario mat contem uma imagem por linha em um vetor com 32x32=1024 posicoes por imagem m_test
            
            y = mat['y']
            
            m_test = mat_test['Z']
            y_test = mat_test['y']
            
#            self.X_train = np.concatenate([y, m], axis=1).astype('uint8')
#            self.X_test = np.concatenate([y_test, m_test], axis=1).astype('uint8') 
            
            self.X_train = m
            self.y_train = y
            
            self.X_test = m_test
            self.y_test = y_test

       
        '''
        elif db == 'umist':
            mat = scio.loadmat('./dataset/'+db+'/'+db+'_cropped.mat')
            
            m = mat['facedat'][0]
            p = mat['dirnames'][0]
            
            
            label =  [i for j in range(m.shape[0]) for i in range(m[j].shape[2])]
            label = np.array(label)
            label = label.reshape(-1, 1)
        
            p = [[i]*m[i].shape[2] for i in range(m.shape[0])]
            p = np.array(list(chain(*p)))
            p = p.reshape(-1, 1)
            
            
            m = [m[i].T[j].reshape(1,-1)[0].astype('float').tolist() for i in range(m.shape[0]) for j in range(m[i].shape[2])]
        '''
        
        
        
        if self.save_csv:
            np.savetxt("./dataset/"+db+"/"+db+".csv", self.X, header='person,label', comments = '', delimiter=',', fmt="%8u") 
    
            
    def load_train_test(self, pTrain = 2, conf_op = 1, pad = 'max', test_size=1.0/3, random_state=None, load_type="standard", k_fold=5, train_folds=[0]):
        # Quando o load_type == "standard", precisamos dos parâmetros
        # pTrain: Subset aleatório com p=(2,3,...,n) imagens por individuo rotuladas para formar o conjunto de treinamento e o restante é considerado para o conjnto de teste. Para cada p, existem 50 divisões aleátorias (conf_op)
        # conf_op: configuração aleatória dentro de um pTrain. Os valores são de 1 atá 50. 
        
        # Quando o load_type == "folds", precisamos dos parâmetros
        # k_fold: quantidade de separação uniforme que acontecerá na base de dados
        # train_folds: lista de quais pedaços da separação ocorrida por k_folds serão utilizadas para o treinamento.
        
        if load_type == 'standard':
            # load_type == 'standard' significa que o carregamento está sendo realizado de acordo com as divisões de pTrain pre-estabelecidas, quando elas existem. 
        
            if self.database_name.lower()  == 'MNIST'.lower():
                pass
        
            elif self.database_name.lower() in [x.lower() for x in ['Yale', 'ORL', 'YaleB']]:
                
                
                try:
                    mat = scio.loadmat('./dataset/'+self.database_name.lower()+'/pTrain/'+str(pTrain)+'Train/'+str(conf_op)+'.mat')
                except:
                    print('A pasta '+str(pTrain)+'Train ou o \'arquivo ' + str(conf_op)+'.mat\'' + ' nao disponivel.')
                    sys.exit()
                    
                trainIdx = mat['trainIdx']
                self.X_train = self.X[(trainIdx.reshape(1, -1)[0] - 1), :]
                
                testIdx  = mat['testIdx']
                self.X_test  = self.X[(testIdx.reshape(1, -1)[0] - 1), :]
                
                self.y_train = self.X_train[:, 0]
                self.X_train = self.X_train[:, 2:]
                
                self.y_test = self.X_test[:, 0]
                self.X_test = self.X_test[:, 2:]
                
            elif self.database_name.lower() == "smallnorb":
                pass
            
            
        # TODO: TESTAR
        elif load_type == "folds":
            # load_type == 'folds' significa que o carregamento está sendo realizado de acordo com as divisões estabelecidas por Iago e Medina, onde definimos a quantidade de divisões de mesmo tamanho que são realizadas no conjunto de entrada. Com essas partes definidas, usa-se algumas para teste e o resto para treino. É necessário que tenha um par de identificador (pessoa) e classe (humor).
            
            if self.X_folds == []:
                n_person  = len(np.unique(self.X[:, 0])) 
                n_classes = len(np.unique(self.X[:, 1])) 
                
                np.random.seed(random_state)
                
                #v = np.arange(1, n_person+1)
                v = np.arange(n_person)
                CV_random_matrix = np.zeros((n_classes, n_person))
                
                for i in range(n_classes):
                    np.random.shuffle(v)
                    CV_random_matrix[i] = v
                    
                CV_random_matrix = CV_random_matrix.astype(int)
                
                # col 0: pessoa
                # col 1: label
#                k_fold = 5
                
#                X_train = CV_random_matrix[:, :k_fold]
                
                #np.delete(CV_random_matrix, 1, 0)
    #            self.X_folds = []
                
                for i in range(int(n_person/k_fold)):
                #    X_new.append([])
                    Xi = []
                    for j in range(n_classes):
                        aux = self.X[self.X[:, 1] == j]
                        for k in CV_random_matrix[j, i*k_fold : i*k_fold+k_fold]:
                            Xi.append(aux[aux[:, 0] == k][0])
                    
                    self.X_folds.append(Xi)
                    
                self.X_folds = np.array(self.X_folds)
            
            if(all(train_folds < k_fold) and len(train_folds) < k_fold):
                
                train_folds = set(train_folds)
                test_folds = set(range(k_fold)).difference(train_folds)  # diferença de conjuntos para ver quais pedaços ficarao no X_test
                
                self.X_train = np.concatenate([self.X_folds[i] for i in train_folds], axis=0)[:, 2:]
                self.y_train = self.X_train[:, 0]
                
                self.X_test  = np.concatenate([self.X_folds[i] for i in test_folds], axis=0)[:, 2:]
                self.y_test  = self.X_test[:, 0]
            
            else:
                print("Algum valor na lista train_folds é maior que ", k_fold, " ou a lista é maior que k_folds")
                
                
        # implementar de acordo com a separação da biblioteca
        elif load_type == "train_test_split":
            pass
        
        if pad != None:
            self.padronizacao(pad)
            
        return (self.X_train, self.y_train), (self.X_test, self.y_test)
            
    def padronizacao(self, pad):
        if pad == 'max':
            self.X_train = self.X_train / 255.0
            self.X_test  = self.X_test / 255.0
            
        
        
    

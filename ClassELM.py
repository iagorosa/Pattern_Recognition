#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:23:42 2020

@author: iagorosa
"""

#%%

'''
    Aqui são construidas as classes do ELM para classificação
    São consideradas duas estruturas: 
        ELMClassifier: ELM padrão com uma camada oculta
        ELMMLPClassifier: ELM com Multicamadas (funcionando parcialmente)
    A classe BaseELM rege todo o funcionamento do ELM e funciona como uma superclasse para as outras duas citadas acima. A estrutura é como a seguir:
        
                            BaseELM
                                '
                     -----------------------
                     '                     '
                 ELMClassifier      ELMMLPClassifier
    
    O parâmetro func_hidden_layer recebe uma função para os pesos entre a entrada e a camada oculta. Todas as funções possíveis se encontram em aux_function.py, onde qualquer outra função para os pesos podem ser implementados. 
    O parâmetro activation_func controla o tipo de função de ativação utilizado. As implementações também estão no arquivo aux_func.py
    O parâmetro regressor controla qual o tipo de regressor será utilizado no processo. Existem 3 opções implementadas:
        pinv: pseudoinversa padrão do ELM,
        ls_reg: regressão pelo LS primal (padrão), mas os resultados mostraram que o resultado é equivalente ao método da pseudoinversa.
        ls_dual: regressão pelo LS dual. Este tem 2 parâmetros:
            lbd: termo de regularização
            degree: grau do kernel polinomial
            sparse: resolução com matriz esparsa. Mantenha falso, pois a implementação deste método não sucedeu-se muito bem. A tentativa era usar gradiente conjugado na resolução.
'''

#%%
import numpy as np
from PIL import Image as pil
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelBinarizer

from sklearn.utils.extmath import safe_sparse_dot

import aux_functions as af

import scipy as sc

from sklearn.preprocessing import Binarizer

#%%

class BaseELM():
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh', random_state=None, bias = True, regressor = 'pinv', lbd = 0.01, degree = 1, sparse = False):
        self.n_hidden = n_hidden
        self.func_hidden_layer = func_hidden_layer
        self.random_state = random_state
        self.activation_func = activation_func
        self.bias = bias
        self.regressor = regressor
        self.lbd = lbd
        self.degree = degree
        self.sparse = sparse
        
    def create_random_layer(self):
        return self.func_hidden_layer(self.input_lenght, self.n_hidden, self.random_state, self.n_classes)
    
    def create_bias(self):
        return np.random.normal(size=[self.n_hidden])

    def input_to_hidden(self, X): # produto pesos de entrada com neuronios de pesos aleatorios + função de ativação
        if self.bias:
            B_mat = np.tile(self.B, (X.shape[0], 1)) # repete a linha varias vezes
            a = np.dot(X, self.W) + B_mat
        else:
            a = np.dot(X, self.W) 
            
        return af.f_activation(a, self.activation_func)
        
    def fit(self, X, y):
        
        self.y_bin = self.binarizer.fit_transform(y)  
        
        self.input_lenght = X.shape[1]
        
        self.n_classes = self.y_bin.shape[1]
        
        self.W = self.create_random_layer()
        if self.bias:
            self.B = self.create_bias()
            
        self.H = self.input_to_hidden(X)
#        Ht = self.H.T
        
#        self.beta = np.dot(np.linalg.pinv(np.dot(Ht, self.H)), np.dot(Ht, self.y_bin))
#        self.beta = np.dot(np.linalg.pinv(self.H), self.y_bin)
        
        self.beta = self._regressor(self.H, self.y_bin)
        
        return self.beta
        
    
    def predict(self, y, y_pred):
#        H = self.input_to_hidden(X)
#        y_pred = np.dot(H, self.beta)
        
        correct = 0
        total = y.shape[0]
        r_test = []
        for i in range(total):
            predicted = np.argmax(y_pred[i])
        #    test = np.argmax(y_test[i])
            test = y[i]
            r_test.append(predicted)
            correct = correct + (1 if predicted == test else 0)
        return correct/total, r_test
    
    
    def _get_weights(self):
        return self.W  
    
    def _get_bias(self):
        return self.B
    
    def _get_beta(self):
        return self.beta
    
    def _get_H(self):
        return self.H
    
    def _regressor(self, A, d):
        
        if self.regressor ==  None:
            return np.linalg.solve(A, d )
        
        elif self.regressor == 'pinv':
            return np.dot(sc.linalg.pinv2(A), d )
#            return np.dot( sc.linalg.pinv2(A.T @ A), np.dot(A.T, d) )
        
        elif self.regressor == 'ls':
#            np.linalg.solve(np.dot(A.T, A), np.dot(A.T, d))  #Somar ldb diagonal principal
            return np.linalg.lstsq(A, d, rcond=None)[0]
        
        elif self.regressor == 'ls_reg':
            I = np.identity(A.shape[1])
            print('aqui')
            return np.linalg.solve(np.dot(A.T, A) + self.lbd * I, np.dot(A.T, d)) 
        
        elif self.regressor == 'ls_dual':
            
            if not self.sparse:
                
                
#                FORMA ABAIXO FOI CONSTRUIDA PELO RAUL. PORÉM, PARECE QUE ESTA ERRADA
                
                self.K = (A @ A.T) + 1
                
                I = np.identity(len(self.K)) # identidade
                
                #TODO: preocupacao com bias?
                self.alpha = np.linalg.solve( self.K ** self.degree + self.lbd * I , d)  #TODO: usei solve comum
                
    #            print('alpha:', self.alpha.shape, '\nA:', A.shape)
    #            print((self.alpha.T @ A).shape)
                return (self.alpha.T @ A).T  # Transposto pq na classificacao da incompatibilidade de dimensao
                '''
                
                if self.degree == 1:
                    self.regressor = 'pinv'
                    return self._regressor(A, d)
                
                self.K = A.T @ A + 1
                
                I = np.identity(len(self.K))
                
                
                # Assim nao da pra calcular o beta. As dimensoes nao batem
#                self.alpha = np.linalg.solve( self.K ** self.degree + self.lbd * I , d)
#                return self.alpha @ A.T                
                
                self.beta = np.linalg.solve(self.K ** self.degree + self.lbd * I, np.dot(A.T, d))
                
                self.alpha = sc.linalg.pinv2(A.T) @ self.beta 
                
                return self.beta
                '''
            
            
            else:
#                transformer = Binarizer().fit(A)
#                A_ = transformer.transform(A)
                
                #TODO: conferir com Medina
#                A_ = A**self.degree
                
                A = A.astype('float32')
                
                print("Matriz sparsa:")
#                A_ = sc.sparse.coo_matrix(A)

                def A_X_mult(X):
#                    return X.T @ ( (self.H @ self.H_test.T)  ** self.degree) 
                    return A.dot(X.T)
                
                
                A_X = sc.sparse.linalg.LinearOperator((A.shape*2)[::2], matvec = lambda x: x, matmat = A_X_mult, dtype='float32')                

                def AAT_mult(x):
                	return (A.dot( A.T ) ** self.degree).dot(x)
#                    return A_.dot( A_.T.dot(x) )
                
                def I_lbd(x):
                    return (self.lbd * I).dot(x)
                
                def sistema_mult(x):
#                    return ( (A @ A.T) ** self.degree + self.lbd * I).dot(x)
                    return ( A_X.matmat(A) ** self.degree + self.lbd * I).dot(x)
                
                
                
#                self.K = sc.sparse.linalg.LinearOperator((A.shape*2)[::2], matvec=AAT_mult)
                
                I = sc.sparse.eye(*(A.shape*2)[::2], dtype='int32')
#                I = sc.sparse.linalg.LinearOperator(I.shape, matvec=I_lbd)
                
                
                
                sistema = sc.sparse.linalg.LinearOperator((A.shape*2)[::2], matvec=sistema_mult, dtype='float32')
                
                
                self.d = d
                
                self.alpha = np.zeros_like(d, dtype=float)
                
                print("\n\nInicio da resolucao dos cgs")
                for i in range(d.shape[1]):
                    r, *info = sc.sparse.linalg.cg( sistema , d.T[i])
                    self.alpha.T[i] = r
                    print("cg para posicao ", i, " resolvida")
                    

                result = (self.alpha.T @ A).T 
                
#                self.alpha = self.alpha.toarray()
#                result = result.toarray()
                
                return result                
            
     

class ELMMLPClassifier(BaseELM):
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh',  binarizer=LabelBinarizer(0, 1), random_state=None, regressor = 'pinv', lbd = 0.01, sparse = False):
        
        super(ELMMLPClassifier, self).__init__(n_hidden=n_hidden, func_hidden_layer = func_hidden_layer, activation_func=activation_func, random_state=random_state, bias=True, sparse = sparse)
        
        self.binarizer = binarizer
        
    def fit(self, X, y):
        
#        self.beta = self._get_beta()
#        self.H = self._get_H()
#        print("fit")
        super(ELMMLPClassifier, self).fit(X, y)
        
        self.H1 = np.dot(self.y_bin, np.linalg.pinv(self.beta)) 
        self.He = np.concatenate([self.H, np.ones(self.H.shape[0]).reshape(-1, 1)], axis=1)
        
#        print(self.H1)s
        H1_inv = af.af_inverse(self.H1, self.activation_func)  
        #TODO: Problemas com a aplicacao da funcao inversa
#        print("H1^-1")
#        print(H1_inv)
#        print()
        
        Whe = np.dot(np.linalg.pinv(self.He), H1_inv)
        
        self.W1 = Whe[:-1, :]
        self.B1  = Whe[-1, :]
        
        #TODO: melhorar essa parte. Colocar os calculos de H2 (input_to_hidden) em uma nova funcao?
        B1_mat = np.tile(self.B1, (self.H.shape[0], 1))
        self.H2 = af.f_activation(np.dot(self.H, self.W1) + B1_mat, self.activation_func)
        self.beta2 = np.dot(np.linalg.pinv(self.H2), self.y_bin)
        
        return self.beta2
          

    def predict(self, X, y):

        B1_mat = np.tile(self.B1, (X.shape[0], 1))
        B_mat  = np.tile(self.B, (X.shape[0], 1))

#        print(B_mat.shape, X.shape, self.W.shape)
        f1 = af.f_activation( (np.dot(X,  self.W)  + B_mat), self.activation_func)
        f2 = af.f_activation( (np.dot(f1, self.W1) + B1_mat), self.activation_func)
        
        y_pred = np.dot(f2, self.beta2)
        
        return super(ELMMLPClassifier, self).predict(y, y_pred)
        
        
    def get_weights(self):
        return super(ELMClassifier, self)._get_weights()
    
    def get_bias(self):
        return super(ELMClassifier, self)._get_bias()
        



class ELMClassifier(BaseELM):
    def __init__(self, n_hidden=20, func_hidden_layer = af.normal_random_layer, activation_func='tanh',  binarizer=LabelBinarizer(0, 1), random_state=None, bias = True, regressor = 'pinv', lbd = 0.01, degree = 1, sparse = False):
        
        super(ELMClassifier, self).__init__(n_hidden=n_hidden, func_hidden_layer = func_hidden_layer, activation_func=activation_func, random_state=random_state, bias=bias, regressor = regressor, degree = degree, sparse = sparse)
                
        self.binarizer = binarizer
    
    def get_weights(self):
        return super(ELMClassifier, self)._get_weights()
    
    def get_bias(self):
        return super(ELMClassifier, self)._get_bias()
    
    def predict(self, X, y):
        self.H_test = self.input_to_hidden(X)
#        y_pred = np.dot(H, self.beta)
                    
        y_pred = self.H_test @ self.beta
        
        if self.regressor == 'ls_dual': # False apenas para um teste onde essa parte nao acontece
        
            if self.sparse:
                
                def mat(X):
                    return X.T @ ( (self.H @ self.H_test.T)  ** self.degree) 
                
                mv = lambda x: x
                
                shape = (self.H.shape[0],)*2 
#                print(shape)
                y_pred = sc.sparse.linalg.LinearOperator(shape, matmat=mat, matvec=mv)
                y_pred = y_pred.matmat(self.alpha).T
#                print("y_pred", y_pred.shape)
                
            else:
                
                
                mat = (self.H @ self.H_test.T) + 1
#                mat = self.H.T @ self.H_test
                y_pred = self.alpha.T @ (mat ** self.degree)
                y_pred = y_pred.T
#                print(y_pred.shape)
                
#                pass
                
        
        return super(ELMClassifier, self).predict(y, y_pred)
        
        


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:33:36 2020

@author: iagorosa
"""


from leitura_geral import Databases
import ClassELM as elm
import aux_functions as af
import numpy as np
import pandas as pd

#from sklearn.preprocessing import Binarizer



db = Databases('yaleb')
(X_train, y_train), (X_test, y_test) = db.load_train_test(pTrain=40)
celm = elm.ELMClassifier(n_hidden=1000, activation_func= 'relu',  func_hidden_layer =  af.uniform_random_layer, bias=True, random_state= 10, degree=2, regressor='ls_dual', sparse=False, lbd=0.2)
#
#
#transformer = Binarizer().fit(X_train)
#X_train_ = transformer.transform(X_train)
#
celm.fit(X_train, y_train)
##

#print(X_train.shape, y_train.shape)
t1 = celm.predict(X_train, y_train) 

#print()
#print(X_test.shape, y_test.shape)
r1 = celm.predict(X_test, y_test)

print(t1[0], r1[0])

Xshape = X_test.shape[0] + X_train.shape[0]
print("X shape", Xshape)
print("X_train / X_shape", X_train.shape[0] / Xshape)



#%%

from leitura_geral import Databases
import ClassELM as elm
import aux_functions as af
import numpy as np
import pandas as pd

df = pd.DataFrame([])

#base_names = ['yale', 'orl', 'MNIST', 'smallnorb']
#base_names = ['yale', 'orl']
base_names = ['YaleB']
#base_names = ['smallnorb']
opt_act_func = ['relu', 'sigmoid', 'tanh']
#opt_act_func = ['relu']
opt_func_hidden_layer = [af.uniform_random_layer, af.normal_random_layer, af.SCAWI, af.nguyanatal]
#opt_regressor = ['ls', 'pinv', 'ls_dual'] 
opt_regressor = ['pinv', 'ls_dual'] 
#opt_regressor = ['ls_dual'] 


hidden_layer = 1000
#activation_func = 'relu'
#func_hidden_layer = af.uniform_random_layer
#regressor='pinv'
degree=2
lbd=0.2
random_state = 13




for name in base_names:
    db = Databases(name)
#    (X_train, y_train), (X_test, y_test) = db.load_train_test()
    
    for oaf in opt_act_func:
        
        for reg in opt_regressor:
        
            for ofhl in opt_func_hidden_layer:
                erros_elm_train  = []
                erros_melm_train = []
                
                erros_elm_test  = []
                erros_melm_test = []
                
                conf = {"nome_database": name, "activaction_func": oaf, "func_hidden_layer": ofhl.__name__, "regressor": reg}
                print("Iniciando as configurações {1}, {2} e {3} para a base {0}".format(*conf.values()) )
                
                for i in range(1, 51):
                    print("\rTeste {}/50".format(i), end="\r")
                    
                    (X_train, y_train), (X_test, y_test) = db.load_train_test(pTrain=40, conf_op=i)
                    
                    try:
#                        celm = elm.ELMClassifier(n_hidden=hidden_layer, activation_func=oaf, func_hidden_layer = ofhl, bias=True, random_state= random_state, regressor=reg, degree=degree, lbd=lbd)
                        celm = elm.ELMClassifier(n_hidden=hidden_layer, activation_func=oaf, func_hidden_layer = ofhl, bias=True, random_state= i, regressor=reg, sparse=False, degree=degree, lbd=lbd)
                        
                        celm.fit(X_train, y_train)
                        
                        t1 = celm.predict(X_train, y_train)  # erro treinamento
                        r1 = celm.predict(X_test, y_test)    # erro teste
                        
                    except:
                        erros_elm_train = [np.nan]
                        erros_elm_test = [np.nan]
                        if i == 1:
                            break
                        else:
                            print("Verificar esse problema")
                    
#                    try:
#                        cmelm = elm.ELMMLPClassifier(n_hidden=hidden_layer, activation_func=oaf, func_hidden_layer = ofhl, random_state= random_state, regressor=reg)
#                        
#                        cmelm.fit(X_train, y_train)
#                        
#                        t2 = cmelm.predict(X_train, y_train)  # erro treinamento        
#                        r2 = cmelm.predict(X_test, y_test) # erro test
#                        
#                    except:
#                        erros_melm_train = [np.nan]
#                        erros_melm_test = [np.nan]
#                        if i == 1:
#                            break
#                        else:
#                            print("Verificar esse problema")
                    
    
                    erros_elm_train.append(1- t1[0])
                    erros_elm_test.append(1- r1[0])
                    
#                    erros_melm_train.append(1- t2[0])
#                    erros_melm_test.append(1 - r2[0])

                    del celm
                    
                    
                
                
                media_erro_elm_train  = np.mean(erros_elm_train)
                devpad_elm_train = np.std(erros_elm_train)
                
                media_erro_elm_test  = np.mean(erros_elm_test)
                devpad_elm_test = np.std(erros_elm_test)


#                media_erro_melm_train  = np.mean(erros_melm_train)
#                devpad_melm_train = np.std(erros_melm_train)
#                
#                media_erro_melm_test  = np.mean(erros_melm_test)
#                devpad_melm_test = np.std(erros_melm_test)
                
                #del celm
            
#                dict_aux = {"nome_database": name, "activaction_func": oaf, "func_hidden_layer": ofhl, "regresso": reg, "media_erro_elm_train": media_erro_elm_train, "devpad_elm_train": devpad_elm_train, "media_erro_elm_test": media_erro_elm_test, "devpad_elm_test": devpad_elm_test, "media_erro_melm_train": media_erro_melm_train, "devpad_melm_train": devpad_melm_train, "media_erro_melm_test": media_erro_melm_test, "devpad_melm_test": devpad_melm_test}
                
                dict_aux = {"nome_database": name, "activaction_func": oaf, "func_hidden_layer": ofhl.__name__, "regressor": reg, "media_erro_elm_train": media_erro_elm_train, "devpad_elm_train": devpad_elm_train, "media_erro_elm_test": media_erro_elm_test, "devpad_elm_test": devpad_elm_test}
                
                if df.empty:
                    df = pd.DataFrame(columns=list(dict_aux.keys()))
                df = df.append(dict_aux, ignore_index=True)

df.to_csv("yaleb_new.csv")

#print("Media erro ELM:", media_erro_elm)
#print("Desvio padrao do erro ELM:", devpad_elm)
#
#print("Media erro MELM:", media_erro_melm)
#print("Desvio padrao do erro MELM:", devpad_melm)

    
    

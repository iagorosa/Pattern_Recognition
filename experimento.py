#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:33:36 2020

@author: iagorosa, ruanmedina
"""


from leitura_geral import Databases
import ClassELM as elm
import aux_functions as af
import numpy as np
import pandas as pd

df = pd.DataFrame([])

base_names = ['yale', 'orl', 'yaleb']
opt_act_func = ['relu', 'sigmoid', 'tanh']
opt_func_hidden_layer = [af.uniform_random_layer, af.normal_random_layer, af.SCAWI, af.nguyanatal]
opt_regressor = ['pinv', 'ls_dual'] 


hidden_layer = 1000
degree=2
lbd=0.2
random_state = 13

qtd_comb = 50 # este parametro controla a quantidade de configurações de imagens que serao escolhidas dentro de cada pasta pTrain. O padraão é escolher sempre todas as 50 combinações da pasta.


for name in base_names:
    db = Databases(name)

    pTrain = 40 if name.lower() == 'yaleb' else 7 # A confiração de pTrain (p padrões ou humores por pessoa para treino) para Yale e ORL foi 7, onde ~64% da base da Yale e 70% da base da ORL ficaram para treinamento. Já para a base Yale B, foram utilizaos pTrain = 40, totalizando ~62% das informaçoes da base para treinamento.
    
    for oaf in opt_act_func:
        
        for reg in opt_regressor:
        
            for ofhl in opt_func_hidden_layer:
                erros_elm_train  = []
                erros_melm_train = []
                
                erros_elm_test  = []
                erros_melm_test = []
                
                conf = {"nome_database": name, "activaction_func": oaf, "func_hidden_layer": ofhl.__name__, "regressor": reg}
                print("Iniciando as configurações {1}, {2} e {3} para a base {0}".format(*conf.values()) )
                
                for i in range(1, qtd_comb):
                    print("\rTeste {}/50".format(i), end="\r")
                    
                    (X_train, y_train), (X_test, y_test) = db.load_train_test(pTrain=pTrain, conf_op=i)
                    
                    try:
                        celm = elm.ELMClassifier(n_hidden=hidden_layer, activation_func=oaf, func_hidden_layer = ofhl, bias=True, random_state= random_state, regressor=reg, degree=degree, lbd=lbd)
#                        celm = elm.ELMClassifier(n_hidden=hidden_layer, activation_func=oaf, func_hidden_layer = ofhl, bias=True, random_state= i, regressor=reg, sparse=False, degree=degree, lbd=lbd)
                        
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
                                      
                    #acuracias
                    erros_elm_train.append(1- t1[0])
                    erros_elm_test.append(1- r1[0])
                    
#                    erros_melm_train.append(1- t2[0])
#                    erros_melm_test.append(1 - r2[0])

                    del celm
                    
                    
                
                
                media_erro_elm_train  = np.mean(erros_elm_train)
                devpad_elm_train = np.std(erros_elm_train)
                
                media_erro_elm_test  = np.mean(erros_elm_test)
                devpad_elm_test = np.std(erros_elm_test)

                
                dict_aux = {"nome_database": name, "activaction_func": oaf, "func_hidden_layer": ofhl.__name__, "regressor": reg, "media_erro_elm_train": media_erro_elm_train, "devpad_elm_train": devpad_elm_train, "media_erro_elm_test": media_erro_elm_test, "devpad_elm_test": devpad_elm_test}
                
                if df.empty:
                    df = pd.DataFrame(columns=list(dict_aux.keys()))
                df = df.append(dict_aux, ignore_index=True)

df.to_csv("resultados.csv")


    

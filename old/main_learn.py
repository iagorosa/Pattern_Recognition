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
def name_files(mypath):

    arqs = [f for f in listdir(mypath) if ( isfile(join(mypath, f)) and f not in arq_ignore ) ]

    return arqs

def prod(x):
    return x[0] * x[1]

path = 'yalefaces/'
arq_ignore = ['Readme.txt']

# img = pil.open(path+"subject01.centerlight")
#img.show() 

#arr = np.array(img)
#img2 = pil.fromarray(arr, 'L')
#img2.show()

# arqs, person, labels = name_files(path)
arqs = name_files(path)

arr = []
rot = []
for i, arq in enumerate(arqs):
    img = pil.open('./'+path+arq)
    aux_arr = np.array(img, dtype='int')
    if i == 0:
        print(aux_arr.shape)
    aux_arr = aux_arr.reshape((1, prod(aux_arr.shape)))[0]

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

X = np.array(arr)
y = rot[:, 1]

y = np.vectorize(class_dict.get)(y)

#a = ELMRegressor()
#a.fit(X, y)

v = -11
clsf = ELMClassifier()
clsf.fit(X[:v], y[:v])


#%%

from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer


def make_datasets():
    """

    :return:
    """

    return [make_moons(n_samples=200, noise=0.3, random_state=0),
            make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
            make_linearly_separable()]


def make_classifiers():
    """

    :return:
    """

    names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]

    nh = 10

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)

    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
    srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
    srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    # use gaussian RBF
    srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
    log_reg = LogisticRegression()

    classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                   GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                   GenELMClassifier(hidden_layer=srhl_sinsq),
                   GenELMClassifier(hidden_layer=srhl_tribas),
                   GenELMClassifier(hidden_layer=srhl_hardlim),
                   GenELMClassifier(hidden_layer=srhl_rbf)]

    return names, classifiers

def make_linearly_separable():
    """

    :return:
    """

    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)

datasets = make_datasets()
names, classifiers = make_classifiers()

        
X, y = datasets[0]

a = ELMRegressor()

a.fit(X, y)

#%%

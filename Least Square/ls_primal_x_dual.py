# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 01:59:54 2020

@author: ia
"""
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split 
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
#%%
dados=pd.read_csv('housing.csv',sep=',',header=0)
dados.head()

#%%
###analise descritiva 
dados.corr()
x=dados['RM']
y=dados['LSTAT']
z=dados['MEDV']
w=dados['PTRATIO']

#%%
#####diagramas de dispersao da variavel(y) dependente com as demais indepe
plt.scatter(z,x,color='b')
plt.title('MEDV vs RM')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()
plt.scatter(z,y,color='red')
plt.title('MEDV vs LSTAT')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()
plt.scatter(z,w,color='orange')
plt.title('MEDV vs PTRATIO')
plt.xlabel('PTRATIO')
plt.ylabel('MEDV')
plt.show()

#%%
######estudo de normalidade da variavel resposta
plt.hist(z,bins=30)
stats.normaltest(z)

#%%
######Divisao da base de dados e classe
atributos=['RM','LSTAT','PTRATIO']
treino=dados[atributos]
treino.head()
classe=dados['MEDV']
classe.head()
X_treino,X_teste,y_treino,y_teste=train_test_split(treino,classe,test_size=0.33,random_state=50)
X_treino.head()
y_treino.head()

#%%

def padronizacao_dados(X_treino, y_treino, X_teste, y_teste, tipo = None):
    
    if tipo == 'minmax':
        scaler = MinMaxScaler(feature_range=(0.01, 1))
        y_treino = scaler.fit_transform(y_treino.reshape(1,-1).T).T[0]
        y_teste = scaler.fit_transform(y_teste.reshape(1,-1).T).T[0]
        
        scaler = MinMaxScaler()
        X_treino[:,:-1] = scaler.fit_transform(X_treino[:,:-1])
        X_teste = scaler.fit_transform(X_teste)
        
        
        
    return X_treino, y_treino, X_teste, y_teste
        

#%%

## DEFINICOESh

X_treino.loc[:, 'const'] = 1.0 # adiciona o bias no treinamento

A = X_treino.to_numpy() # transforma df X_treino em numpy array
d = y_treino.to_numpy() # transforma df y_treino em numpy array

X_t = X_teste.to_numpy() # transforma df X_teste em numpy array
y_t = y_teste.to_numpy() # transforma df y_teste em numpy array

A, d, X_t, y_t = padronizacao_dados(A, d, X_t, y_t, tipo = 'minmax')

mse = lambda x: sum( x ** 2 ) / (2 * len (x)) # funcao para o caluculo do mse
                                              # que recebe como argumento o vetor das diferencas
                                              # entre y - ŷ
                                              


#%%

''' CRIANDO O MODELO LS PRIMAL ''' 
                                                 
# Solucao de w = (A^T A)^-1 A^T y
w = np.linalg.inv(A.T @ A) @ A.T @ d  

w_ = w[:-1] # retirando o valor referente ao bias 


#%%

# y_predito eh igual a X_teste . w_ somado ao valor do bias (w[-1])
y_pred = X_t @ w_.T + w[-1]

# diferenca y - ŷ em porcentagem
res = (y_t - y_pred)

print(mse(res))

#%%

# Grafico comparativo y_teste e y_pred
plt.figure(figsize=(10,8))
plt.plot(range(len(y_t)), y_t, 'bo')
plt.plot(range(len(y_pred)), y_pred, 'rx')

#%%

'''
LS DUAL COM REGULARIZACAO
'''

# funcao para caluclar o somatorio de alpha * <x_i, x_j> para cada nova observacao
def ls_dual(X, alpha):
    y_pred = []
    for x in X:
        norma = (x @ x.T) 
        y_ = sum( norma * alpha) 
        y_pred.append(y_ / len(alpha)) # apenas divide por len(alpha) para ficar em porcentagem
    
    return y_pred

#%%

### LS DUAL COM REGLARIZACAO - valor unico

# definicoes
A_ = A[:, :-1] # retira a coluna do bias 
K = A_ @ A_.T
n = len(K)
I = np.identity(len(K)) # identidade
lbd = 1

# calculo de alpha = (K + lambda*I) . y
alpha = np.linalg.inv( K + lbd * I ) @ d 

# valores de y predito pelo ls dual calculado na funcao ls_dual()
y_pred_dual = ls_dual(X_t, alpha)

# diferenca y - ŷ em porcentagem
res_dual = (y_t - y_pred_dual)

print(mse(res_dual))

#%%

plt.figure(figsize=(10,8))
plt.plot(range(len(y_t)), y_t, 'bo')
plt.plot(range(len(y_pred)), y_pred_dual, 'rx')

#%%

### LS DUAL COM REGLARIZACAO - repeticao


lbd = [1, 8] # intervalo de valores de lambda
y_pred_dual = {} # dicionario de resultados de y_pred com ls dual
res_dual = {} # dicionario de resultados a diferenca percentual de y - ŷ

for i in range(*lbd):

    alpha = np.linalg.inv( K + i * I ) @ d
    
    y_pred_dual[i] = ls_dual(X_t, alpha)
    
    res_dual[i] = (y_t - y_pred_dual[i]) 
    
    print("lambda="+str(i)+":", mse(res_dual[i]))
    
#%%
    
plt.figure(figsize=(10,8))
plt.plot(range(len(y_t)), y_t, 'bo')
plt.plot(range(len(y_pred_dual[2])), y_pred_dual[2], 'rx')


#%%
'''
#%%
######construindo o modelo de regressao com ls forma fechada
#reg = sm.OLS(y_treino,X_const,data=df).fit()
reg = sm.OLS(y_treino,X_treino.iloc[:, :-1], data=X_treino.iloc[:, :-1]).fit()

print(reg.summary())

#%%
##### avaliacao do modelo
yhat=reg.predict(X_teste)
res=y_teste-yhat
res
plt.hist(res,bins=40)



#%%

####homocedasticidade
plt.scatter(res,yhat,color='red')
plt.title('residuos vs valores preditos')
plt.xlabel('valores preditos')
plt.ylabel('Residuos')
plt.show()


#%%
coefs = pd.DataFrame(reg.params)
coefs.columns = ['Coeficientes']
print(coefs)

#%%
######## Implementacao dual do ls#########
#v=np.transpose(X_const).dot(X_const) #### produto interno
v= A.T @ A
v.head()
w=np.linalg.inv(v)
print(w)
t=w.dot(np.transpose(X_const))
print(t)
beta=t.dot(y_treino)
print(beta)

'''

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 01:59:54 2020

@author: iagorosa
"""
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split 
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor

import scipy.stats as scs
from statsmodels.stats.diagnostic import lilliefors
import pylab as pl
import seaborn as sns

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

D= pd.DataFrame(dados.describe())
D.loc['skewness', :] = scs.skew(dados)
D.loc['kurtosis', :] = scs.kurtosis(dados, fisher=False)

D.to_csv('describe.csv')

#%%

pl.figure(figsize=(10,8))
sns.heatmap(dados.corr(), linewidths=.5, annot=True)
pl.title("Matrix de Correlação")
pl.savefig("matriz_correlacao.png")
pl.close()

g = sns.pairplot(dados)
g.fig.suptitle('Pairplot dos Dados')
pl.savefig("paiplot.png")
pl.close()

#%%

pl.figure(figsize=(10,8))
dados.iloc[:, :-1].boxplot()
pl.grid(axis='x')
pl.title("Boxplot dos Dados")
pl.savefig('boxplot.png')
pl.show()
pl.close()

#%%

for i in range(len(dados.columns[:-1])):

    pl.figure(figsize=(10,8))
    Y = dados.iloc[:, i]
    Y.hist(histtype='bar', density=True, ec='black', zorder=2)
    
    min_ = int(round(Y.min()-0.5))
    max_ = int(round(Y.max()+0.5))
    
    pl.xticks(range(min_, max_, round((max_-min_)/10+0.5)))
    
    pl.xlabel(Y.name)
    pl.ylabel("Frequência Relativa (%)")
    
    pl.title("Histograma " + Y.name)
    pl.grid(axis='x')
    
    # estatistica
    mu, std = scs.norm.fit(Y)
    
    # Plot the PDF.
    xmin, xmax = pl.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scs.norm.pdf(x, mu, std)
    pl.plot(x, p, 'r--', linewidth=2)
    
    print(mu, std)
    print(x)
    
    # Teste de hipotese de normalidade com 5% de significancia:
    # H0: A amostra provem de uma população normal
    # H1: A amostra nao provem de uma distribuicao normal
    
    # Testes de shapiro e lillefors: 
    s   = scs.shapiro(Y)
    lil = lilliefors(Y)
    
    ymin, ymax = pl.ylim()
    pl.text(xmin+xmin*0.01, ymax-ymax*0.12, 'Shapiro: '+str(round(s[1], 5) )+'\nLilliefors: '+str(round(lil[1], 5)), bbox=dict(facecolor='red', alpha=0.4), zorder=4 )
    pl.tight_layout()
    pl.savefig('teste_hipotese_'+Y.name+'.png')
    pl.show()
    pl.close()
    
pl.figure(figsize=(10,8))
Y = dados.iloc[:, -1]
Y.hist(histtype='bar', density=True, ec='black', zorder=2)

min_ = int(round(Y.min()-0.5))
max_ = int(round(Y.max()+0.5))

pl.xticks(range(min_, max_, round((max_-min_)/10+0.5)))

pl.xlabel(Y.name)
pl.ylabel("Frequência Relativa (%)")

pl.title("Histograma " + Y.name)
pl.grid(axis='x')
pl.tight_layout()
pl.savefig("histograma_"+Y.name+".png")
pl.show()
pl.close()


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

## DEFINICOES

X_treino.loc[:, 'const'] = 1.0 # adiciona o bias no treinamento

A  = X_treino.to_numpy(copy=True) # transforma df X_treino em numpy array
d  = y_treino.to_numpy(copy=True) # transforma df y_treino em numpy array

X_t = X_teste.to_numpy(copy=True) # transforma df X_teste em numpy array
y_t = y_teste.to_numpy(copy=True) # transforma df y_teste em numpy array

tipo = 'minmax' # tipo de padronização 
                # use None para nenhuma padronização
                
porcentagem = False # se True, calculo da diferenca entre y - ŷ em porcentagem

A, d, X_t, y_t = padronizacao_dados(A, d, X_t, y_t, tipo = tipo)

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

if porcentagem == True:
    res = res / y_t

print(mse(res))

#%%

# Grafico comparativo y_teste e y_pred
plt.figure(figsize=(12,8))
plt.plot(range(len(y_t)), y_t, 'bo', label='y')
plt.plot(range(len(y_pred)), y_pred, 'rx', label='ŷ')
pl.legend(fontsize=12)
pl.title('y x ŷ', fontsize=20)
pl.xlabel("observação", fontsize=14)
pl.ylabel("valor", fontsize=14)
pl.tight_layout()
pl.savefig("predicao_linear.png", dpi=200)
pl.close()

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

'''
# definicoes
#A_ = A[:, :-1] # retira a coluna do bias 
#K = A_ @ A_.T
    
K = A @ A.T
n = len(K)
I = np.identity(len(K)) # identidade
lbd = 0

# calculo de alpha = (K + lambda*I) . y
alpha = np.linalg.inv( K + lbd * I ) @ d 

# valores de y predito pelo ls dual calculado na funcao ls_dual()
y_pred_dual = ls_dual(X_t, alpha)

# diferenca y - ŷ em porcentagem
res_dual = (y_t - y_pred_dual)

if porcentagem == None:
    res_dual = res_dual / y_t

print(mse(res_dual))

'''

K = A @ A.T
n = len(K)
I = np.identity(len(K)) # identidade
lbd = 0.1
# forma de fazer sem bias

# calculo de alpha = (K + lambda*I) . y
#alpha = np.linalg.inv( K ** degree + lbd * I ) @ d 
alpha = np.linalg.solve( K + lbd * I , d) 


w_dual = alpha @ A  # deve ser igual ao w do dual

y_pred_dual = X_t @ w_dual[:-1].T + w_dual[-1]

res_dual = (y_t - y_pred_dual)

mse(res_dual)

#%%

# forma de fazer com bias

X_t_bias = np.concatenate([X_t, np.ones((len(X_t),1))], axis=1)

mat = A @ X_t_bias.T

y_pred_dual_bias = alpha @ mat

res_dual_bias = (y_t - y_pred_dual_bias)

mse(res_dual_bias)

# %%

# forma de fazer com bias - não linear

degree = 4

alpha = np.linalg.solve( K ** degree + lbd * I , d) 

X_t_bias = np.concatenate([X_t, np.ones((len(X_t),1))], axis=1)

mat_nl = A @ X_t_bias.T

y_pred_dual_nl = alpha @ mat_nl ** degree

res_dual_nl = (y_t - y_pred_dual_nl)

mse(res_dual_nl)

#%%

plt.figure(figsize=(10,8))
plt.plot(range(len(y_t)), y_t, 'bo')
plt.plot(range(len(y_pred)), y_pred_dual, 'rx')


#%%

### LS DUAL COM REGLARIZACAO - repeticao

degrees = [1, 7]
X_t_bias = np.concatenate([X_t, np.ones((len(X_t),1))], axis=1)

dt = 0.05
lambdas = [0+dt, 10+dt, dt] # intervalo de valores de lambda
y_pred_dual = [] # dicionario de resultados de y_pred com ls dual
res_dual = [] # dicionario de resultados a diferenca percentual de y - ŷ
mse_dual = []

for degree in range(*degrees):
    y_pred_dual.append((degree, {}))
    res_dual.append((degree, {}))
    mse_dual.append((degree, []))

    for lbd in np.arange(*lambdas):
    
        alpha = np.linalg.solve( K ** degree + lbd * I , d) 
        
        mat_nl = A @ X_t_bias.T
        
        y_pred_dual[-1][1][i] = alpha @ mat ** degree
        
        res_dual[-1][1][i] = (y_t - y_pred_dual[-1][1][i]) 
        
        mse_dual[-1][1].append(mse(res_dual[-1][1][i]))
        
        print("lambda="+str(lbd)+":", mse_dual[-1][1][-1])
    
#%%
    
plt.figure(figsize=(12,8))

for i in range(len(mse_dual)):
    pl.plot(np.arange(*lambdas), mse_dual[i][1], 'o', markersize=2, label=mse_dual[i][0])

pl.title('MSE em relação ao valor de $\lambda$', fontsize=20)
pl.xlabel("$\lambda$", fontsize=14)
pl.ylabel("MSE", fontsize=14)
pl.xticks(fontsize=12)
pl.yticks(fontsize=12)
pl.legend()
pl.grid()
pl.tight_layout()
pl.savefig("variacao_MSE_dual.png", dpi=200)
    
#%%
    
plt.figure(figsize=(12,8))
plt.plot(range(len(y_t)), y_t, 'bo', label='y')
plt.plot(range(len(y_pred_dual[2])), y_pred_dual[2], 'rx', label='ŷ')
pl.legend(fontsize=12)
pl.title('y x ŷ ($\lambda = 2$)', fontsize=20)
pl.xlabel("observação", fontsize=14)
pl.ylabel("valor", fontsize=14)
pl.tight_layout()
pl.savefig("predicao_nao_linear_dual.png", dpi=200)


#%%

###ELM regressao########

#t=np.array(X_treino['x'])
#print(t)
#elm=ELMRegressor(activation_func='sigmoid',alpha=0.1,n_hidden=2,regressor=None)

n_hidden = [10, 110]
activation_func = ['gaussian', 'sigmoid']

df_elm = pd.DataFrame(columns=activation_func)

for n in range(*n_hidden, 1):
    for af in activation_func:

        elm = ELMRegressor(n_hidden = n, activation_func=af)
        #ELMRegressor()
        
        elm.fit(A, d)
        
        y_pred_elm = elm.predict(X_t)
        
        elm.score(A, d)
        
        res_elm = (y_t - y_pred_elm)
        
        df_elm.loc[n, af] = mse(res_elm)
        
        print(mse(res_elm))

df_elm.to_csv('df_elm.csv')
#%%

elm = ELMRegressor(n_hidden = 10, activation_func='gaussian')
elm.fit(A, d)
y_pred_elm = elm.predict(X_t)
        

plt.figure(figsize=(12,8))
plt.plot(range(len(y_t)), y_t, 'bo', label = 'y')
plt.plot(range(len(y_pred_elm)), y_pred_elm, 'rx', label = 'ŷ')
pl.legend(fontsize=12)
pl.title('y x ŷ (n_hidden = 10 e activation_func = Gaussian)', fontsize=20)
pl.xlabel("observação", fontsize=14)
pl.ylabel("valor", fontsize=14)
pl.tight_layout()
pl.savefig("predicao_elm.png", dpi=200)


#%%

plt.figure(figsize=(12,8))
df1 = df_elm[df_elm.iloc[:, 0] < 100]
pl.plot(df1.index, df1.iloc[:, 0], 'bo', markersize=12, label=df_elm.columns[0])
df2 = df_elm[df_elm.iloc[:, 1] < 100]
pl.plot(df2.index, df2.iloc[:, 1], 'ro', markersize=12, label=df_elm.columns[1])
pl.legend()
pl.title('MSE em Relação a Quantidade de Nós na Camada Interna', fontsize=20)
pl.xlabel("n_hidden", fontsize=14)
pl.ylabel("MSE", fontsize=14)
pl.xticks(fontsize=12)
pl.yticks(fontsize=12)
pl.grid()
pl.tight_layout()
pl.savefig("variacao_MSE_elm.png", dpi=200)
pl.show()
pl.close()

#%%

# KNN para Classificação 

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# essa linha significa que estamos criando um objeto do classificador KNN (K-Nearest Neighbors) 
# da biblioteca scikit-learn. O parâmetro n_neighbors=3 indica que o modelo irá considerar 
# os 3 vizinhos mais próximos para fazer a classificação de um ponto de dados.
knn = KNeighborsClassifier(n_neighbors=3)  

# nas linhas abaixo , estamos carregando um conjunto de dados de frutas a partir de um arquivo 
# chamado 'fruit_data_with_colors.txt' usando a função read_table do pandas.
data = pd.read_table('fruit_data_with_colors.txt')

# nas linhas abaixo, estamos separando os dados em variáveis independentes (x) 
# e variável dependente (y), para que possamos treinar o modelo de classificação.
x = data[['mass', 'width', 'height', 'color_score']]
y = data['fruit_label']

# nas linhas abaixo, estamos dividindo os dados em conjuntos de treinamento e teste.
# A função train_test_split do scikit-learn é usada para dividir os dados em duas partes:
# 1. Conjunto de treinamento (train): usado para treinar o modelo.
# 2. Conjunto de teste (test): usado para avaliar o desempenho do modelo.

# o X maiusculo geralmente utilizado em aprendizado de máquina, pois o X maiúsculo é o conjunto
# de atributos que normalmente é uma matriz e na matemática utilizamos letras maiúsculas para
# representar matrizes e o y minúsculo porque ele é um vetor de rótulos.
X_train, X_test, y_train, y_test = train_test_split(x,y)

# nestas linhas abaixo
print(knn.fit(X_train,y_train))
print(knn.score(X_test,y_test))
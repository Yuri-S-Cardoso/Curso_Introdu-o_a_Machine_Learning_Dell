# Regressão

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

temperatura = np.array([30, 25, 36, 18, 25, 29, 30, 33, 37, 31, 26, 37, 29, 26, 30, 31, 34, 38])
numero_sorvetes = np.array([20, 12, 50, 10, 18, 25, 26, 32, 48, 22, 16, 52, 24, 20, 28, 29, 35, 40])

# cria um DataFrame com os dados de temperatura e número de sorvetes vendidos, 
# Data frame é uma estrutura de dados bidimensional, como uma tabela, que pode armazenar diferentes tipos de dados em colunas.
df = pd.DataFrame({'Temperatura': temperatura, 'Numero de Sorvetes': numero_sorvetes}) 
#print(df.head()) # printa as 5 primeiras linhas do DataFrame

plt.plot(df['Temperatura'], df['Numero de Sorvetes'], 'o') # plota os dados de temperatura e número de sorvetes vendidos
plt.xlabel('Temperatura (°C)') # adiciona o rótulo do eixo x
plt.ylabel('Sorvetes') # adiciona o rótulo do eixo y
plt.show() # exibe o gráfico

x = df['Temperatura'].to_numpy() # converte a coluna 'Temperatura' do DataFrame em um array numpy
y = df['Numero de Sorvetes'].to_numpy() # converte a coluna 'Numero de Sorvetes' do DataFrame em um array numpy

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2) # divide os dados em conjuntos de treino e teste, com 20% dos dados reservados para teste

modelo = LinearRegression() # cria um modelo de regressão linear
modelo.fit(x_treino.reshape(-1, 1), y_treino.reshape(-1, 1)) # treina o modelo com os dados de treino

y_previsto = modelo.predict(x_teste.reshape(-1, 1)) # faz previsões com os dados de teste

plt.plot(range(y_previsto.shape[0]), y_previsto, 'r--' ) # plota as previsões do modelo em vermelho tracejado
plt.plot(range(y_teste.shape[0]), y_teste, 'g--') # plota os valores reais dos dados de teste em verde tracejado
plt.legend(['Sorvetes previstos', 'Sorvetes vendidos']) # adiciona uma legenda ao gráfico
plt.xlabel('Índice') # adiciona o rótulo do eixo x
plt.ylabel('Sorvetes') # adiciona o rótulo do eixo y    
plt.show() # exibe o gráfico
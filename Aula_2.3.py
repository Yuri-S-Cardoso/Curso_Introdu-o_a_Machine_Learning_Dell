import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

knn = KNeighborsClassifier(n_neighbors=3)  

data = pd.read_table('fruit_data_with_colors.txt')

x = data[['mass', 'width', 'height', 'color_score']]
y = data['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(x,y)

mm = MinMaxScaler()

# estamos usando este comando para pegar os dados de train e transformando cada coluna 
# em dados entre 0 e 1 e jogando na própria variável de conjuntode train
X_train = mm.fit_transform(X_train)

# Nessa linha, estamos utilizando apenas "mm.transform". Nesta  segunda linha, estamos 
# apenas transformando os dados de acordo com a escala efetuada nos dados de treino, pois 
# os dados de treino serão utilizados para preparar  e  construir  o  modelo.  Então,  
# toda  transformação  deve  ser  baseada  na transformação executada nos dados de treino
X_test = mm.transform(X_test)
print(X_test)

knn = KNeighborsClassifier(n_neighbors=3)

print(knn.fit(X_train,y_train))
print(knn.score(X_test,y_test))
print(knn.predict(X_test))
print(y_test)
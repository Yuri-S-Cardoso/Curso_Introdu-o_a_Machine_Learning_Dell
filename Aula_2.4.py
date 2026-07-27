#knn para Regressão

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston

knn = KNeighborsClassifier(n_neighbors=3)  

data = pd.read_table('fruit_data_with_colors.txt')

x = data[['mass', 'width', 'height', 'color_score']]
y = data['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(x,y)

mm = MinMaxScaler()

X_train = mm.fit_transform(X_train)

X_test = mm.transform(X_test)
print(X_test)

knn = KNeighborsClassifier(n_neighbors=3)

#print(knn.fit(X_train,y_train))
#print(knn.score(X_test,y_test))
#print(knn.predict(X_test))
#print(y_test)

knn = KNeighborsRegressor(n_neighbors=3) 
data = load_boston()
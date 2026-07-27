import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

knn = KNeighborsClassifier(n_neighbors=3)  

data = pd.read_table('fruit_data_with_colors.txt')

x = data[['mass', 'width', 'height', 'color_score']]
y = data['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(x,y)

mm == MinMaxScaler()

# esta liha 
X_train = mm.fit_transform(X_train)
# KNN = k-Nearest Neighbors
# É um modelo muito intuitivo, pois se vale da ideia simples de classificar elementos 
# ou executar a regressão de alguns elementos, a partir da distância entre eles usando 
# a distância euclidiana

# Exemplo:

print(2-5) # esse resultado por ser negativo, não reflete uma distância

print ((2-5)**2) # ** significa "elevado a", porque qualquer número elevado ao quadrado é positivo

# distância euclidiana 
print(((2-5)**2)**(0.5)) # **(0.5) significa "raiz quadrada"


# Duas coordenadas
a = [5,0.75] # fruta com massa 5 e cor 0.75
b = [2,0.50] # fruta com massa 2 e cor 0.50

print(((5-2)**2 + (0.75-0.50)**2)**0.5) 
# A distância euclidiana entre os pontos a e b 
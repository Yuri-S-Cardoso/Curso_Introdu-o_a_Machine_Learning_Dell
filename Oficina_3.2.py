import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.shape)
print(df.columns)

X = df[["Age", "Spending Score (1-100)"]].copy()

print(X.head())
print(X.isnull().sum())

erros = []
valores_k = range(1, 11)

for k in valores_k:
    modelo = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )
    
    modelo.fit(X)
    erros.append(modelo.inertia_)

    plt.figure(figsize=(8, 5))
plt.plot(valores_k, erros, marker="o")

plt.title("Método do Cotovelo")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Soma dos erros quadráticos")
plt.xticks(valores_k)
plt.grid()

plt.show()

k_escolhido = 4

kmeans = KMeans(
    n_clusters=k_escolhido,
    init="k-means++",
    n_init=10,
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(X)

print(df.head())

centroides = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Age"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)

plt.scatter(
    centroides[:, 0],
    centroides[:, 1],
    marker="X",
    s=250,
    label="Centróides"
)

plt.title("Agrupamento de clientes com K-means")
plt.xlabel("Idade")
plt.ylabel("Pontuação de gastos")
plt.legend()
plt.grid()

plt.show()

resumo_clusters = df.groupby("Cluster").agg(
    quantidade=("CustomerID", "count"),
    idade_media=("Age", "mean"),
    gasto_medio=("Spending Score (1-100)", "mean")
).round(2)

print(resumo_clusters)

resumo_clusters = df.groupby("Cluster")[
    ["Age", "Spending Score (1-100)"]
].agg(["count", "mean", "min", "max"]).round(2)

print(resumo_clusters)
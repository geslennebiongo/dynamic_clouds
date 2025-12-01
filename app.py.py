import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ===================== CLASSE ===================== #
class DynamicClouds:
    def __init__(self, K=3, ni=5, max_iter=100, tol=1e-6, random_state=None):
        self.K = K
        self.ni = ni
        self.max_iter = max_iter
        self.tol = tol
        if random_state is not None:
            np.random.seed(random_state)

    def _euclidean(self, x, center):
        return np.linalg.norm(x - center, axis=1)

    def fit(self, X):
        N = X.shape[0]
        idx = np.random.choice(N, self.K, replace=False)
        self.centers = X[idx]

        for it in range(self.max_iter):
            distances = np.zeros((N, self.K))
            for i in range(self.K):
                distances[:, i] = self._euclidean(X, self.centers[i])

            labels = np.argmin(distances, axis=1)
            self.labels_ = labels

            new_centers = []
            for i in range(self.K):
                cluster_points = X[labels == i]
                if len(cluster_points) == 0:
                    new_centers.append(X[np.random.choice(N)])
                else:
                    d = self._euclidean(cluster_points, self.centers[i])
                    nearest = cluster_points[np.argsort(d)[: min(self.ni, len(cluster_points))]]
                    new_centers.append(nearest.mean(axis=0))

            new_centers = np.array(new_centers)
            shift = sum(np.linalg.norm(self.centers[i] - new_centers[i]) for i in range(self.K))
            self.centers = new_centers

            if shift < self.tol:
                break

        return self

    def inertia(self, X):
        inertia = 0
        for i in range(self.K):
            pts = X[self.labels_ == i]
            if len(pts) > 0:
                inertia += self._euclidean(pts, self.centers[i]).sum()
        return inertia

    def plot_clusters(self, X):
        # PCA pour réduire à 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        centers_2d = pca.transform(self.centers)

        fig, ax = plt.subplots(figsize=(7,5))
        for i in range(self.K):
            cluster_points = X_2d[self.labels_ == i]
            ax.scatter(cluster_points[:,0], cluster_points[:,1], label=f'Cluster {i+1}', s=50)
            ax.scatter(centers_2d[i,0], centers_2d[i,1], marker='X', s=200, c='black')

        ax.set_title("Nuées Dynamiques - Visualisation PCA 2D")
        ax.set_xlabel("Composante principale 1")
        ax.set_ylabel("Composante principale 2")
        ax.grid(True)
        ax.legend()
        return fig

# ===================== STREAMLIT UI ===================== #
st.title("☁️ Clustering Nuées Dynamiques avec Visualisation")

# Paramètres interactifs
K = st.slider("Nombre de clusters (K)", 2, 10, 4)
ni = st.slider("Nombre de voisins proches (ni)", 1, 10, 3)

# Chargement de données
uploaded = st.file_uploader("Charger un fichier CSV (facultatif)", type="csv")
if uploaded is not None:
    data = np.loadtxt(uploaded, delimiter=",", skiprows=1)
    st.success(f"Données chargées : {data.shape[0]} individus, {data.shape[1]} features")
else:
    data = np.random.rand(60,5)
    st.info("Jeu de données aléatoire généré : 60 individus x 5 features")

# Bouton pour exécuter le clustering
if st.button("🚀 Lancer le clustering"):
    model = DynamicClouds(K=K, ni=ni, random_state=42)
    model.fit(data)

    st.subheader("📌 Résultats")
    st.write("**Clusters assignés :**", model.labels_)
    st.write("**Centres finaux :**", model.centers)
    st.write("**Inertie intra-classe :**", model.inertia(data))

    # Graphe
    fig = model.plot_clusters(data)
    st.pyplot(fig)

# Test de stabilité
st.subheader("🔁 Test de stabilité sur 10 runs")
if st.button("Tester la stabilité"):
    partitions = []
    for i in range(10):
        m = DynamicClouds(K=K, ni=ni)
        m.fit(data)
        partitions.append(m.labels_)

    partitions = np.array(partitions)
    st.write("Partitions obtenues :")
    st.dataframe(partitions)

    stable = all(np.array_equal(partitions[0], p) for p in partitions[1:])
    st.write("✅ Identique sur tous les runs ?", stable)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("spotify_tracks.csv")

# -------------------------------
# 2. Select Audio Features
# -------------------------------
features = df[['danceability','energy','loudness','speechiness',
               'acousticness','instrumentalness','liveness',
               'valence','tempo']]

features = features.dropna()

# -------------------------------
# 3. Feature Scaling
# -------------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -------------------------------
# 4. Elbow Method
# -------------------------------
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# -------------------------------
# 5. Apply K-Means
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# -------------------------------
# 6. PCA
# -------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# -------------------------------
# 7. SHOW BOTH OUTPUTS TOGETHER
# -------------------------------
plt.figure(figsize=(14,6))

# ---- Elbow Method Plot ----
plt.subplot(1, 2, 1)
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

# ---- PCA Cluster Plot ----
plt.subplot(1, 2, 2)
plt.scatter(pca_result[:,0], pca_result[:,1], c=clusters, cmap='tab10')
plt.title("Music Genre Clustering (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.tight_layout()
plt.show()

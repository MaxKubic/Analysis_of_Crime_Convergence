import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data_cleaning import clean_gdp, clean_crime

# =========================
# 1. SETTINGS
# =========================

selected_offences = [
    "Intentional homicide",
    "Robbery",
    "Burglary of private residential premises",
    "Unlawful acts involving controlled drugs or precursors"
]

os.makedirs("../outputs/clustering", exist_ok=True)

# =========================
# 2. LOAD & CLEAN DATA
# =========================

gdp = clean_gdp("../data/tec00114__custom_20353208_linear.csv")
crime = clean_crime("../data/crim_off_cat__custom_20354171_linear.csv")


# =========================
# 3. PREPARE DATA FOR CLUSTERING
# =========================

def prepare_clustering_data(crime, gdp, selected_offences):
    # filtr trestných činů
    df = crime[crime["offence"].isin(selected_offences)].copy()

    # průměr za celé období (2013–2023)
    df = df.groupby(["country", "offence"])["crime_rate"].mean().reset_index()

    # pivot → sloupce = offence
    df_pivot = df.pivot(index="country", columns="offence", values="crime_rate")

    # průměrné GDP
    gdp_avg = gdp.groupby("country")["gdp_index"].mean()

    # spojení
    df_final = df_pivot.merge(gdp_avg, left_index=True, right_index=True)

    df_final = df_final.dropna()

    return df_final


df_final = prepare_clustering_data(crime, gdp, selected_offences)

print("\nPrepared dataset:")
print(df_final.head())

# =========================
# 4. NORMALIZATION
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_final)

# =========================
# 5. ELBOW METHOD
# =========================

inertia = []

for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 8), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/clustering/elbow_method.png", dpi=200)
plt.close()

# =========================
# 6. FINAL CLUSTERING
# =========================

k = 3  # můžeš změnit podle elbow

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_final["cluster"] = clusters

# =========================
# 7. SAVE RESULTS
# =========================

df_final.to_csv("../outputs/clustering/clusters.csv")

print("\nCluster assignment:")
print(df_final.sort_values("cluster"))

# =========================
# 8. PCA VISUALIZATION (IMPROVED)
# =========================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9,7))

scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)

# lehký offset pro text (aby se nepřekrývaly)
for i, country in enumerate(df_final.index):
    plt.annotate(
        country,
        (X_pca[i,0], X_pca[i,1]),
        textcoords="offset points",
        xytext=(5,5),
        fontsize=8
    )

# legenda clusterů
handles, _ = scatter.legend_elements()
labels = [f"Cluster {i}" for i in range(len(handles))]
plt.legend(handles, labels, title="Clusters")

plt.title("Clustering of EU countries (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()

plt.savefig("../outputs/clustering/pca_clusters.png", dpi=200)
plt.show()
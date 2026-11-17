# -*- coding: utf-8 -*-
"""
Yapay Zeka Teknikleri - Kümeleme Projesi
Veri seti : Wine (sklearn içinde hazır, 178 örnek, 13 özellik, 3 gerçek sınıf)
Modeller  : K-Means, Hiyerarşik (Agglomerative - Ward), DBSCAN, Gaussian Mixture (GMM)
Performans: Silhouette, Davies-Bouldin, Calinski-Harabasz (iç ölçütler)
            ARI, NMI (gerçek etiketlerle karşılaştırma) ve çalışma süresi
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from scipy.cluster.hierarchy import linkage, dendrogram

RS = 42

# 1) Veri
data = load_wine()
X_ham, y_gercek = data.data, data.target
print("Veri boyutu:", X_ham.shape, "| Gerçek sınıf sayısı:", len(np.unique(y_gercek)))

# 2) Standartlaştırma (mesafe tabanlı yöntemler için şart)
X = StandardScaler().fit_transform(X_ham)

# 3) Küme sayısı seçimi (K-Means ile): dirsek + silhouette
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
K = list(ks)[int(np.argmax(sil))]
print(f"\nSilhouette'e göre seçilen küme sayısı: K = {K}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Dirsek Yöntemi (Inertia)")
ax[0].set_xlabel("Küme sayısı (k)"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, "o-", color="tab:green"); ax[1].set_title("Silhouette Skoru")
ax[1].set_xlabel("Küme sayısı (k)"); ax[1].set_ylabel("Silhouette")
plt.tight_layout(); plt.savefig("k_secimi.png", dpi=150); plt.close()

# 4) DBSCAN için eps / min_samples seçimi (silhouette'i en yüksek, gürültüsü %30'dan az olan)
en_iyi, en_iyi_skor = None, -1
for eps in np.arange(1.5, 4.6, 0.1):
    for ms in (3, 5, 8):
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        n_kume = len(set(lab)) - (1 if -1 in lab else 0)
        gurultu = np.mean(lab == -1)
        if n_kume >= 2 and gurultu <= 0.30:
            s = silhouette_score(X[lab != -1], lab[lab != -1])
            if s > en_iyi_skor:
                en_iyi_skor, en_iyi = s, (round(eps, 2), ms)
if en_iyi is None:
    en_iyi = (2.5, 5)
print(f"DBSCAN için seçilen parametreler: eps={en_iyi[0]}, min_samples={en_iyi[1]}")

# 5) Modeller
modeller = {
    "K-Means": lambda: KMeans(n_clusters=K, n_init=10, random_state=RS),
    "Hiyerarşik (Ward)": lambda: AgglomerativeClustering(n_clusters=K, linkage="ward"),
    "DBSCAN": lambda: DBSCAN(eps=en_iyi[0], min_samples=en_iyi[1]),
    "GMM": lambda: GaussianMixture(n_components=K, covariance_type="full", random_state=RS),
}

satirlar, etiketler = [], {}
for ad, uret in modeller.items():
    # Süre: 5 çalıştırmanın ortalaması
    sureler = []
    for _ in range(5):
        t0 = time.perf_counter()
        m = uret(); lab = m.fit_predict(X)
        sureler.append(time.perf_counter() - t0)
    etiketler[ad] = lab

    gurultu = lab == -1
    n_kume = len(set(lab)) - (1 if -1 in lab else 0)
    Xe, le = X[~gurultu], lab[~gurultu]          # DBSCAN'de gürültü noktaları ölçüte dahil edilmez
    ok = len(set(le)) >= 2
    satirlar.append({
        "Model": ad,
        "Küme Sayısı": n_kume,
        "Gürültü": int(gurultu.sum()),
        "Silhouette": silhouette_score(Xe, le) if ok else np.nan,
        "Davies-Bouldin": davies_bouldin_score(Xe, le) if ok else np.nan,
        "Calinski-Harabasz": calinski_harabasz_score(Xe, le) if ok else np.nan,
        "ARI": adjusted_rand_score(y_gercek, lab),
        "NMI": normalized_mutual_info_score(y_gercek, lab),
        "Süre (ms)": np.mean(sureler) * 1000,
    })

sonuc = pd.DataFrame(satirlar).round(4)
print("\n=== Kümeleme Sonuçları ===")
print(sonuc.to_string(index=False))
sonuc.to_csv("kumeleme_sonuclari.csv", index=False)

# 6) Görselleştirme: PCA ile 2 boyuta indirip her algoritmanın kümelerini çiz
Z = PCA(n_components=2, random_state=RS).fit_transform(X)
fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharex=True, sharey=True)
axes[0].scatter(Z[:, 0], Z[:, 1], c=y_gercek, cmap="viridis", s=22)
axes[0].set_title("Gerçek Sınıflar")
for ax, (ad, lab) in zip(axes[1:], etiketler.items()):
    ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap="viridis", s=22)
    s = sonuc.loc[sonuc.Model == ad, "Silhouette"].values[0]
    ax.set_title(f"{ad}\nSilhouette = {s:.3f}")
for ax in axes: ax.set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.tight_layout(); plt.savefig("kume_gorsel.png", dpi=150); plt.close()

# 7) Dendrogram (Hiyerarşik kümeleme için)
plt.figure(figsize=(10, 4))
dendrogram(linkage(X, method="ward"), truncate_mode="lastp", p=20, no_labels=True)
plt.title("Dendrogram (Ward)"); plt.ylabel("Mesafe")
plt.tight_layout(); plt.savefig("dendrogram.png", dpi=150); plt.close()

print("\nKaydedilen dosyalar: kumeleme_sonuclari.csv, k_secimi.png, kume_gorsel.png, dendrogram.png")
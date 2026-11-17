# -*- coding: utf-8 -*-
"""
Yapay Zeka Teknikleri - Anomali Tespiti (Anomaly Detection) Projesi
Veri seti : Healthcare_Providers.csv (Medicare sağlık hizmeti sağlayıcı kayıtları, 100.000 satır)
Modeller  : Isolation Forest, Local Outlier Factor (LOF), One-Class SVM
Not       : Veride gerçek "anomali" etiketi yok. Bu yüzden iki değerlendirme yapılır:
            A) Gerçek veri  -> anomali sayısı, algoritmalar arası uyum, süre, anomali profili
            B) Doğrulama    -> verinin içine yapay anomaliler eklenir; Precision/Recall/F1/ROC-AUC hesaplanır
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
from scipy.stats import spearmanr

RS = 42
N_ORNEK = 10000          # LOF ve One-Class SVM 100.000 satırda çok yavaş olduğu için rastgele örneklem
CONTAM = 0.02            # beklenen anomali oranı (%2)
DOSYA = "Healthcare_Providers.csv"

# ------------------------------------------------------------------ 1) Veri
df = pd.read_csv(DOSYA)
print("Ham veri boyutu:", df.shape)

SAYISAL = ["Number of Services",
           "Number of Medicare Beneficiaries",
           "Number of Distinct Medicare Beneficiary/Per Day Services",
           "Average Medicare Allowed Amount",
           "Average Submitted Charge Amount",
           "Average Medicare Payment Amount",
           "Average Medicare Standardized Amount"]
for c in SAYISAL:   # bazı dosyalarda sayılar metin olarak gelebilir (ör. "1,234")
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
df = df.dropna(subset=SAYISAL)
df = df[(df[SAYISAL] > 0).all(axis=1)]                 # log için sıfır/negatifleri at
df = df.sample(n=min(N_ORNEK, len(df)), random_state=RS).reset_index(drop=True)
print("Kullanılan örneklem:", df.shape)

def ozellik_uret(d):
    """Sayısal özellikler + türetilmiş oranlar + ikili kolonlar. Çarpık dağılım için log1p."""
    f = pd.DataFrame(index=d.index)
    for c in SAYISAL:
        f[c] = np.log1p(d[c])
    f["Ücret/İzin Verilen Oran"] = np.log1p(d["Average Submitted Charge Amount"] / d["Average Medicare Allowed Amount"])
    f["Ödeme/İzin Verilen Oran"] = d["Average Medicare Payment Amount"] / d["Average Medicare Allowed Amount"]
    f["Hizmet/Hasta Oranı"] = np.log1p(d["Number of Services"] / d["Number of Medicare Beneficiaries"])
    f["Tesis Hizmeti (F)"] = (d["Place of Service"] == "F").astype(int)
    f["İlaç Kodu (Y)"] = (d["HCPCS Drug Indicator"] == "Y").astype(int)
    return f

def modeller():
    return {
        "Isolation Forest": IsolationForest(n_estimators=200, contamination=CONTAM, random_state=RS, n_jobs=-1),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=30, contamination=CONTAM, n_jobs=-1),
        "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAM),
    }

def calistir(X):
    """Her model için (anomali=1 etiketi, anomali skoru, süre) döndürür. Skor yüksek = daha anormal."""
    sonuc = {}
    for ad, m in modeller().items():
        t0 = time.perf_counter()
        if ad == "Local Outlier Factor":
            tahmin = m.fit_predict(X)
            skor = -m.negative_outlier_factor_
        else:
            m.fit(X)
            tahmin = m.predict(X)
            skor = -m.decision_function(X)
        sure = time.perf_counter() - t0
        sonuc[ad] = (tahmin == -1).astype(int), skor, sure
    return sonuc

# ------------------------------------------------------------------ 2) A) Gerçek veri
F = ozellik_uret(df)
X = StandardScaler().fit_transform(F)
print("Özellik sayısı:", X.shape[1])

res = calistir(X)
etiket = pd.DataFrame({ad: v[0] for ad, v in res.items()})
skorlar = pd.DataFrame({ad: v[1] for ad, v in res.items()})

ozet = pd.DataFrame({
    "Model": list(res.keys()),
    "Anomali Sayısı": [int(v[0].sum()) for v in res.values()],
    "Anomali Oranı (%)": [round(100 * v[0].mean(), 2) for v in res.values()],
    "Süre (sn)": [round(v[2], 3) for v in res.values()],
})
print("\n=== A) Gerçek Veri: Genel Özet ===")
print(ozet.to_string(index=False))

# Algoritmalar arası uyum (Jaccard) ve skor korelasyonu
uyum = []
for a, b in combinations(etiket.columns, 2):
    kes = int(((etiket[a] == 1) & (etiket[b] == 1)).sum())
    bir = int(((etiket[a] == 1) | (etiket[b] == 1)).sum())
    rho = spearmanr(skorlar[a], skorlar[b])[0]
    uyum.append({"Model Çifti": f"{a} - {b}", "Ortak Anomali": kes,
                 "Jaccard": round(kes / bir, 4), "Skor Korelasyonu (Spearman)": round(rho, 4)})
uyum = pd.DataFrame(uyum)
print("\n=== Algoritmalar Arası Uyum ===")
print(uyum.to_string(index=False))

oy = etiket.sum(axis=1)
print("\nKaç algoritma tarafından anomali işaretlendi:")
print(oy.value_counts().sort_index().rename({0: "0 (normal)", 1: "1", 2: "2", 3: "3"}).to_string())

# Anomali profili: ortak anomalilerin (>=2 model) ham değerleri normallerle karşılaştırılır
ortak = oy >= 2
profil = pd.DataFrame({
    "Normal (medyan)": df.loc[~ortak, SAYISAL].median(),
    "Anomali (medyan)": df.loc[ortak, SAYISAL].median(),
}).round(2)
profil["Kat Farkı"] = (profil["Anomali (medyan)"] / profil["Normal (medyan)"]).round(2)
print(f"\n=== Anomali Profili (en az 2 model anomali dedi: {int(ortak.sum())} kayıt) ===")
print(profil.to_string())

# En şüpheli kayıtlar (ortalama sıra skoruna göre)
sira = skorlar.rank(pct=True).mean(axis=1)
en = df.assign(Oy=oy, Ortalama_Sira_Skoru=sira.round(4)).sort_values(
    ["Oy", "Ortalama_Sira_Skoru"], ascending=False).head(15)
kolon = ["Provider Type", "State Code of the Provider", "HCPCS Code", "HCPCS Description"] + SAYISAL + ["Oy", "Ortalama_Sira_Skoru"]
en[kolon].to_csv("en_supheli_kayitlar.csv", index=False)

ozet.to_csv("anomali_ozet.csv", index=False)
uyum.to_csv("anomali_uyum.csv", index=False)
profil.to_csv("anomali_profil.csv")

# ------------------------------------------------------------------ 3) B) Yapay anomali ile doğrulama
rng = np.random.default_rng(RS)
d2 = df.copy()
n_an = int(CONTAM * len(d2))
idx = rng.choice(len(d2), n_an, replace=False)
hedef = ["Number of Services", "Average Submitted Charge Amount",
         "Average Medicare Allowed Amount", "Average Medicare Payment Amount"]
for i in idx:                                    # her anomalide 1-3 kolon 5-20 kat büyütülür
    for c in rng.choice(hedef, size=rng.integers(1, 4), replace=False):
        d2.loc[i, c] = d2.loc[i, c] * rng.uniform(5, 20)
y = np.zeros(len(d2), dtype=int); y[idx] = 1

X2 = StandardScaler().fit_transform(ozellik_uret(d2))
res2 = calistir(X2)

satir = []
for ad, (tah, skor, sure) in res2.items():
    satir.append({"Model": ad,
                  "Precision": precision_score(y, tah),
                  "Recall": recall_score(y, tah),
                  "F1-Score": f1_score(y, tah),
                  "ROC-AUC": roc_auc_score(y, skor),
                  "Süre (sn)": sure})
dog = pd.DataFrame(satir).round(4)
print(f"\n=== B) Doğrulama (yapay anomali: {n_an} kayıt, %{CONTAM*100:.0f}) ===")
print(dog.to_string(index=False))
dog.to_csv("anomali_dogrulama.csv", index=False)

# ------------------------------------------------------------------ 4) Grafikler
Z = PCA(n_components=2, random_state=RS).fit_transform(X)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
for a, ad in zip(ax, etiket.columns):
    m = etiket[ad].values == 1
    a.scatter(Z[~m, 0], Z[~m, 1], s=6, alpha=0.4, label="Normal")
    a.scatter(Z[m, 0], Z[m, 1], s=14, c="red", label="Anomali")
    a.set_title(f"{ad}\n({int(m.sum())} anomali)"); a.set_xlabel("PC1")
ax[0].set_ylabel("PC2"); ax[0].legend()
plt.tight_layout(); plt.savefig("anomali_pca.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
vc = oy[oy > 0].value_counts().reindex([1, 2, 3], fill_value=0)
plt.bar(["1 model", "2 model", "3 model"], vc.values, color=["tab:gray", "tab:orange", "tab:red"])
for i, v in enumerate(vc.values): plt.text(i, v, str(v), ha="center", va="bottom")
plt.title("Anomali işaretleyen model sayısı"); plt.ylabel("Kayıt sayısı")
plt.tight_layout(); plt.savefig("anomali_uyum.png", dpi=150); plt.close()

plt.figure(figsize=(5.5, 5))
for ad, (tah, skor, sure) in res2.items():
    fpr, tpr, _ = roc_curve(y, skor)
    plt.plot(fpr, tpr, label=f"{ad} (AUC={roc_auc_score(y, skor):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Eğrileri (yapay anomali doğrulaması)"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("anomali_roc.png", dpi=150); plt.close()

print("\nKaydedilen dosyalar: anomali_ozet.csv, anomali_uyum.csv, anomali_profil.csv, "
      "anomali_dogrulama.csv, en_supheli_kayitlar.csv, anomali_pca.png, anomali_uyum.png, anomali_roc.png")
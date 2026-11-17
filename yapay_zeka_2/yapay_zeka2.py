# -*- coding: utf-8 -*-
"""
Yapay Zeka Teknikleri - Regresyon Projesi
Veri seti : Possum (104 örnek)
Hedef     : totlngth (possumun toplam vücut uzunluğu, cm)
Modeller  : Ridge Regression, KNN Regressor, Gradient Boosting Regressor
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

# 1) Veriyi oku
df = pd.read_csv("possum.csv")
print("Veri boyutu:", df.shape)
print("Eksik değerler:\n", df.isna().sum()[df.isna().sum() > 0], "\n")

# 2) Gereksiz kolonları çıkar (case = sıra numarası, site = Pop ile aynı bilgiyi taşıyor)
df = df.drop(columns=["case", "site"])

# 3) Özellik / hedef ayrımı
TARGET = "totlngth"
X = df.drop(columns=[TARGET])
y = df[TARGET]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()   # Pop, sex

# 4) Ön işleme: eksikleri medyan ile doldur, sayısalları standartlaştır, kategorikleri one-hot yap
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols),
])

# 5) %75 eğitim / %25 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE)

# 6) Modeller
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE),
}

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, preds = [], {}

for name, reg in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", reg)])
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    preds[name] = p
    cv_r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2").mean()
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, p),
        "MSE": mean_squared_error(y_test, p),
        "RMSE": np.sqrt(mean_squared_error(y_test, p)),
        "R2": r2_score(y_test, p),
        "5-Fold CV R2": cv_r2,
    })

results = pd.DataFrame(rows).round(4)
print("=== Test Kümesi Sonuçları ===")
print(results.to_string(index=False))
results.to_csv("regresyon2_sonuclari.csv", index=False)

# 7) Grafik: gerçek vs tahmin
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
lo, hi = y.min() - 1, y.max() + 1
for ax, (name, p) in zip(axes, preds.items()):
    ax.scatter(y_test, p, alpha=0.8)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    r2 = results.loc[results.Model == name, "R2"].values[0]
    ax.set_title(f"{name}\nR² = {r2:.3f}")
    ax.set_xlabel("Gerçek değer (cm)")
axes[0].set_ylabel("Tahmin (cm)")
plt.tight_layout()
plt.savefig("gercek_vs_tahmin_2.png", dpi=150)

# 8) Gradient Boosting özellik önemleri
rf = Pipeline([("prep", preprocess), ("model", models["Gradient Boosting"])]).fit(X_train, y_train)
names = num_cols + rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values()
plt.figure(figsize=(7, 4.5))
imp.plot.barh()
plt.title("Gradient Boosting - Özellik Önemleri")
plt.tight_layout()
plt.savefig("ozellik_onemleri_2.png", dpi=150)
print("\nEn önemli 3 özellik:\n", imp.sort_values(ascending=False).head(3).round(3))
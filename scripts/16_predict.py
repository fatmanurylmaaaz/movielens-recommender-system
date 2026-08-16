# SVD ile tum kullanici-film ciftleri icin tahmin uretme

import numpy as np
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

model = np.load(SCRIPT_DIR / "svd_model.npz")
U, sigma, Vt, user_means = model["U"], model["sigma"], model["Vt"], model["user_means"]

with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)

# U * diag(sigma) * Vt -> merkezlenmis tahminler, sonra kullanici ortalamasini geri ekliyoruz
predictions = U @ np.diag(sigma) @ Vt + user_means[:, np.newaxis]

print("=== tahmin matrisi boyutu ===")
print(predictions.shape)

print("\n=== clip oncesi min/max tahmin ===")
print("min:", predictions.min(), " max:", predictions.max())

# ratingler 0.5-5.0 araliginda oldugu icin tahminleri bu araliga sikistiriyoruz
predictions = np.clip(predictions, 0.5, 5.0)

print("\n=== clip sonrasi min/max tahmin ===")
print("min:", predictions.min(), " max:", predictions.max())

# ornek: 1. kullanicinin (indeks 0) ilk 5 film icin tahminleri
print("\n=== ornek kullanici (indeks 0) ilk 5 film tahmini ===")
print(predictions[0, :5])

np.save(SCRIPT_DIR / "predictions.npy", predictions)
print("\nKaydedildi: predictions.npy")

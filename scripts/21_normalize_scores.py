# CF ve content-based skorlarini ortak olcege (0-1) normalize etme

import importlib.util
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename, SCRIPT_DIR / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


content_based = load_module("12_content_based_recommend")

predictions = np.load(SCRIPT_DIR / "predictions.npy")
with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)
with open(SCRIPT_DIR / "user_profiles.pkl", "rb") as f:
    user_profiles = pickle.load(f)

USER_ID = 1
user_idx = mappers["user_mapper"][USER_ID]

# CF: bu kullanicinin TUM filmler icin ham tahminleri
cf_scores = predictions[user_idx]

# content-based: kullanici profili ile TUM filmlerin tfidf vektorleri arasindaki benzerlik
profile = user_profiles[USER_ID].reshape(1, -1)
content_scores = cosine_similarity(profile, content_based.tfidf_matrix).flatten()

print("=== normalizasyon ONCESI ham skor istatistikleri ===")
print(f"CF      -> min: {cf_scores.min():.3f}  max: {cf_scores.max():.3f}  ortalama: {cf_scores.mean():.3f}")
print(f"Content -> min: {content_scores.min():.3f}  max: {content_scores.max():.3f}  ortalama: {content_scores.mean():.3f}")


def min_max_normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())


cf_norm = min_max_normalize(cf_scores)
content_norm = min_max_normalize(content_scores)

print("\n=== normalizasyon SONRASI (ikisi de 0-1 araliginda) ===")
print(f"CF      -> min: {cf_norm.min():.3f}  max: {cf_norm.max():.3f}  ortalama: {cf_norm.mean():.3f}")
print(f"Content -> min: {content_norm.min():.3f}  max: {content_norm.max():.3f}  ortalama: {content_norm.mean():.3f}")

np.save(SCRIPT_DIR / "cf_scores_example.npy", cf_norm)
np.save(SCRIPT_DIR / "content_scores_example.npy", content_norm)
print("\nOrnek normalize skorlar kaydedildi (bir sonraki adimda hybrid formulunu test etmek icin)")

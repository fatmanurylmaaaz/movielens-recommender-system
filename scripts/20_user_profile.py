# Kullanici profil vektoru: content-based'i kullaniciya uyarlama

import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename, SCRIPT_DIR / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


content_based = load_module("12_content_based_recommend")
movies = content_based.movies
tfidf_matrix = content_based.tfidf_matrix

ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

# movieId -> tfidf_matrix satir indeksi (movies_processed.csv'deki sira ile ayni)
movie_id_to_row = {mid: i for i, mid in enumerate(movies["movieId"])}


def build_user_profile(user_id):
    user_ratings = ratings[ratings["userId"] == user_id]
    rows = [movie_id_to_row[mid] for mid in user_ratings["movieId"] if mid in movie_id_to_row]
    weights = user_ratings[user_ratings["movieId"].isin(movie_id_to_row)]["rating"].values

    if len(rows) == 0:
        return None

    # kullanicinin izledigi filmlerin tfidf vektorlerini rating agirlikli ortalamasini al
    movie_vectors = tfidf_matrix[rows]
    weighted_sum = movie_vectors.T.dot(weights)
    profile = weighted_sum / weights.sum()

    # cosine similarity hesaplarken tutarli olmasi icin profil vektorunu de normalize ediyoruz
    profile = normalize(profile.reshape(1, -1))
    return profile


# ornek: userId=1 icin profil olustur ve en baskin kelimelere bakalim
USER_ID = 1
profile = build_user_profile(USER_ID)

print(f"=== userId {USER_ID} profil vektoru boyutu ===")
print(profile.shape)

# TfidfVectorizer'i tekrar kurmadan feature isimlerine erismek icin 10_tfidf.py'deki
# ayni vectorizer mantigini kisaca tekrar calistiriyoruz (kelime listesi ayni olmali)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=2)
vectorizer.fit_transform(movies["content_text"].fillna(""))
feature_names = vectorizer.get_feature_names_out()

top_indices = profile.flatten().argsort()[::-1][:15]
print(f"\n=== userId {USER_ID} profilinde en baskin 15 kelime ===")
for i in top_indices:
    if profile.flatten()[i] > 0:
        print(f"{feature_names[i]}: {profile.flatten()[i]:.4f}")

# tum kullanicilar icin profil vektorlerini onceden hesaplayip kaydedelim (sonraki adimda lazim)
all_user_ids = ratings["userId"].unique()
profiles = {}
for uid in all_user_ids:
    p = build_user_profile(uid)
    if p is not None:
        profiles[uid] = p.flatten()

import pickle
with open(SCRIPT_DIR / "user_profiles.pkl", "wb") as f:
    pickle.dump(profiles, f)

print(f"\n{len(profiles)} kullanici icin profil vektoru kaydedildi: user_profiles.pkl")

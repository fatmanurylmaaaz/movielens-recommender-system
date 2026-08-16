# Content-based ve Collaborative Filtering modellerini karsilastirma

import importlib.util
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"


def load_module(filename):
    # dosya adi "12_content_based_recommend" gibi rakamla basladigi icin
    # normal import calismiyor, dosya yolundan dinamik olarak yukluyoruz
    spec = importlib.util.spec_from_file_location(filename, SCRIPT_DIR / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


content_based = load_module("12_content_based_recommend")
cf = load_module("18_recommend_cf")

all_ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

USER_ID = 1

# bu kullanicinin en yuksek puan verdigi filmi bul -> content-based icin referans noktasi
user_ratings = all_ratings[all_ratings["userId"] == USER_ID].merge(
    pd.read_csv(f"{DATA_DIR}/movies.csv"), on="movieId"
)
favorite = user_ratings.sort_values("rating", ascending=False).iloc[0]

print(f"userId {USER_ID}'in en yuksek puan verdigi film: '{favorite['title']}' ({favorite['rating']} yildiz)")

print("\n" + "=" * 60)
print(f"CONTENT-BASED: '{favorite['title']}' filmine benzer filmler")
print("=" * 60)
content_based.recommend(favorite["title"].split(" (")[0], k=5)

print("\n" + "=" * 60)
print(f"COLLABORATIVE FILTERING: userId {USER_ID} icin oneriler")
print("=" * 60)
cf.recommend_for_user(USER_ID, k=5)

# iki listenin ne kadar ortusuyor oldugunu kontrol edelim
cb_matches = content_based.find_movie(favorite["title"].split(" (")[0], content_based.movies)
cb_idx = cb_matches.index[0]
sim_scores = content_based.cosine_similarity(content_based.tfidf_matrix[cb_idx], content_based.tfidf_matrix).flatten()
cb_top_ids = set(
    content_based.movies.loc[i, "movieId"] for i in sim_scores.argsort()[::-1] if i != cb_idx
)
cb_top_ids = list(cb_top_ids)[:5]

user_idx = cf.user_mapper[USER_ID]
already_watched = set(all_ratings[all_ratings["userId"] == USER_ID]["movieId"])
cf_top_ids = [
    cf.movie_inv_mapper[i]
    for i in cf.predictions[user_idx].argsort()[::-1]
    if cf.movie_inv_mapper[i] not in already_watched
][:5]

overlap = set(cb_top_ids) & set(cf_top_ids)
print("\n" + "=" * 60)
print(f"Ortak onerilen film sayisi: {len(overlap)} / 5")
print("=" * 60)

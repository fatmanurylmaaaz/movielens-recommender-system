# SVD tabanli collaborative filtering oneri fonksiyonu

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "ml-latest-small"

predictions = np.load(SCRIPT_DIR / "predictions.npy")
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
all_ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")  # train+test hepsi -> zaten izlenenleri bulmak icin

with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)
user_mapper = mappers["user_mapper"]
movie_inv_mapper = mappers["movie_inv_mapper"]


def recommend_for_user(user_id, k=10):
    if user_id not in user_mapper:
        print(f"userId {user_id} modelde yok (cold start kullanici).")
        return

    user_idx = user_mapper[user_id]
    user_predictions = predictions[user_idx]  # bu kullanicinin TUM filmler icin tahminleri

    already_watched = set(all_ratings[all_ratings["userId"] == user_id]["movieId"])

    # tahminleri buyukten kucuge sirala, zaten izlenenleri atla
    top_movie_indices = user_predictions.argsort()[::-1]

    print(f"\n=== userId {user_id} icin oneriler ===")
    print(f"(bu kullanici zaten {len(already_watched)} film puanlamis)")
    shown = 0
    for movie_idx in top_movie_indices:
        movie_id = movie_inv_mapper[movie_idx]
        if movie_id in already_watched:
            continue
        title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
        score = user_predictions[movie_idx]
        print(f"  - {title}  (tahmini puan: {score:.2f})")
        shown += 1
        if shown == k:
            break


if __name__ == "__main__":
    for uid in [1, 50, 414]:
        recommend_for_user(uid, k=5)

"""
Hybrid Recommender - MovieLens (ml-latest-small)

content_based.py (tur/etiket benzerligi) ile collaborative_filtering.py
(kullanici davranisi benzerligi) skorlarini agirlikli birlestirir.
Ikisinin de tek basina eksigini kapatir: content-based yeni/az puanlanan
filmleri de onerebilir, CF ise sadece populer turler yerine gercek zevke gore onerir.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from content_based import load_data as load_movies_with_tags, build_tfidf_matrix
from collaborative_filtering import build_matrix

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = "../dataset/ml-latest-small"
TOP_N = 10
ALPHA = 0.5  # 1.0 = tamamen CF, 0.0 = tamamen content-based
MIN_RATINGS = 20  # bu esigin altinda puanlanan filmler onerilmez (gurultulu benzerlik skorlari abartilmasin diye)


def load_data():
    movies = load_movies_with_tags()  # movieId, title, genres, tag
    ratings = pd.read_csv(f"{DATASET_DIR}/ratings.csv")
    return movies, ratings


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    # CF ve content skorlari farkli olceklerde oldugu icin birlestirmeden once 0-1'e cekilir
    rng = scores.max() - scores.min()
    return (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores)


def content_scores(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame, tfidf_matrix) -> np.ndarray:
    # kullanicinin puanladigi filmlerin TF-IDF vektorlerinin rating-agirlikli ortalamasi -> "zevk profili"
    movie_id_to_row = {mid: i for i, mid in enumerate(movies["movieId"])}
    user_ratings = ratings[ratings["userId"] == user_id]

    rows, weights = [], []
    for mid, r in zip(user_ratings["movieId"], user_ratings["rating"]):
        if mid in movie_id_to_row:
            rows.append(movie_id_to_row[mid])
            weights.append(r)

    if not rows:
        return np.zeros(movies.shape[0])

    weights = np.array(weights)
    profile = tfidf_matrix[rows].T.dot(weights) / weights.sum()
    profile = np.asarray(profile).reshape(1, -1)
    return cosine_similarity(profile, tfidf_matrix).flatten()


def cf_scores(user_id: int, ratings: pd.DataFrame, X, movie_mapper: dict, movie_inv_mapper: dict) -> dict:
    # item-based CF tahmin formulu: pred(i) = sum(sim(i,j)*rating(u,j)) / sum(|sim(i,j)|), j = kullanicinin puanladigi filmler
    user_ratings = ratings[ratings["userId"] == user_id]
    rated = [(movie_mapper[mid], r) for mid, r in zip(user_ratings["movieId"], user_ratings["rating"]) if mid in movie_mapper]
    if not rated:
        return {}

    rated_idx, rated_vals = zip(*rated)
    sims = cosine_similarity(X[list(rated_idx)], X)  # (kullanicinin puanladigi film sayisi, tum filmler)
    numerator = np.array(rated_vals) @ sims
    denominator = np.abs(sims).sum(axis=0) + 1e-9
    scores = numerator / denominator
    return {movie_inv_mapper[i]: s for i, s in enumerate(scores)}


def recommend_hybrid(user_id: int, movies: pd.DataFrame, ratings: pd.DataFrame, tfidf_matrix,
                      X, movie_mapper: dict, movie_inv_mapper: dict, k: int = TOP_N, alpha: float = ALPHA) -> pd.DataFrame:
    c_scores = min_max_normalize(content_scores(user_id, ratings, movies, tfidf_matrix))

    cf_dict = cf_scores(user_id, ratings, X, movie_mapper, movie_inv_mapper)
    f_scores = movies["movieId"].map(cf_dict).fillna(0).values
    f_scores = min_max_normalize(f_scores)

    combined = alpha * f_scores + (1 - alpha) * c_scores

    watched = set(ratings[ratings["userId"] == user_id]["movieId"])
    rating_counts = ratings.groupby("movieId").size()
    well_known = set(rating_counts[rating_counts >= MIN_RATINGS].index)

    result = movies[["movieId", "title"]].copy()
    result["score"] = combined
    result = result[~result["movieId"].isin(watched) & result["movieId"].isin(well_known)]
    return result.sort_values("score", ascending=False).head(k)[["title", "score"]]


def main():
    movies, ratings = load_data()
    tfidf_matrix = build_tfidf_matrix(movies)
    X, movie_mapper, movie_inv_mapper = build_matrix(ratings)

    for user_id in [1, 414]:
        print(f"===== Kullanıcı {user_id} için Hybrid Öneri (alpha={ALPHA}) =====")
        result = recommend_hybrid(user_id, movies, ratings, tfidf_matrix, X, movie_mapper, movie_inv_mapper)
        print(result.to_string(index=False))
        print()


if __name__ == "__main__":
    main()

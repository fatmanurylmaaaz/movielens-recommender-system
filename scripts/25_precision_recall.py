# Precision@K / Recall@K ile CF, content-based ve hybrid modellerini nicel karsilastirma

import importlib.util
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

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

predictions = np.load(SCRIPT_DIR / "predictions.npy")
with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)
user_mapper = mappers["user_mapper"]
movie_mapper = mappers["movie_mapper"]
with open(SCRIPT_DIR / "user_profiles.pkl", "rb") as f:
    user_profiles = pickle.load(f)

all_ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
train_ratings = pd.read_csv(SCRIPT_DIR / "ratings_train.csv")
test_ratings = pd.read_csv(SCRIPT_DIR / "ratings_test.csv")
train_movies = set(train_ratings["movieId"])

# ayni "az bilinen film" dersini burada da uyguluyoruz (adim 5 / adim 22)
MIN_RATINGS = 5
rating_counts = all_ratings.groupby("movieId").size()
well_known_movies = set(rating_counts[rating_counts >= MIN_RATINGS].index)

canonical_ids = movies["movieId"].values
# degerlendirmeye sadece hem yeterince bilinen HEM train'de gorulen filmleri aliyoruz
# (boylece CF ve content-based tam olarak ayni film havuzu uzerinden kiyaslanmis olur)
eval_mask = np.array([mid in well_known_movies and mid in train_movies for mid in canonical_ids])
eval_indices = np.where(eval_mask)[0]
eval_movie_ids = canonical_ids[eval_indices]
cf_cols_for_eval = np.array([movie_mapper[mid] for mid in eval_movie_ids])

print(f"Degerlendirmeye dahil edilen film sayisi: {len(eval_indices)} / {len(canonical_ids)}")

user_ids_sorted = sorted(user_mapper, key=lambda u: user_mapper[u])
profile_matrix = np.vstack(
    [user_profiles.get(uid, np.zeros(tfidf_matrix.shape[1])) for uid in user_ids_sorted]
)

# content ve CF skorlarini TUM kullanicilar x eval filmleri icin tek seferde hesapliyoruz
content_scores_full = cosine_similarity(profile_matrix, tfidf_matrix[eval_indices])
cf_scores_full = predictions[:, cf_cols_for_eval]


def min_max_rows(mat):
    mins = mat.min(axis=1, keepdims=True)
    maxs = mat.max(axis=1, keepdims=True)
    rng = np.where(maxs - mins == 0, 1, maxs - mins)
    return (mat - mins) / rng


cf_norm_full = min_max_rows(cf_scores_full)
content_norm_full = min_max_rows(content_scores_full)


def precision_recall_at_k(score_matrix, k=10, relevance_threshold=4.0):
    precisions, recalls = [], []
    for uid in user_ids_sorted:
        u_idx = user_mapper[uid]
        user_test = test_ratings[
            (test_ratings["userId"] == uid) & (test_ratings["rating"] >= relevance_threshold)
        ]
        relevant = set(user_test["movieId"]) & set(eval_movie_ids)
        if len(relevant) == 0:
            continue

        scores = score_matrix[u_idx].copy()
        watched = set(train_ratings[train_ratings["userId"] == uid]["movieId"])
        watched_mask = np.isin(eval_movie_ids, list(watched))
        scores[watched_mask] = -np.inf

        top_k_idx = np.argsort(scores)[::-1][:k]
        recommended = set(eval_movie_ids[top_k_idx])

        hit = len(recommended & relevant)
        precisions.append(hit / k)
        recalls.append(hit / len(relevant))

    return np.mean(precisions), np.mean(recalls), len(precisions)


K = 10
print(f"\n=== Precision@{K} / Recall@{K} (relevance esigi: rating >= 4.0) ===")
for name, matrix in [("CF (SVD)", cf_norm_full), ("Content-based", content_norm_full)]:
    p, r, n = precision_recall_at_k(matrix, k=K)
    print(f"{name:15s} -> Precision@{K}: {p:.4f}  Recall@{K}: {r:.4f}  (degerlendirilen kullanici: {n})")

np.savez(
    SCRIPT_DIR / "eval_matrices.npz",
    cf_norm_full=cf_norm_full,
    content_norm_full=content_norm_full,
    eval_movie_ids=eval_movie_ids,
)

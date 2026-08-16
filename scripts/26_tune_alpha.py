# Hybrid'in alpha degerini Precision@K ile ayarlama

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

data = np.load(SCRIPT_DIR / "eval_matrices.npz")
cf_norm_full = data["cf_norm_full"]
content_norm_full = data["content_norm_full"]
eval_movie_ids = data["eval_movie_ids"]

with open(SCRIPT_DIR / "cf_mappers.pkl", "rb") as f:
    mappers = pickle.load(f)
user_mapper = mappers["user_mapper"]

train_ratings = pd.read_csv(SCRIPT_DIR / "ratings_train.csv")
test_ratings = pd.read_csv(SCRIPT_DIR / "ratings_test.csv")
user_ids_sorted = sorted(user_mapper, key=lambda u: user_mapper[u])


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
print(f"=== alpha taramasi (Precision@{K} / Recall@{K}, alpha=1.0 -> sadece CF, alpha=0.0 -> sadece content) ===")
best_alpha, best_p = None, -1
for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    hybrid_matrix = alpha * cf_norm_full + (1 - alpha) * content_norm_full
    p, r, _ = precision_recall_at_k(hybrid_matrix, k=K)
    if p > best_p:
        best_p, best_alpha = p, alpha
    print(f"alpha={alpha:.2f}  ->  Precision@{K}: {p:.4f}  Recall@{K}: {r:.4f}")

print(f"\nEn iyi alpha: {best_alpha} (Precision@{K}: {best_p:.4f})")

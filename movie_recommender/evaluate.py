"""Degerlendirme metrikleri: RMSE, MAE, Precision@K, Recall@K (scripts/17, 25-26)."""

import numpy as np
from . import config


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def precision_recall_at_k(
    score_matrix, eval_movie_ids, user_mapper, train_ratings, test_ratings,
    k=10, relevance_threshold=config.RELEVANCE_THRESHOLD,
):
    """score_matrix: (n_users, len(eval_movie_ids)) boyutunda, satirlari user_mapper
    sirasina gore dizilmis skor matrisi (CF, content ya da hybrid olabilir)."""
    user_ids_sorted = sorted(user_mapper, key=lambda u: user_mapper[u])
    precisions, recalls = [], []

    for uid in user_ids_sorted:
        u_idx = user_mapper[uid]
        user_test = test_ratings[
            (test_ratings["userId"] == uid) & (test_ratings["rating"] >= relevance_threshold)
        ]
        relevant = set(user_test["movieId"]) & set(eval_movie_ids)
        if not relevant:
            continue

        scores = score_matrix[u_idx].copy()
        watched = set(train_ratings[train_ratings["userId"] == uid]["movieId"])
        watched_mask = np.isin(eval_movie_ids, list(watched))
        scores[watched_mask] = -np.inf

        top_k_idx = np.argsort(scores)[::-1][:k]
        recommended = set(np.asarray(eval_movie_ids)[top_k_idx])

        hit = len(recommended & relevant)
        precisions.append(hit / k)
        recalls.append(hit / len(relevant))

    return np.mean(precisions), np.mean(recalls), len(precisions)
